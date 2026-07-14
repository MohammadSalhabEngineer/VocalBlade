"""
Video frame interpolation at edit cut boundaries.

For each boundary in the concatenated video, extracts the last frame before
the cut and the first frame after, runs the selected method, and replaces
the boundary frame with the interpolated result. All operations preserve
A/V sync — we replace existing frames, never insert new ones.

Methods:
  NONE  — no-op, returns input path unchanged
  BLEND — ffmpeg minterpolate (motion-compensated, no model download needed)
  RIFE  — RIFE v4.25 neural optical flow interpolation (local vendored weights)
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)

# RIFE model cache (module-level singleton)
_rife_model = None
_rife_device: Optional[str] = None

# Path to vendored RIFE weights (already on disk — no download needed)
_RIFE_WEIGHTS = os.path.join(os.path.dirname(__file__), "rife_model", "flownet.pkl")

# Skip RIFE interpolation when frames are too dissimilar (avoids ghosting).
# Speech videos: same-scene consecutive frames typically score > 0.85.
# Threshold at 0.70 — only interpolate when the speaker is in nearly the same position.
SSIM_SKIP_THRESHOLD = 0.70


def _load_rife_model(device: str):
    """Lazy-load and cache the RIFE IFNet model from local vendored weights."""
    global _rife_model, _rife_device
    if _rife_model is not None and _rife_device == device:
        return _rife_model

    import torch
    from .rife_model.IFNet_HDv3 import IFNet

    model = IFNet()
    state = torch.load(_RIFE_WEIGHTS, map_location=device)
    # Handle state dict wrapped under 'model' key
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Strip DataParallel 'module.' prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    if device == "cuda":
        import torch
        if torch.cuda.is_available():
            model = model.half().cuda()
    _rife_model = model
    _rife_device = device
    logger.info(f"RIFE model loaded on {device}")
    return model


def _compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute a simple structural similarity index between two BGR uint8 frames."""
    import cv2
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mu_a, mu_b = gray_a.mean(), gray_b.mean()
    sigma_a = float(np.sqrt(((gray_a - mu_a) ** 2).mean()))
    sigma_b = float(np.sqrt(((gray_b - mu_b) ** 2).mean()))
    sigma_ab = float(((gray_a - mu_a) * (gray_b - mu_b)).mean())
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2)
    return float(np.clip(num / (den + 1e-10), 0.0, 1.0))


def _frame_to_tensor(frame: np.ndarray, device: str, half: bool):
    """Convert BGR uint8 HxWx3 frame to BCHW float tensor in [0, 1]."""
    import torch
    import cv2
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0)  # (1, 3, H, W)
    if half and device == "cuda":
        t = t.half()
    return t.to(device)


def _tensor_to_frame(tensor) -> np.ndarray:
    """Convert BCHW float tensor back to BGR uint8 HxWx3."""
    import cv2
    arr = tensor.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy()
    arr = (arr * 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return bgr


def _pad_to_multiple(frame: np.ndarray, multiple: int = 32):
    """Pad frame height and width to multiples of `multiple` for RIFE."""
    import cv2
    h, w = frame.shape[:2]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return frame, (h, w)
    padded = cv2.copyMakeBorder(frame, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return padded, (h, w)


def _rife_interpolate_pair(frame_a: np.ndarray, frame_b: np.ndarray,
                            model, device: str) -> np.ndarray:
    """Run RIFE inference on a frame pair. Returns interpolated BGR uint8 frame."""
    import torch
    half = device == "cuda"

    # Pad to multiple of 32 (required for RIFE scale pyramid)
    fa_padded, orig_size = _pad_to_multiple(frame_a)
    fb_padded, _ = _pad_to_multiple(frame_b)

    ta = _frame_to_tensor(fa_padded, device, half)
    tb = _frame_to_tensor(fb_padded, device, half)
    inp = torch.cat([ta, tb], dim=1)  # (1, 6, H_pad, W_pad)

    with torch.no_grad():
        # scale_list needs 5 elements for 5 IFBlocks
        _, _, merged = model(inp, timestep=0.5, scale_list=[8, 4, 2, 1, 1])

    result_tensor = merged[4]  # (1, 3, H_pad, W_pad) — final blended frame
    result = _tensor_to_frame(result_tensor)

    # Crop back to original size
    h, w = orig_size
    return result[:h, :w]


def _rife_process_boundaries(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    device: str,
) -> str:
    """
    Replace one frame at each cut boundary with a RIFE-interpolated frame.

    Frame at boundary index is replaced with RIFE(frame[idx-1], frame[idx]).
    SSIM gating: if frames are too dissimilar (scene change, large head movement),
    skip interpolation to avoid ghosting artifacts and keep the hard cut.
    """
    import cv2
    import torch

    try:
        model = _load_rife_model(device)
    except Exception as e:
        logger.error(f"RIFE model load failed: {e}")
        return video_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return video_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return video_path

    # Build replacement map: {frame_index: interpolated_frame}
    replacement_frames: Dict[int, np.ndarray] = {}

    for ts in boundary_timestamps:
        boundary_idx = int(round(ts * fps))
        prev_idx = max(0, boundary_idx - 1)
        curr_idx = min(total_frames - 1, boundary_idx)

        if prev_idx == curr_idx:
            continue

        # Read frame before boundary
        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ret_a, frame_a = cap.read()
        # Read frame at boundary (first frame of next segment)
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_idx)
        ret_b, frame_b = cap.read()

        if not (ret_a and ret_b):
            logger.warning(f"RIFE: could not read frames at boundary {ts:.2f}s")
            continue

        ssim = _compute_ssim(frame_a, frame_b)
        if ssim < SSIM_SKIP_THRESHOLD:
            logger.info(f"RIFE: boundary {ts:.2f}s SSIM={ssim:.3f} < {SSIM_SKIP_THRESHOLD}, keeping hard cut")
            continue

        try:
            interp = _rife_interpolate_pair(frame_a, frame_b, model, device)
            replacement_frames[curr_idx] = interp
            logger.info(f"RIFE: interpolated boundary at {ts:.2f}s (SSIM={ssim:.3f})")
        except Exception as e:
            logger.warning(f"RIFE inference failed at {ts:.2f}s: {e}")

    cap.release()

    if not replacement_frames:
        logger.info("RIFE: no frames to replace (all boundaries skipped)")
        return video_path

    # Pipe raw BGR frames directly to ffmpeg — avoids codec availability issues with cv2.VideoWriter.
    # Single-pass: takes piped video + original audio in one command.
    fd, output_path = tempfile.mkstemp(suffix='.mkv')
    os.close(fd)
    temp_files.append(output_path)

    ffmpeg_proc = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo',
         '-pixel_format', 'bgr24',
         '-video_size', f'{width}x{height}',
         '-framerate', str(fps),
         '-i', 'pipe:0',       # video frames piped from Python
         '-i', video_path,     # original concat video — audio track source
         '-map', '0:v:0',
         '-map', '1:a:0',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
         '-pix_fmt', 'yuv420p',   # required for broad player compatibility
         '-c:a', 'copy',
         output_path],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    cap2 = cv2.VideoCapture(video_path)
    for frame_idx in range(total_frames):
        ret, frame = cap2.read()
        if not ret:
            break
        out_frame = replacement_frames[frame_idx] if frame_idx in replacement_frames else frame
        ffmpeg_proc.stdin.write(out_frame.tobytes())
    cap2.release()
    ffmpeg_proc.stdin.close()
    try:
        ffmpeg_proc.wait(timeout=600)
    except subprocess.TimeoutExpired:
        ffmpeg_proc.kill()
        ffmpeg_proc.wait()
        logger.error("RIFE frame write timed out")
        return video_path
    stderr = ffmpeg_proc.stderr.read()

    if ffmpeg_proc.returncode != 0:
        logger.error(f"RIFE frame write failed: {stderr.decode()[:400]}")
        return video_path

    logger.info(f"RIFE: processed {len(replacement_frames)} boundaries → {output_path}")
    return output_path


def _probe_fps(video_path: str) -> float:
    """Probe video FPS using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', video_path],
        capture_output=True, text=True, timeout=30
    )
    raw = result.stdout.strip()
    if '/' in raw:
        num, den = raw.split('/')
        return float(num) / max(float(den), 1.0)
    try:
        return float(raw)
    except Exception:
        return 30.0


def _probe_duration(video_path: str) -> float:
    """Probe video duration in seconds."""
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', video_path],
        capture_output=True, text=True, timeout=30
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _blend_minterpolate(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    window_sec: float = 0.5,
) -> str:
    """
    Apply ffmpeg minterpolate around each cut boundary.

    For each boundary at time T, extracts a [T-window, T+window] clip,
    applies motion-compensated interpolation, then reassembles the full
    video: [0..T-window] + [smoothed] + [T+window..end].

    Processes boundaries right-to-left to avoid timestamp shifts.
    """
    fps = _probe_fps(video_path)
    total_dur = _probe_duration(video_path)

    if total_dur <= 0 or fps <= 0:
        return video_path

    current_path = video_path

    # Process boundaries from last to first to avoid timestamp shifts
    for ts in sorted(boundary_timestamps, reverse=True):
        clip_start = max(0.0, ts - window_sec)
        clip_end = min(total_dur, ts + window_sec)
        clip_dur = clip_end - clip_start

        if clip_dur < 0.1:
            continue

        # Extract boundary clip
        fd, clip_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(clip_path)

        extract_result = subprocess.run(
            ['ffmpeg', '-y',
             '-ss', str(clip_start), '-t', str(clip_dur),
             '-i', current_path,
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le',
             clip_path],
            capture_output=True, text=True, timeout=120
        )
        if extract_result.returncode != 0:
            logger.warning(f"BLEND: clip extraction failed at {ts:.2f}s")
            continue

        # Apply minterpolate to the clip
        fd2, smoothed_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd2)
        temp_files.append(smoothed_path)

        interp_result = subprocess.run(
            ['ffmpeg', '-y',
             '-i', clip_path,
             '-vf', f'minterpolate=fps={fps:.3f}:mi_mode=mci:mc_mode=aobmc:me=epzs:vsbmc=1',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le',
             smoothed_path],
            capture_output=True, text=True, timeout=180
        )
        if interp_result.returncode != 0:
            logger.warning(f"BLEND: minterpolate failed at {ts:.2f}s: {interp_result.stderr[:200]}")
            continue

        # Reassemble: [0..clip_start] + [smoothed] + [clip_end..end]
        fd3, reassembled = tempfile.mkstemp(suffix='.mkv')
        os.close(fd3)
        temp_files.append(reassembled)

        # Build concat list
        fd4, concat_list = tempfile.mkstemp(suffix='.txt')
        os.close(fd4)
        temp_files.append(concat_list)

        segments = []

        # Pre-segment: beginning to clip_start
        if clip_start > 0.1:
            fd5, pre_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd5)
            temp_files.append(pre_path)
            subprocess.run(
                ['ffmpeg', '-y',
                 '-i', current_path,
                 '-ss', '0', '-to', str(clip_start),
                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                 '-c:a', 'pcm_s16le',
                 pre_path],
                capture_output=True, text=True, timeout=120
            )
            segments.append(pre_path)

        segments.append(smoothed_path)

        # Post-segment: clip_end to end
        if clip_end < total_dur - 0.1:
            fd6, post_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd6)
            temp_files.append(post_path)
            subprocess.run(
                ['ffmpeg', '-y',
                 '-ss', str(clip_end),
                 '-i', current_path,
                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                 '-c:a', 'pcm_s16le',
                 post_path],
                capture_output=True, text=True, timeout=120
            )
            segments.append(post_path)

        if len(segments) == 1:
            # Only smoothed clip — use it directly
            current_path = smoothed_path
            total_dur = _probe_duration(current_path)
            continue

        with open(concat_list, 'w') as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        concat_result = subprocess.run(
            ['ffmpeg', '-y',
             '-f', 'concat', '-safe', '0',
             '-i', concat_list,
             '-c', 'copy',
             '-fflags', '+genpts',
             reassembled],
            capture_output=True, text=True, timeout=300
        )

        if concat_result.returncode != 0:
            logger.warning(f"BLEND: reassembly failed at {ts:.2f}s")
            continue

        current_path = reassembled
        total_dur = _probe_duration(current_path)

    if current_path == video_path:
        return video_path

    logger.info(f"BLEND: minterpolate applied to {len(boundary_timestamps)} boundaries")
    return current_path


def _frame_pad_boundaries(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    hold_frames: int = 2,
    dissolve_sec: float = 0.12,
) -> str:
    """
    Freeze-frame dissolve transition at each cut boundary.

    At each boundary timestamp T:
      1. Extract pre-cut clip ending at T; freeze its last `hold_frames` frames.
      2. Extract post-cut clip starting at T; freeze its first `hold_frames` frames.
      3. Dissolve (xfade) between the two frozen-frame clips.
      4. Reassemble the full video.

    No new frames are synthesised — only existing boundary frames are held and blended.
    The transition adds ~(hold_frames/fps)*2 seconds at each cut.
    """
    fps = _probe_fps(video_path)
    total_dur = _probe_duration(video_path)

    if total_dur <= 0 or fps <= 0:
        return video_path

    hold_sec = hold_frames / max(fps, 1.0)
    window_sec = hold_sec + dissolve_sec + 0.05   # a little extra to ensure we have frames

    current_path = video_path

    for ts in sorted(boundary_timestamps, reverse=True):
        clip_start = max(0.0, ts - window_sec)
        clip_end = min(total_dur, ts + window_sec)

        # --- Extract pre-cut clip (clip_start → ts) ---
        fd, pre_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(pre_path)
        r = subprocess.run(
            ['ffmpeg', '-y',
             '-ss', str(clip_start), '-to', str(ts),
             '-i', current_path,
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le', pre_path],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: pre-clip extract failed at {ts:.2f}s")
            continue

        # --- Hold last `hold_frames` frames of pre_clip ---
        fd, pre_held = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(pre_held)
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', pre_path,
             '-vf', f'tpad=stop={hold_frames}:stop_mode=clone',
             '-af', f'apad=pad_dur={hold_sec}',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le', pre_held],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: pre-hold failed at {ts:.2f}s")
            continue

        # --- Extract post-cut clip (ts → clip_end) ---
        fd, post_path = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(post_path)
        r = subprocess.run(
            ['ffmpeg', '-y',
             '-ss', str(ts), '-to', str(clip_end),
             '-i', current_path,
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le', post_path],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: post-clip extract failed at {ts:.2f}s")
            continue

        # --- Hold first `hold_frames` frames of post_clip (prepend cloned frames) ---
        fd, post_held = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(post_held)
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', post_path,
             '-vf', f'tpad=start={hold_frames}:start_mode=clone',
             '-af', f'adelay={int(hold_sec * 1000)}:all=1',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le', post_held],
            capture_output=True, timeout=60,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: post-hold failed at {ts:.2f}s")
            continue

        # --- xfade dissolve between pre_held and post_held ---
        pre_dur = _probe_duration(pre_held)
        xfade_offset = max(0.0, pre_dur - dissolve_sec)

        fd, transition = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(transition)
        r = subprocess.run(
            ['ffmpeg', '-y',
             '-i', pre_held, '-i', post_held,
             '-filter_complex',
             f'[0:v][1:v]xfade=transition=dissolve:duration={dissolve_sec}:offset={xfade_offset}[v];'
             f'[0:a][1:a]acrossfade=d={dissolve_sec}[a]',
             '-map', '[v]', '-map', '[a]',
             '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
             '-c:a', 'pcm_s16le', transition],
            capture_output=True, timeout=120,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: xfade failed at {ts:.2f}s: {r.stderr.decode()[:300]}")
            continue

        # --- Reassemble: [start..clip_start] + [transition] + [clip_end..end] ---
        segs = []
        if clip_start > 0.05:
            fd, pre_part = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)
            temp_files.append(pre_part)
            subprocess.run(
                ['ffmpeg', '-y', '-ss', '0', '-to', str(clip_start), '-i', current_path,
                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                 '-c:a', 'pcm_s16le', pre_part],
                capture_output=True, timeout=120,
            )
            segs.append(pre_part)

        segs.append(transition)

        if clip_end < total_dur - 0.05:
            fd, post_part = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)
            temp_files.append(post_part)
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(clip_end), '-i', current_path,
                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                 '-c:a', 'pcm_s16le', post_part],
                capture_output=True, timeout=120,
            )
            segs.append(post_part)

        if len(segs) == 1:
            current_path = transition
            total_dur = _probe_duration(current_path)
            continue

        fd, concat_txt = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        temp_files.append(concat_txt)
        with open(concat_txt, 'w') as f:
            for s in segs:
                f.write(f"file '{s}'\n")

        fd, reassembled = tempfile.mkstemp(suffix='.mkv')
        os.close(fd)
        temp_files.append(reassembled)
        r = subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', concat_txt,
             '-c', 'copy', '-fflags', '+genpts', reassembled],
            capture_output=True, timeout=300,
        )
        if r.returncode != 0:
            logger.warning(f"FRAME_PAD: reassembly failed at {ts:.2f}s")
            continue

        current_path = reassembled
        total_dur = _probe_duration(current_path)

    if current_path == video_path:
        return video_path

    logger.info(f"FRAME_PAD: applied freeze-dissolve to {len(boundary_timestamps)} boundaries")
    return current_path


def interpolate_at_cuts(
    video_path: str,
    boundary_timestamps: List[float],
    method,
    temp_files: List[str],
    device: str = "cuda",
) -> str:
    """
    Apply video frame interpolation at each cut boundary.

    Args:
        video_path: Path to concatenated .mkv video file (hard cuts)
        boundary_timestamps: List of timestamps (seconds) where segments were joined
        method: VideoInterpolationMethod enum value
        temp_files: List to append temp paths to for cleanup
        device: "cuda" or "cpu"

    Returns:
        Path to processed video (new temp file), or original path on failure/NONE.
    """
    from .types import VideoInterpolationMethod

    if method == VideoInterpolationMethod.NONE or not boundary_timestamps:
        return video_path

    try:
        if method == VideoInterpolationMethod.BLEND:
            return _blend_minterpolate(video_path, boundary_timestamps, temp_files)
        elif method == VideoInterpolationMethod.RIFE:
            return _rife_process_boundaries(video_path, boundary_timestamps, temp_files, device)
        elif method == VideoInterpolationMethod.FRAME_PAD:
            return _frame_pad_boundaries(video_path, boundary_timestamps, temp_files)
    except Exception as e:
        logger.error(f"Video interpolation ({method.value}) failed: {e}", exc_info=True)

    return video_path
