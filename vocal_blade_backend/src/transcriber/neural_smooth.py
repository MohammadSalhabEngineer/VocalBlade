"""
Smart Edit: Combined room-tone audio bridge + face-morph video bridge.

For each cut boundary in the concatenated video:
  1. Compute bridge duration (80ms rounded to whole frames at source fps)
  2. Generate video bridge: Delaunay face morph if faces detected + movement small,
     else opacity dissolve
  3. Generate audio bridge: quietest room-tone segment of matching duration
  4. Insert both simultaneously via ffmpeg concat → A/V stays in sync
  5. Process boundaries right-to-left to avoid timestamp shift

Falls back gracefully: if face detection fails → dissolve; if ffmpeg step fails →
skip that boundary (keep hard cut).
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded mediapipe face mesh (module-level cache)
_mp_face_mesh = None

# Movement threshold: normalised landmark distance above which dissolve is used
FACE_MOVEMENT_THRESHOLD = 0.15

# Duration of room-tone bridge (ms); actual frame count = round(ms * fps / 1000)
BRIDGE_MS = 300.0


# ---------------------------------------------------------------------------
# Room-tone extraction
# ---------------------------------------------------------------------------

def _extract_room_tone_segment(audio_path: str, duration_sec: float) -> Tuple[np.ndarray, int]:
    """
    Return a numpy array (samples,) of room tone and its sample rate.

    Finds the quietest window of `duration_sec` in the source audio.
    Falls back to zeros if loading fails.
    """
    try:
        import librosa

        # Always extract audio via ffmpeg first — avoids PySoundFile failing on video files
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', audio_path,
                 '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', wav_path],
                capture_output=True, timeout=60,
            )
            load_path = wav_path if r.returncode == 0 else audio_path
        except Exception:
            load_path = audio_path

        audio, sr = librosa.load(load_path, sr=None, mono=True)

        try:
            os.unlink(wav_path)
        except Exception:
            pass
        target_samples = int(duration_sec * sr)
        target_samples = max(1, min(target_samples, len(audio)))

        if len(audio) <= target_samples:
            return audio.copy(), sr

        step = max(target_samples // 4, 1)
        best_start, best_rms = 0, float('inf')
        for start in range(0, len(audio) - target_samples, step):
            window = audio[start:start + target_samples]
            rms = float(np.sqrt(np.mean(window ** 2)))
            if rms < best_rms:
                best_rms, best_start = rms, start

        return audio[best_start:best_start + target_samples].copy(), sr
    except Exception as e:
        logger.warning(f"Room tone extraction failed: {e}. Using silence.")
        sr = 44100
        return np.zeros(int(duration_sec * sr), dtype=np.float32), sr


# ---------------------------------------------------------------------------
# Face landmark detection + movement scoring
# ---------------------------------------------------------------------------

def _get_face_mesh():
    """
    Return a callable that takes a BGR uint8 frame and returns landmark results,
    or None if mediapipe is unavailable.

    Supports both mediapipe 0.9 (mp.solutions API) and 0.10+ (mp.tasks API).
    """
    global _mp_face_mesh
    if _mp_face_mesh is not None:
        return _mp_face_mesh

    # Try legacy mp.solutions API (mediapipe < 0.10)
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        _mp_face_mesh = ('legacy', face_mesh)
        return _mp_face_mesh
    except Exception:
        pass

    # Try new mp.tasks API (mediapipe >= 0.10)
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # Download model if not cached
        model_path = os.path.join(os.path.dirname(__file__), 'face_landmarker.task')
        if not os.path.exists(model_path):
            import urllib.request
            url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
            urllib.request.urlretrieve(url, model_path)

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        detector = mp_vision.FaceLandmarker.create_from_options(options)
        _mp_face_mesh = ('tasks', detector)
        return _mp_face_mesh
    except Exception as e:
        logger.warning(f"mediapipe init failed (both APIs): {e}")
        _mp_face_mesh = None
        return None


def _detect_face_landmarks(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect face mesh landmarks in a BGR uint8 frame.
    Returns (N, 2) array of (x, y) pixel coords, or None.
    """
    handle = _get_face_mesh()
    if handle is None:
        return None

    api, detector = handle
    h, w = frame.shape[:2]

    try:
        if api == 'legacy':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0].landmark
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            return pts
        else:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)
            if not results.face_landmarks:
                return None
            lm = results.face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            return pts
    except Exception:
        return None


def _compute_face_movement(lm_a: np.ndarray, lm_b: np.ndarray) -> float:
    """
    Normalised mean landmark distance between two frames.
    Uses the face bounding-box diagonal as normaliser.
    Returns a value in [0, ∞); values > FACE_MOVEMENT_THRESHOLD → use dissolve.
    """
    diff = np.mean(np.linalg.norm(lm_a - lm_b, axis=1))
    # Normalise by bounding box diagonal of frame A
    x_range = lm_a[:, 0].max() - lm_a[:, 0].min()
    y_range = lm_a[:, 1].max() - lm_a[:, 1].min()
    diag = float(np.sqrt(x_range ** 2 + y_range ** 2)) + 1e-6
    return float(diff / diag)


# ---------------------------------------------------------------------------
# Frame generation: morph and dissolve
# ---------------------------------------------------------------------------

def _morph_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    lm_a: np.ndarray,
    lm_b: np.ndarray,
    n_frames: int,
) -> List[np.ndarray]:
    """
    Generate n_frames intermediate frames using Delaunay triangulation morphing.

    For each alpha in (0, 1):
      1. Interpolate landmark positions: lm_mid = lm_a*(1-a) + lm_b*a
      2. Delaunay triangulate lm_mid
      3. Affine-warp each triangle from frame_a and frame_b toward lm_mid
      4. Blend warped images: result = warped_a*(1-a) + warped_b*a
    """
    h, w = frame_a.shape[:2]
    alphas = np.linspace(0.0, 1.0, n_frames + 2)[1:-1]
    frames = []

    try:
        from scipy.spatial import Delaunay

        # Add corner points so triangulation covers the whole frame
        corners = np.array([
            [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
            [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2],
        ], dtype=np.float32)
        lm_a_aug = np.vstack([lm_a, corners])
        lm_b_aug = np.vstack([lm_b, corners])

        for alpha in alphas:
            lm_mid = lm_a_aug * (1.0 - alpha) + lm_b_aug * alpha
            tri = Delaunay(lm_mid)
            result = np.zeros_like(frame_a, dtype=np.float32)

            for simplex in tri.simplices:
                pts_mid = lm_mid[simplex].astype(np.float32)
                pts_a = lm_a_aug[simplex].astype(np.float32)
                pts_b = lm_b_aug[simplex].astype(np.float32)

                # Bounding rect of the triangle in mid space
                # cv2.boundingRect needs integer contour shape (N,1,2)
                rx, ry, rw, rh = cv2.boundingRect(pts_mid.reshape(-1, 1, 2).astype(np.int32))
                if rw <= 0 or rh <= 0:
                    continue

                # Offset triangle coords to bounding rect — keep float32 explicitly
                offset = np.array([rx, ry], dtype=np.float32)
                pts_mid_r = (pts_mid - offset).astype(np.float32)
                pts_a_r   = (pts_a   - offset).astype(np.float32)
                pts_b_r   = (pts_b   - offset).astype(np.float32)

                # Warp frame_a patch → mid
                M_a = cv2.getAffineTransform(pts_a_r, pts_mid_r)
                patch_a = cv2.warpAffine(
                    frame_a, M_a, (rw, rh),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )
                # Warp frame_b patch → mid
                M_b = cv2.getAffineTransform(pts_b_r, pts_mid_r)
                patch_b = cv2.warpAffine(
                    frame_b, M_b, (rw, rh),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
                )

                # Blend
                blended = patch_a.astype(np.float32) * (1.0 - alpha) + \
                          patch_b.astype(np.float32) * alpha

                # Create triangle mask
                mask = np.zeros((rh, rw), dtype=np.uint8)
                cv2.fillConvexPoly(mask, pts_mid_r.astype(np.int32), 255)

                # Write into result
                region = result[ry:ry + rh, rx:rx + rw]
                mask_3 = (mask[:region.shape[0], :region.shape[1]] > 0)[:, :, np.newaxis]
                region[:] = np.where(mask_3, blended[:region.shape[0], :region.shape[1]], region)

            frames.append(np.clip(result, 0, 255).astype(np.uint8))

    except Exception as e:
        logger.warning(f"Delaunay morph failed: {e}. Using dissolve instead.")
        return _dissolve_frames(frame_a, frame_b, n_frames)

    return frames


def _freeze_frames(frame_a: np.ndarray, n_frames: int) -> List[np.ndarray]:
    """Hold frame_a for the entire bridge — natural pause, no ghosting."""
    return [frame_a.copy() for _ in range(n_frames)]


def _dissolve_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    n_frames: int,
) -> List[np.ndarray]:
    """Simple opacity dissolve (used internally as morph fallback only)."""
    alphas = np.linspace(0.0, 1.0, n_frames + 2)[1:-1]
    return [
        np.clip(
            frame_a.astype(np.float32) * (1.0 - a) + frame_b.astype(np.float32) * a,
            0, 255
        ).astype(np.uint8)
        for a in alphas
    ]


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _write_bridge_video(
    frames: List[np.ndarray],
    fps: float,
    width: int,
    height: int,
    audio_data: np.ndarray,
    audio_sr: int,
    temp_files: List[str],
) -> Optional[str]:
    """
    Write bridge frames + room-tone audio into a single .mkv file.
    Returns path or None on failure.
    """
    import soundfile as sf

    # Write room-tone audio to temp wav
    fd_a, audio_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd_a)
    temp_files.append(audio_path)
    try:
        sf.write(audio_path, audio_data, audio_sr)
    except Exception as e:
        logger.error(f"Bridge audio write failed: {e}")
        return None

    fd_v, bridge_path = tempfile.mkstemp(suffix='.mkv')
    os.close(fd_v)
    temp_files.append(bridge_path)

    # Pipe BGR frames → ffmpeg, combine with room-tone audio
    proc = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pixel_format', 'bgr24',
         '-video_size', f'{width}x{height}',
         '-framerate', str(fps),
         '-i', 'pipe:0',
         '-i', audio_path,
         '-map', '0:v:0', '-map', '1:a:0',
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
         '-pix_fmt', 'yuv420p',
         '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
         '-shortest',
         bridge_path],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        logger.error("Bridge video write timed out")
        return None

    if proc.returncode != 0:
        logger.error(f"Bridge video ffmpeg failed: {proc.stderr.read().decode()[:300]}")
        return None

    return bridge_path


def _split_and_concat(
    current_path: str,
    split_time: float,
    bridge_path: str,
    temp_files: List[str],
    pre_end: Optional[float] = None,
    post_start: Optional[float] = None,
) -> Optional[str]:
    """
    Split current_path, insert bridge_path between pre and post, return new path.

    pre_end: where to cut the pre segment (defaults to split_time).
    post_start: where to start the post segment (defaults to split_time).
    Setting pre_end < split_time and post_start > split_time replaces content
    on both sides of the cut with the bridge — preventing audio duplication.
    """
    actual_pre_end = pre_end if pre_end is not None else split_time
    actual_post_start = post_start if post_start is not None else split_time

    fd1, pre_path = tempfile.mkstemp(suffix='.mkv')
    os.close(fd1)
    temp_files.append(pre_path)
    fd2, post_path = tempfile.mkstemp(suffix='.mkv')
    os.close(fd2)
    temp_files.append(post_path)

    # Extract pre segment
    r1 = subprocess.run(
        ['ffmpeg', '-y', '-i', current_path,
         '-t', str(actual_pre_end),
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
         pre_path],
        capture_output=True, text=True, timeout=120,
    )
    if r1.returncode != 0:
        logger.error(f"Pre-split failed: {r1.stderr[:200]}")
        return None

    # Extract post segment
    r2 = subprocess.run(
        ['ffmpeg', '-y', '-i', current_path,
         '-ss', str(actual_post_start),
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
         post_path],
        capture_output=True, text=True, timeout=120,
    )
    if r2.returncode != 0:
        logger.error(f"Post-split failed: {r2.stderr[:200]}")
        return None

    # Write concat list
    fd3, list_path = tempfile.mkstemp(suffix='.txt')
    os.close(fd3)
    temp_files.append(list_path)
    with open(list_path, 'w') as f:
        f.write(f"file '{pre_path}'\n")
        f.write(f"file '{bridge_path}'\n")
        f.write(f"file '{post_path}'\n")

    fd4, out_path = tempfile.mkstemp(suffix='.mkv')
    os.close(fd4)
    temp_files.append(out_path)

    r3 = subprocess.run(
        ['ffmpeg', '-y',
         '-f', 'concat', '-safe', '0', '-i', list_path,
         '-c:v', 'copy',
         '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
         out_path],
        capture_output=True, text=True, timeout=300,
    )
    if r3.returncode != 0:
        logger.error(f"Concat failed: {r3.stderr[:200]}")
        return None

    return out_path


def _probe_fps(path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', path],
        capture_output=True, text=True, timeout=30,
    )
    raw = r.stdout.strip()
    if '/' in raw:
        n, d = raw.split('/')
        return float(n) / max(float(d), 1.0)
    try:
        return float(raw)
    except Exception:
        return 30.0


def _probe_duration(path: str) -> float:
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'csv=p=0', path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def apply_neural_smooth(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    source_audio_path: str,
    device: str = "cuda",
) -> str:
    """
    Apply Smart Edit at each cut boundary: insert a room-tone audio bridge +
    face-morph (or dissolve) video bridge of identical duration.

    Processes boundaries right-to-left to avoid cumulative timestamp shifts.
    Returns path to processed video, or original path if nothing changed.
    """
    if not boundary_timestamps:
        return video_path

    fps = _probe_fps(video_path)
    if fps <= 0:
        fps = 30.0

    # Compute bridge frame count once (same for all boundaries)
    bridge_frames_count = max(2, round(BRIDGE_MS * fps / 1000.0))
    actual_bridge_sec = bridge_frames_count / fps

    logger.info(
        f"SmartEdit: {len(boundary_timestamps)} boundaries, "
        f"bridge={bridge_frames_count} frames ({actual_bridge_sec*1000:.1f}ms @ {fps:.2f}fps)"
    )

    # Extract room tone once from source audio
    room_tone_data, room_tone_sr = _extract_room_tone_segment(
        source_audio_path, actual_bridge_sec
    )
    # Trim/pad to exact duration
    target_rt_samples = int(actual_bridge_sec * room_tone_sr)
    if len(room_tone_data) > target_rt_samples:
        room_tone_data = room_tone_data[:target_rt_samples]
    elif len(room_tone_data) < target_rt_samples:
        room_tone_data = np.concatenate([
            room_tone_data,
            np.zeros(target_rt_samples - len(room_tone_data), dtype=np.float32)
        ])

    # Apply 10ms fade-in and fade-out to prevent audible clicks at bridge edges
    fade_samples = min(int(0.010 * room_tone_sr), len(room_tone_data) // 4)
    if fade_samples >= 2:
        fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        room_tone_data = room_tone_data.copy()
        room_tone_data[:fade_samples]  *= fade_in
        room_tone_data[-fade_samples:] *= fade_out

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("SmartEdit: cannot open video")
        return video_path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    current_path = video_path
    applied = 0

    # Process right-to-left: a boundary at time T in the ORIGINAL video stays
    # at T + (applied_so_far * actual_bridge_sec) in current_path.
    # Easier to sort descending and always reference the running current_path.
    for ts in sorted(boundary_timestamps, reverse=True):
        # Adjust timestamp for previously inserted bridges (bridges inserted
        # AFTER this one in original time, so they don't shift this boundary)
        adjusted_ts = ts  # right-to-left: no earlier bridges have shifted this point yet

        # Read the two frames around the boundary
        cap2 = cv2.VideoCapture(current_path)
        if not cap2.isOpened():
            logger.warning(f"SmartEdit: cannot open {current_path}, skipping boundary {ts:.2f}s")
            continue

        boundary_idx = int(round(adjusted_ts * fps))
        prev_idx = max(0, boundary_idx - 1)

        cap2.set(cv2.CAP_PROP_POS_FRAMES, prev_idx)
        ret_a, frame_a = cap2.read()
        cap2.set(cv2.CAP_PROP_POS_FRAMES, boundary_idx)
        ret_b, frame_b = cap2.read()
        cap2.release()

        if not (ret_a and ret_b):
            logger.warning(f"SmartEdit: could not read frames at {ts:.2f}s, skipping")
            continue

        # Generate video bridge frames
        lm_a = _detect_face_landmarks(frame_a)
        lm_b = _detect_face_landmarks(frame_b)

        if (lm_a is not None and lm_b is not None and
                _compute_face_movement(lm_a, lm_b) < FACE_MOVEMENT_THRESHOLD):
            bridge_frames = _morph_frames(frame_a, frame_b, lm_a, lm_b, bridge_frames_count)
            method_used = "morph"
        else:
            # Freeze on last frame before cut — avoids ghost artifacts from blending
            # two frames with very different head positions
            bridge_frames = _freeze_frames(frame_a, bridge_frames_count)
            method_used = "freeze"

        logger.info(f"SmartEdit: boundary {ts:.2f}s → {method_used} ({bridge_frames_count} frames)")

        # Write bridge video+audio
        bridge_path = _write_bridge_video(
            bridge_frames, fps, width, height,
            room_tone_data, room_tone_sr, temp_files
        )
        if bridge_path is None:
            logger.warning(f"SmartEdit: bridge write failed at {ts:.2f}s, skipping")
            continue

        # Insert bridge at boundary
        new_path = _split_and_concat(current_path, adjusted_ts, bridge_path, temp_files)
        if new_path is None:
            logger.warning(f"SmartEdit: concat failed at {ts:.2f}s, skipping")
            continue

        current_path = new_path
        applied += 1

    if applied == 0:
        logger.info("SmartEdit: no boundaries processed")
        return video_path

    logger.info(f"SmartEdit: applied to {applied}/{len(boundary_timestamps)} boundaries → {current_path}")
    return current_path


# ---------------------------------------------------------------------------
# FOMM (First Order Motion Model) integration
# ---------------------------------------------------------------------------

_FOMM_BASE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'first-order-model')
_FOMM_DIR = os.path.normpath(_FOMM_BASE)
_FOMM_CHECKPOINT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'vox-cpk.pth.tar')
)
_FOMM_CONFIG = os.path.join(_FOMM_DIR, 'config', 'vox-256.yaml')

_fomm_generator = None
_fomm_kp_detector = None
_fomm_device: str = ""


def _setup_fomm(device: str) -> bool:
    """Clone FOMM repo + download checkpoint if needed. Load and cache model. Returns True on success."""
    global _fomm_generator, _fomm_kp_detector, _fomm_device

    if _fomm_generator is not None and _fomm_device == device:
        return True

    try:
        # 1. Clone repo if not present
        if not os.path.exists(_FOMM_DIR):
            logger.info("FOMM: cloning first-order-model repo...")
            r = subprocess.run(
                ['git', 'clone', '--depth', '1',
                 'https://github.com/AliaksandrSiarohin/first-order-model',
                 _FOMM_DIR],
                capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                logger.error(f"FOMM: git clone failed: {r.stderr[:300]}")
                return False

        # 2. Download checkpoint if not present (~700MB)
        os.makedirs(os.path.dirname(_FOMM_CHECKPOINT), exist_ok=True)
        if not os.path.exists(_FOMM_CHECKPOINT) or os.path.getsize(_FOMM_CHECKPOINT) < 1024:
            logger.info("FOMM: checkpoint not found at %s", _FOMM_CHECKPOINT)
            logger.info("FOMM: please manually download vox-cpk.pth.tar (~700MB) from "
                        "https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH "
                        "and place it at %s", _FOMM_CHECKPOINT)
            logger.warning("FOMM: checkpoint unavailable — will use freeze-frame fallback")
            return False

        # 3. Add FOMM dir to sys.path and import modules
        import sys
        if _FOMM_DIR not in sys.path:
            sys.path.insert(0, _FOMM_DIR)

        import torch
        import yaml

        with open(_FOMM_CONFIG) as f:
            cfg = yaml.safe_load(f)

        from modules.generator import OcclusionAwareGenerator
        from modules.keypoint_detector import KPDetector

        generator = OcclusionAwareGenerator(
            **cfg['model_params']['generator_params'],
            **cfg['model_params']['common_params'],
        )
        kp_detector = KPDetector(
            **cfg['model_params']['kp_detector_params'],
            **cfg['model_params']['common_params'],
        )

        checkpoint = torch.load(_FOMM_CHECKPOINT, map_location='cpu')
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        generator = generator.to(device).eval()
        kp_detector = kp_detector.to(device).eval()

        _fomm_generator = generator
        _fomm_kp_detector = kp_detector
        _fomm_device = device
        logger.info("FOMM: model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"FOMM setup failed: {e}", exc_info=True)
        return False


def _fomm_morph_frames(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    n_frames: int,
    device: str,
) -> List[np.ndarray]:
    """
    Generate n_frames bridging frame_a → frame_b using FOMM inference.
    Falls back to freeze-frame on any failure.

    frame_a, frame_b: BGR uint8 (H, W, 3)
    Returns: list of n_frames BGR uint8 frames
    """
    if not _setup_fomm(device):
        logger.warning("FOMM unavailable — using freeze fallback")
        return _freeze_frames(frame_a, n_frames)

    try:
        import torch
        import sys
        if _FOMM_DIR not in sys.path:
            sys.path.insert(0, _FOMM_DIR)

        h, w = frame_a.shape[:2]

        # Normalize + resize to 256×256, BGR→RGB
        fa_256 = cv2.resize(cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
        fb_256 = cv2.resize(cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0

        # Build source tensor (1, 3, 256, 256)
        source_t = torch.tensor(fa_256[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        target_t = torch.tensor(fb_256[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if device != 'cpu':
            source_t = source_t.to(device)
            target_t = target_t.to(device)

        # Extract keypoints for both endpoints — no pixel blending, no ghost artifacts
        with torch.no_grad():
            kp_source = _fomm_kp_detector(source_t)
            kp_target = _fomm_kp_detector(target_t)

        # Interpolate keypoints directly between A and B for each intermediate frame
        result_frames = []
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)  # strictly between 0 and 1
            kp_interp = {'value': kp_source['value'] * (1.0 - t) + kp_target['value'] * t}
            if 'jacobian' in kp_source and 'jacobian' in kp_target:
                kp_interp['jacobian'] = (
                    kp_source['jacobian'] * (1.0 - t) + kp_target['jacobian'] * t
                )
            with torch.no_grad():
                out = _fomm_generator(source_t, kp_source=kp_source, kp_driving=kp_interp)
            pred = out['prediction'].data.cpu().numpy()[0].transpose(1, 2, 0)
            f = (pred * 255.0).clip(0, 255).astype(np.uint8)
            f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            if (f.shape[1], f.shape[0]) != (w, h):
                f = cv2.resize(f, (w, h))
            result_frames.append(f)

        return result_frames

    except Exception as e:
        logger.warning(f"FOMM inference failed: {e} — using freeze fallback")
        return _freeze_frames(frame_a, n_frames)


# ---------------------------------------------------------------------------
# Speech crossfade bridge (audio for Smart Cut)
# ---------------------------------------------------------------------------

def _extract_speech_crossfade_bridge(
    source_audio_path: str,
    cut_a_end_sec: float,
    cut_b_start_sec: float,
    duration_sec: float,
) -> Tuple[np.ndarray, int]:
    """
    Blend the last `duration_sec` of segment A with the first `duration_sec`
    of segment B to create a natural speech crossfade bridge.

    Returns (bridge_audio, sample_rate).
    Falls back to zeros (silence) on any failure.
    """
    sr = 48000
    n_samples = int(duration_sec * sr)
    silence = np.zeros(n_samples, dtype=np.float32)

    try:
        import librosa

        # Extract source audio via ffmpeg to ensure correct format
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        try:
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', source_audio_path,
                 '-vn', '-acodec', 'pcm_s16le', '-ar', str(sr), wav_path],
                capture_output=True, timeout=60,
            )
            load_path = wav_path if r.returncode == 0 else source_audio_path
        except Exception:
            load_path = source_audio_path

        audio, actual_sr = librosa.load(load_path, sr=sr, mono=True)

        try:
            os.unlink(wav_path)
        except Exception:
            pass

        # Resample if needed
        if actual_sr != sr:
            import librosa.core
            audio = librosa.resample(audio, orig_sr=actual_sr, target_sr=sr)

        # Extract tail of segment A: last duration_sec before cut_a_end
        a_end_sample = int(cut_a_end_sec * sr)
        a_start_sample = max(0, a_end_sample - n_samples)
        a_tail = audio[a_start_sample:a_end_sample].copy()

        # Extract head of segment B: first duration_sec after cut_b_start
        b_start_sample = int(cut_b_start_sec * sr)
        b_end_sample = min(len(audio), b_start_sample + n_samples)
        b_head = audio[b_start_sample:b_end_sample].copy()

        # Pad both to exactly n_samples
        def _pad(arr: np.ndarray, n: int) -> np.ndarray:
            if len(arr) >= n:
                return arr[:n]
            return np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float32)])

        a_tail = _pad(a_tail, n_samples)
        b_head = _pad(b_head, n_samples)

        # Equal-power cosine crossfade
        t = np.linspace(0.0, np.pi / 2.0, n_samples, dtype=np.float32)
        fade_out = np.cos(t) ** 2  # 1 → 0
        fade_in  = np.sin(t) ** 2  # 0 → 1

        bridge = a_tail * fade_out + b_head * fade_in
        return bridge, sr

    except Exception as e:
        logger.warning(f"Speech crossfade bridge extraction failed: {e}. Using silence.")
        return silence, sr


# ---------------------------------------------------------------------------
# Smart Cut entry point (FOMM + speech crossfade)
# ---------------------------------------------------------------------------

def apply_smart_cut(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    source_audio_path: str,
    keep_ranges: list,
    device: str = "cuda",
) -> str:
    """
    Apply Smart Cut at each cut boundary: insert a speech-crossfade audio bridge
    + FOMM-animated video bridge of identical duration.

    keep_ranges: the (already snapped) TimeRange list used to build the concatenated video.
    keep_ranges[i].end is the original source timestamp of segment i's end (= cut_a_end).
    keep_ranges[i+1].start is the original source timestamp of segment i+1's start (= cut_b_start).

    Processes boundaries right-to-left to avoid cumulative timestamp shifts.
    Returns path to processed video, or original path if nothing changed.
    """
    if not boundary_timestamps:
        return video_path

    fps = _probe_fps(video_path)
    if fps <= 0:
        fps = 30.0

    bridge_frames_count = max(2, round(BRIDGE_MS * fps / 1000.0))
    actual_bridge_sec = bridge_frames_count / fps

    logger.info(
        f"SmartCut: {len(boundary_timestamps)} boundaries, "
        f"bridge={bridge_frames_count} frames ({actual_bridge_sec * 1000:.1f}ms @ {fps:.2f}fps)"
    )

    # Build a map: boundary_timestamp → (cut_a_end, cut_b_start) from keep_ranges
    # boundary_timestamps[i] corresponds to the join between keep_ranges[i] and keep_ranges[i+1]
    boundary_source_map: dict = {}
    sorted_boundaries = sorted(boundary_timestamps)
    for i, ts in enumerate(sorted_boundaries):
        if i < len(keep_ranges) - 1:
            boundary_source_map[ts] = (keep_ranges[i].end, keep_ranges[i + 1].start)
        else:
            # Fallback: use boundary timestamp itself
            boundary_source_map[ts] = (ts, ts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("SmartCut: cannot open video")
        return video_path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    current_path = video_path
    applied = 0

    for ts in sorted(boundary_timestamps, reverse=True):
        cut_a_end, cut_b_start = boundary_source_map.get(ts, (ts, ts))

        # Extract speech crossfade audio bridge.
        # The bridge REPLACES actual_bridge_sec of content on each side of the cut
        # (pre is trimmed by actual_bridge_sec, post is trimmed by actual_bridge_sec).
        # This prevents audio duplication: without trimming the pre's last
        # actual_bridge_sec would be heard twice (once in pre, once as a_tail in the bridge).
        bridge_audio, bridge_sr = _extract_speech_crossfade_bridge(
            source_audio_path, cut_a_end, cut_b_start, actual_bridge_sec
        )

        # Read frames at the trimmed edges (not the original cut point) so FOMM
        # morphs the faces that will actually be visible at the bridge boundaries.
        cap2 = cv2.VideoCapture(current_path)
        if not cap2.isOpened():
            logger.warning(f"SmartCut: cannot open {current_path}, skipping {ts:.2f}s")
            continue

        boundary_idx = int(round(ts * fps))
        pre_edge_idx  = max(0, boundary_idx - bridge_frames_count)
        post_edge_idx = min(int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)) - 1,
                            boundary_idx + bridge_frames_count)

        cap2.set(cv2.CAP_PROP_POS_FRAMES, pre_edge_idx)
        ret_a, frame_a = cap2.read()
        cap2.set(cv2.CAP_PROP_POS_FRAMES, post_edge_idx)
        ret_b, frame_b = cap2.read()
        cap2.release()

        if not (ret_a and ret_b):
            logger.warning(f"SmartCut: could not read frames at {ts:.2f}s, skipping")
            continue

        # Generate video bridge: FOMM with freeze fallback
        bridge_frames = _fomm_morph_frames(frame_a, frame_b, bridge_frames_count, device)
        method_used = "fomm" if len(bridge_frames) == bridge_frames_count else "freeze"
        logger.info(f"SmartCut: boundary {ts:.2f}s → {method_used} ({bridge_frames_count} frames)")

        # Write bridge video + speech crossfade audio
        bridge_path = _write_bridge_video(
            bridge_frames, fps, width, height,
            bridge_audio, bridge_sr, temp_files
        )
        if bridge_path is None:
            logger.warning(f"SmartCut: bridge write failed at {ts:.2f}s, skipping")
            continue

        # Replace content on both sides of the cut with the bridge to avoid duplication.
        # pre_end trims the last actual_bridge_sec from segment A so the bridge's
        # a_tail starts exactly where the pre ends (no echo).
        # post_start skips the first actual_bridge_sec of segment B so the bridge's
        # b_head covers that content (no echo at the other side).
        pre_end    = max(0.0, ts - actual_bridge_sec)
        post_start = ts + actual_bridge_sec
        new_path = _split_and_concat(
            current_path, ts, bridge_path, temp_files,
            pre_end=pre_end, post_start=post_start,
        )
        if new_path is None:
            logger.warning(f"SmartCut: concat failed at {ts:.2f}s, skipping")
            continue

        current_path = new_path
        applied += 1

    if applied == 0:
        logger.info("SmartCut: no boundaries processed")
        return video_path

    logger.info(f"SmartCut: applied to {applied}/{len(boundary_timestamps)} boundaries → {current_path}")
    return current_path
