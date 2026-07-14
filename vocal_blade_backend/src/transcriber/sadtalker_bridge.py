"""
SadTalker audio-driven face reenactment bridge.

For each cut boundary, generates a talking-head video clip where the face from
the last frame of segment A is animated to speak the F5-TTS synthesised audio
bridge, then the clip is inserted at the cut point.

Pipeline per boundary:
  1. Extract last frame of segment A as reference image
  2. Generate F5-TTS voice-cloning audio bridge (or fall back to crossfade)
  3. Run SadTalker (subprocess) to animate the face from that audio
  4. Blend SadTalker output back into original video via _split_and_concat
"""
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_SADTALKER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'SadTalker')
)
_SADTALKER_CHECKPOINT_DIR = os.path.join(_SADTALKER_DIR, 'checkpoints')
_SADTALKER_RUNNER = os.path.join(_SADTALKER_DIR, 'run_sadtalker.py')


def _is_sadtalker_available() -> bool:
    return (
        os.path.exists(_SADTALKER_RUNNER)
        and os.path.exists(os.path.join(_SADTALKER_CHECKPOINT_DIR, 'SadTalker_V0.0.2_256.safetensors'))
    )


def _extract_frame(video_path: str, time_sec: float, output_path: str) -> bool:
    """Extract a single frame at time_sec from video_path and write as JPEG."""
    result = subprocess.run(
        ['ffmpeg', '-y', '-ss', str(time_sec), '-i', video_path,
         '-frames:v', '1', '-q:v', '2', output_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0 and os.path.exists(output_path)


def _generate_f5tts_bridge(
    source_audio_path: str,
    cut_a_end_sec: float,
    cut_b_start_sec: float,
    duration_sec: float,
    temp_files: list,
    device: str = "cuda",
) -> Optional[str]:
    """
    Generate a voice-cloned bridge audio using F5-TTS.
    Returns path to bridge WAV, or None if F5-TTS unavailable.
    Falls back to speech crossfade on failure.
    """
    try:
        from .audio_continuity import ContinuityMethod
        from .neural_smooth import _extract_speech_crossfade_bridge
        import soundfile as sf

        # Try F5-TTS first
        try:
            from .audio_continuity import _bridge_ai_regeneration
            import torchaudio
            import numpy as _np

            waveform, sr = torchaudio.load(source_audio_path)
            audio = waveform.numpy()  # (channels, samples)

            seg_a_end_sample = int(cut_a_end_sec * sr)
            seg_b_start_sample = int(cut_b_start_sec * sr)
            bridge_samples = int(duration_sec * sr)

            # Context tail from segment A
            ctx_samples = int(3.0 * sr)
            tail_a_start = max(0, seg_a_end_sample - ctx_samples)
            tail_a = audio[:, tail_a_start:seg_a_end_sample]
            head_b_end = min(audio.shape[1], seg_b_start_sample + ctx_samples)
            head_b = audio[:, seg_b_start_sample:head_b_end]

            bridge_audio = _bridge_ai_regeneration(
                tail_a, head_b, bridge_samples, sr,
                audio, seg_a_end_sample, seg_b_start_sample, device
            )

            fd, bridge_path = tempfile.mkstemp(suffix='_f5bridge.wav')
            os.close(fd)
            temp_files.append(bridge_path)
            sf.write(bridge_path, bridge_audio.T if bridge_audio.ndim > 1 else bridge_audio, sr)
            logger.info(f"SadTalker: F5-TTS bridge generated ({duration_sec:.2f}s)")
            return bridge_path

        except Exception as e:
            logger.warning(f"SadTalker: F5-TTS bridge failed ({e}), falling back to crossfade")

        # Fallback: speech crossfade bridge
        import soundfile as sf
        bridge_data, bridge_sr = _extract_speech_crossfade_bridge(
            source_audio_path, cut_a_end_sec, cut_b_start_sec, duration_sec
        )
        fd, bridge_path = tempfile.mkstemp(suffix='_xfade_bridge.wav')
        os.close(fd)
        temp_files.append(bridge_path)
        sf.write(bridge_path, bridge_data, bridge_sr)
        return bridge_path

    except Exception as e:
        logger.error(f"SadTalker: audio bridge generation failed: {e}")
        return None


def _run_sadtalker(
    source_image: str,
    driven_audio: str,
    result_dir: str,
    device: str = "cuda",
    size: int = 256,
) -> Optional[str]:
    """
    Run SadTalker inference via subprocess.
    Returns path to generated MP4, or None on failure.
    """
    if not _is_sadtalker_available():
        logger.warning("SadTalker: checkpoints not found — skipping")
        return None

    python_exe = sys.executable
    cmd = [
        python_exe, _SADTALKER_RUNNER,
        '--source_image', source_image,
        '--driven_audio', driven_audio,
        '--result_dir', result_dir,
        '--checkpoint_dir', _SADTALKER_CHECKPOINT_DIR,
        '--size', str(size),
        '--device', device,
        '--still',
        '--preprocess', 'crop',
        '--batch_size', '1',
    ]

    logger.info(f"SadTalker: running inference (size={size}, device={device})")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=_SADTALKER_DIR,
        )
        if result.returncode != 0:
            logger.error(f"SadTalker subprocess failed:\n{result.stderr[:500]}")
            return None

        # Find generated video in result_dir (deepest mp4 file)
        mp4_files = list(Path(result_dir).rglob('*.mp4'))
        if not mp4_files:
            logger.error("SadTalker: no .mp4 output found in result_dir")
            return None

        # Pick the most-recently modified file (SadTalker creates timestamped subdirs)
        best = max(mp4_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"SadTalker: output → {best}")
        return str(best)

    except subprocess.TimeoutExpired:
        logger.error("SadTalker: inference timed out (>300s)")
        return None
    except Exception as e:
        logger.error(f"SadTalker: subprocess error: {e}")
        return None


def _match_video_format(
    sadtalker_path: str,
    reference_video: str,
    duration_sec: float,
    temp_files: list,
    driven_audio: str,
) -> Optional[str]:
    """
    Re-encode the SadTalker output to match the reference video's format,
    attach the bridge audio, and trim to exactly duration_sec.

    SadTalker produces 256×256 video; we need to match the source resolution.
    """
    # Probe source video size
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate',
         '-of', 'csv=p=0', reference_video],
        capture_output=True, text=True, timeout=30,
    )
    w, h, fps_raw = 256, 256, '25/1'
    if probe.returncode == 0 and probe.stdout.strip():
        parts = probe.stdout.strip().split(',')
        if len(parts) >= 3:
            try:
                w, h = int(parts[0]), int(parts[1])
                fps_raw = parts[2]
            except Exception:
                pass

    fd, out_path = tempfile.mkstemp(suffix='_st_bridge.mkv')
    os.close(fd)
    temp_files.append(out_path)

    result = subprocess.run(
        ['ffmpeg', '-y',
         '-i', sadtalker_path,
         '-i', driven_audio,
         '-map', '0:v:0', '-map', '1:a:0',
         '-vf', f'scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2',
         '-r', fps_raw,
         '-t', str(duration_sec),
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
         '-pix_fmt', 'yuv420p',
         '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
         '-shortest',
         out_path],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        logger.error(f"SadTalker format match failed: {result.stderr[:300]}")
        return None

    return out_path


def apply_sadtalker_cut(
    video_path: str,
    boundary_timestamps: List[float],
    temp_files: List[str],
    source_audio_path: str,
    keep_ranges: list,
    device: str = "cuda",
) -> str:
    """
    Apply SadTalker-driven face reenactment at each cut boundary.

    For each boundary:
      1. Extract reference face frame from video
      2. Generate F5-TTS (or crossfade) audio bridge
      3. Run SadTalker to animate the face from the audio
      4. Insert the SadTalker clip at the cut point

    Falls back to freeze-frame + crossfade audio if SadTalker fails at any step.
    Returns path to processed video (original path if nothing applied).
    """
    from .neural_smooth import (
        _probe_fps, _probe_duration, _split_and_concat,
        _write_bridge_video, _freeze_frames,
        BRIDGE_MS,
    )

    if not boundary_timestamps:
        return video_path

    if not _is_sadtalker_available():
        logger.warning("SadTalker: not available — using freeze fallback")
        return video_path

    fps = _probe_fps(video_path)
    if fps <= 0:
        fps = 30.0

    bridge_frames_count = max(2, round(BRIDGE_MS * fps / 1000.0))
    actual_bridge_sec = bridge_frames_count / fps

    logger.info(
        f"SadTalkerCut: {len(boundary_timestamps)} boundaries, "
        f"bridge={bridge_frames_count} frames ({actual_bridge_sec*1000:.1f}ms @ {fps:.2f}fps)"
    )

    # Build boundary→source range map
    sorted_boundaries = sorted(boundary_timestamps)
    boundary_source_map = {}
    for i, ts in enumerate(sorted_boundaries):
        if i < len(keep_ranges) - 1:
            boundary_source_map[ts] = (keep_ranges[i].end, keep_ranges[i + 1].start)
        else:
            boundary_source_map[ts] = (ts, ts)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("SadTalkerCut: cannot open video")
        return video_path
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    current_path = video_path
    applied = 0

    for ts in sorted(boundary_timestamps, reverse=True):
        cut_a_end, cut_b_start = boundary_source_map.get(ts, (ts, ts))

        # Step 1: Generate audio bridge
        bridge_audio_path = _generate_f5tts_bridge(
            source_audio_path, cut_a_end, cut_b_start, actual_bridge_sec,
            temp_files, device
        )
        if bridge_audio_path is None:
            logger.warning(f"SadTalkerCut: audio bridge failed at {ts:.2f}s, skipping")
            continue

        # Step 2: Extract reference face frame (slightly before the cut)
        pre_edge_time = max(0.0, ts - actual_bridge_sec * 0.5)
        fd_img, frame_path = tempfile.mkstemp(suffix='_sadtalker_ref.jpg')
        os.close(fd_img)
        temp_files.append(frame_path)

        if not _extract_frame(current_path, pre_edge_time, frame_path):
            logger.warning(f"SadTalkerCut: frame extraction failed at {ts:.2f}s, skipping")
            continue

        # Step 3: Run SadTalker
        result_dir = tempfile.mkdtemp(prefix='sadtalker_result_')
        temp_files.append(result_dir)

        sadtalker_video = _run_sadtalker(
            source_image=frame_path,
            driven_audio=bridge_audio_path,
            result_dir=result_dir,
            device=device,
            size=256,
        )

        if sadtalker_video is None:
            logger.warning(f"SadTalkerCut: SadTalker failed at {ts:.2f}s, using freeze fallback")
            # Freeze fallback: use last frame + crossfade audio
            cap2 = cv2.VideoCapture(current_path)
            boundary_idx = int(round(ts * fps))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, max(0, boundary_idx - 1))
            ret, frame_a = cap2.read()
            cap2.release()
            if not ret:
                continue
            bridge_frames = _freeze_frames(frame_a, bridge_frames_count)
            import soundfile as sf
            bridge_data, bridge_sr = _load_bridge_audio_array(bridge_audio_path)
            bridge_path = _write_bridge_video(
                bridge_frames, fps, width, height,
                bridge_data, bridge_sr, temp_files
            )
            if bridge_path is None:
                continue
        else:
            # Step 4: Match format and trim to bridge duration
            bridge_path = _match_video_format(
                sadtalker_video, current_path, actual_bridge_sec,
                temp_files, bridge_audio_path
            )
            if bridge_path is None:
                logger.warning(f"SadTalkerCut: format match failed at {ts:.2f}s, skipping")
                continue

        # Step 5: Insert bridge (replacing actual_bridge_sec on each side)
        pre_end = max(0.0, ts - actual_bridge_sec)
        post_start = ts + actual_bridge_sec
        new_path = _split_and_concat(
            current_path, ts, bridge_path, temp_files,
            pre_end=pre_end, post_start=post_start,
        )
        if new_path is None:
            logger.warning(f"SadTalkerCut: concat failed at {ts:.2f}s, skipping")
            continue

        current_path = new_path
        applied += 1

        # Clean up SadTalker result dir
        try:
            import shutil
            shutil.rmtree(result_dir, ignore_errors=True)
        except Exception:
            pass

    if applied == 0:
        logger.info("SadTalkerCut: no boundaries processed")
        return video_path

    logger.info(f"SadTalkerCut: applied to {applied}/{len(boundary_timestamps)} boundaries → {current_path}")
    return current_path


def _load_bridge_audio_array(audio_path: str) -> Tuple[np.ndarray, int]:
    """Load bridge audio file into numpy array for _write_bridge_video."""
    try:
        import soundfile as sf
        data, sr = sf.read(audio_path, dtype='float32')
        return data, sr
    except Exception:
        return np.zeros(int(0.3 * 48000), dtype=np.float32), 48000
