"""
4-Tier Audio Continuity System for VocalBlade.

Processes audio boundaries between kept segments during export to eliminate
clicks/pops and produce smooth transitions. Four methods available:

1. Crossfade — Fast equal-power cosine crossfade (numpy)
2. Smart Silence — Silence-aware cut placement (scipy/numpy RMS)
3. Spectral — FFT-based phase-coherent crossfade (torch)
4. AI Inpainting — Neural network audio generation (diffusers, optional)

Dependencies: numpy, scipy, soundfile, torch (all available via whisperx env).
"""

import logging
import os
import subprocess
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from .types import ContinuityMethod, TimeRange

logger = logging.getLogger(__name__)

# Check optional AI dependencies
try:
    from diffusers import AudioLDM2Pipeline
    AI_INPAINTING_AVAILABLE = True
except ImportError:
    AI_INPAINTING_AVAILABLE = False

# librosa — available via whisperx dependency; used for better pitch detection and Griffin-Lim
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# EnCodec — Meta's neural audio codec; used for latent-space boundary blending
try:
    from encodec import EncodecModel
    ENCODEC_AVAILABLE = True
except ImportError:
    ENCODEC_AVAILABLE = False

# Module-level EnCodec model cache
_encodec_model = None
_encodec_device: str = ""


def _load_audio(input_path: str) -> tuple:
    """Load audio using torchaudio, return (numpy array shape (channels, samples), sample_rate)."""
    waveform, sr = torchaudio.load(input_path)
    # waveform: (channels, samples) tensor
    return waveform.numpy(), sr


# RMS energy above this threshold means the candidate is still in speech — don't snap.
# ~-36dB relative to full scale (0.015 * 0.015 = 2.25e-4 mean square).
_SILENCE_RMS_THRESHOLD = 0.015


def _find_optimal_cut_point(
    mono: np.ndarray, sample_pos: int, sr: int,
    search_ms: float = 80.0, bias: str = "none",
) -> int:
    """Find the optimal cut point within ±search_ms of sample_pos.

    Exploits WhisperX word-level alignment: since sample_pos is at a word
    boundary, the search window likely spans a silence gap between words.

    If the best candidate energy is still above _SILENCE_RMS_THRESHOLD (still
    in speech), returns sample_pos unchanged — prevents snapping into a phoneme
    when no real silence exists in the window.

    Args:
        bias: Directional preference for cut placement.
            "outward_right" — prefer points to the right of sample_pos (for range.end)
            "outward_left"  — prefer points to the left of sample_pos (for range.start)
            "none"          — symmetric search (default)

    Strategy:
    1. Compute short-window RMS energy (~5ms) to find the lowest-energy region
    2. If that region is above silence threshold → bail out, use original timestamp
    3. Within the low-energy region, find actual zero-crossings (waveform sign changes)
    4. If zero-crossing found → use it (cleanest possible cut)
    5. Otherwise → use minimum-amplitude sample in lowest-energy region
    """
    search_samples = int(search_ms / 1000.0 * sr)

    # With bias, restrict search to the outward direction only
    if bias == "outward_right":
        left = sample_pos
        right = min(len(mono), sample_pos + search_samples)
    elif bias == "outward_left":
        left = max(0, sample_pos - search_samples)
        right = sample_pos
    else:
        left = max(0, sample_pos - search_samples)
        right = min(len(mono), sample_pos + search_samples)

    if right <= left:
        return sample_pos

    window = mono[left:right]

    # Step 1: Find lowest-energy region using short RMS windows (~5ms)
    rms_win = max(int(0.005 * sr), 8)  # ~5ms window
    if len(window) < rms_win:
        # Window too short — just find minimum amplitude
        return left + int(np.argmin(np.abs(window)))

    # Compute rolling RMS energy
    n = len(window)
    energy = np.zeros(n)
    for i in range(n):
        w_start = max(0, i - rms_win // 2)
        w_end = min(n, i + rms_win // 2)
        energy[i] = np.sqrt(np.mean(window[w_start:w_end] ** 2))

    # Find the sample with lowest energy (center of quietest region)
    low_energy_idx = int(np.argmin(energy))

    # Step 2: If the quietest point is still in speech, don't snap — return original
    if energy[low_energy_idx] > _SILENCE_RMS_THRESHOLD:
        return sample_pos

    # Step 3: Search for actual zero-crossings near the low-energy point
    # Look within ±5ms of the lowest-energy point
    zc_search = max(int(0.005 * sr), 8)
    zc_left = max(0, low_energy_idx - zc_search)
    zc_right = min(n - 1, low_energy_idx + zc_search)

    for i in range(zc_left, zc_right):
        if window[i] * window[i + 1] < 0:  # Sign change = zero crossing
            return left + i

    # Step 4: No zero-crossing found — use minimum amplitude in low-energy region
    region = np.abs(window[zc_left:zc_right + 1])
    best_in_region = int(np.argmin(region))
    return left + zc_left + best_in_region


def snap_boundaries(audio_path: str, keep_ranges: list) -> list:
    """Snap all keep_range start/end timestamps to nearest zero-crossings.

    Loads audio once, converts to mono, and adjusts each boundary point.
    Returns new list of ranges with adjusted timestamps.
    """
    try:
        audio, sr = _load_audio(audio_path)
        mono = audio.mean(axis=0) if audio.ndim > 1 else audio.flatten()

        adjusted = []
        for i, r in enumerate(keep_ranges):
            start_sample = int(r.start * sr)
            end_sample = int(r.end * sr)

            # Don't snap the very first start (0) or very last end (file duration)
            new_start = r.start
            if r.start > 0.05:  # Only snap if not near file start
                # Bias right: snap toward kept content (inward), staying in the safety buffer
                # Frontend already expanded start outward — snap inward to find clean cut point
                snapped = _find_optimal_cut_point(mono, start_sample, sr, bias="outward_right")
                new_start = snapped / sr
                # Hard limit: never snap inward past the original word start timestamp
                new_start = max(new_start, r.start)

            new_end = r.end
            if end_sample < len(mono) - int(0.05 * sr):  # Only snap if not near file end
                # Bias left: snap toward kept content (inward), staying in the safety buffer
                # Frontend already expanded end outward — snap inward to find clean cut point
                snapped = _find_optimal_cut_point(mono, end_sample, sr, bias="outward_left")
                new_end = snapped / sr
                # Hard limit: never snap inward past the original word end timestamp
                new_end = min(new_end, r.end)

            # Create a new range-like object with adjusted timestamps
            adjusted.append(type(r)(start=new_start, end=new_end))

        logger.info(f"Snapped {len(keep_ranges)} range boundaries to optimal cut points (energy+zero-crossing)")
        return adjusted
    except Exception as e:
        logger.warning(f"Boundary snapping failed: {e}. Using original ranges.")
        return keep_ranges


def process_audio_continuity(
    input_path: str,
    keep_ranges: List[TimeRange],
    method: ContinuityMethod,
    device: str,
) -> str:
    """
    Process audio with the selected continuity method.

    Loads audio from input_path, extracts segments defined by keep_ranges,
    applies boundary smoothing, concatenates, and writes to a temp WAV file.

    Returns the path to the processed WAV file.
    """
    if len(keep_ranges) == 0:
        raise ValueError("No ranges provided for audio continuity processing")

    # Load full audio at original sample rate
    audio, sr = _load_audio(input_path)
    # audio shape: (channels, samples)

    if len(keep_ranges) == 1:
        # Single range — extract and return directly, no boundary processing needed
        r = keep_ranges[0]
        start_sample = int(r.start * sr)
        end_sample = int(r.end * sr)
        start_sample = max(0, min(start_sample, audio.shape[1]))
        end_sample = max(start_sample, min(end_sample, audio.shape[1]))
        segment = audio[:, start_sample:end_sample]
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        if segment.shape[0] == 1:
            sf.write(output_path, segment[0], sr)
        else:
            sf.write(output_path, segment.T, sr)
        return output_path

    # Extract segments
    segments = []
    for r in keep_ranges:
        start_sample = int(r.start * sr)
        end_sample = int(r.end * sr)
        start_sample = max(0, min(start_sample, audio.shape[1]))
        end_sample = max(start_sample, min(end_sample, audio.shape[1]))
        segments.append(audio[:, start_sample:end_sample])

    # Apply selected continuity method at boundaries
    if method == ContinuityMethod.CROSSFADE:
        result = _crossfade_segments(segments, sr)
    elif method == ContinuityMethod.SMART_SILENCE:
        result = _smart_silence_segments(segments, sr)
    elif method == ContinuityMethod.SPECTRAL:
        result = _spectral_crossfade_segments(segments, sr, device)
    elif method == ContinuityMethod.AI_INPAINTING:
        result = _ai_inpainting_segments(segments, sr, device)
    else:
        result = _crossfade_segments(segments, sr)

    # Write to temp WAV file
    # soundfile expects (samples, channels) for multi-channel or (samples,) for mono
    fd, output_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    if result.shape[0] == 1:
        sf.write(output_path, result[0], sr)
    else:
        sf.write(output_path, result.T, sr)

    return output_path


# =============================================================================
# Method 1: Time-Domain Equal-Power Crossfade
# =============================================================================

def _crossfade_segments(
    segments: List[np.ndarray],
    sr: int,
    fade_duration_ms: float = 20.0,
) -> np.ndarray:
    """
    Apply equal-power cosine crossfade between adjacent segments.

    Uses cos²/sin² curves for energy-preserving blending at boundaries.
    The crossfade overlaps the tail of segment A with the head of segment B,
    so total duration = sum(segment durations) - (N-1) * fade_duration.

    All arrays: shape (channels, samples).
    """
    fade_samples = int(fade_duration_ms / 1000.0 * sr)

    if len(segments) == 0:
        return np.zeros((1, 0))

    if len(segments) == 1:
        return segments[0]

    result = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]
        # Determine actual fade length (limited by segment sizes)
        actual_fade = min(fade_samples, result.shape[1], seg.shape[1])

        if actual_fade <= 0:
            # Segments too short for crossfade — just concatenate
            result = np.concatenate([result, seg], axis=1)
            continue

        # Equal-power crossfade curve
        t = np.linspace(0, np.pi / 2, actual_fade)
        fade_out = np.cos(t) ** 2  # 1 → 0
        fade_in = np.sin(t) ** 2   # 0 → 1

        # Apply crossfade in the overlap region
        overlap = result[:, -actual_fade:] * fade_out + seg[:, :actual_fade] * fade_in

        # Build: result[:-fade] + overlap + seg[fade:]
        result = np.concatenate([
            result[:, :-actual_fade],
            overlap,
            seg[:, actual_fade:],
        ], axis=1)

    return result


# =============================================================================
# Method 2: Smart Silence Detection (using numpy/scipy RMS)
# =============================================================================

def _compute_rms(audio_1d: np.ndarray, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Compute RMS energy per frame (replaces librosa.feature.rms)."""
    n_frames = 1 + (len(audio_1d) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([np.sqrt(np.mean(audio_1d ** 2))])

    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio_1d[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    return rms


def _find_silence_intervals(
    audio_1d: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Find non-silent intervals in audio (replaces librosa.effects.split).
    Returns array of shape (N, 2) with [start_sample, end_sample] pairs.
    """
    rms = _compute_rms(audio_1d, frame_length, hop_length)

    if len(rms) == 0 or np.max(rms) == 0:
        return np.array([])

    # Convert threshold from dB below peak
    ref = np.max(rms)
    threshold = ref * 10 ** (-top_db / 20)

    # Find frames above threshold
    is_loud = rms > threshold

    # Find contiguous loud regions
    intervals = []
    in_interval = False
    start_frame = 0

    for i, loud in enumerate(is_loud):
        if loud and not in_interval:
            start_frame = i
            in_interval = True
        elif not loud and in_interval:
            intervals.append([start_frame * hop_length, i * hop_length])
            in_interval = False

    if in_interval:
        intervals.append([start_frame * hop_length, len(audio_1d)])

    return np.array(intervals) if intervals else np.array([]).reshape(0, 2)


def _smart_silence_segments(
    segments: List[np.ndarray],
    sr: int,
    window_ms: float = 500.0,
    silence_db: float = 40.0,
) -> np.ndarray:
    """
    Silence-aware boundary optimization.

    For each boundary between segments:
    1. If silence exists near the cut point → cut at silence center, minimal fade
    2. If no silence → find lowest-energy frame, apply short crossfade there
    3. Fallback → standard crossfade
    """
    if len(segments) <= 1:
        return segments[0] if segments else np.zeros((1, 0))

    window_samples = int(window_ms / 1000.0 * sr)
    min_fade_samples = int(0.005 * sr)   # 5ms minimal fade

    result = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]

        # Extract boundary regions for analysis (mono mix for detection)
        tail_len = min(window_samples, result.shape[1])
        head_len = min(window_samples, seg.shape[1])
        tail_region = result[:, -tail_len:].mean(axis=0)
        head_region = seg[:, :head_len].mean(axis=0)

        # Strategy 1: Look for silence in tail region
        tail_cut = _find_silence_point(tail_region, sr, silence_db)
        # Strategy 2: Look for silence in head region
        head_cut = _find_silence_point(head_region, sr, silence_db)

        if tail_cut is not None:
            cut_offset = tail_len - tail_cut
            trimmed_result = result[:, :-cut_offset] if cut_offset > 0 else result
            result = _apply_minimal_fade_and_concat(trimmed_result, seg, sr, min_fade_samples)
        elif head_cut is not None:
            trimmed_seg = seg[:, head_cut:] if head_cut > 0 else seg
            result = _apply_minimal_fade_and_concat(result, trimmed_seg, sr, min_fade_samples)
        else:
            # No silence found — find lowest energy point and crossfade
            lowest_tail = _find_lowest_energy_point(tail_region, sr)
            if lowest_tail is not None and lowest_tail < tail_len:
                result = _crossfade_segments([result, seg], sr, fade_duration_ms=15.0)
            else:
                result = _crossfade_segments([result, seg], sr, fade_duration_ms=20.0)

    return result


def _find_silence_point(
    audio_1d: np.ndarray,
    sr: int,
    top_db: float = 40.0,
) -> Optional[int]:
    """Find a silence point in the audio. Returns sample index or None."""
    if len(audio_1d) < sr * 0.01:
        return None

    try:
        intervals = _find_silence_intervals(audio_1d, sr, top_db)
        if len(intervals) == 0:
            # Entire region is silence — cut at center
            return len(audio_1d) // 2

        # Find gaps between non-silent intervals
        for j in range(len(intervals) - 1):
            gap_start = intervals[j][1]
            gap_end = intervals[j + 1][0]
            if gap_end > gap_start:
                return int((gap_start + gap_end) // 2)

        # Check if there's silence at the start or end
        if intervals[0][0] > sr * 0.005:
            return int(intervals[0][0] // 2)
        if intervals[-1][1] < len(audio_1d) - sr * 0.005:
            return int((intervals[-1][1] + len(audio_1d)) // 2)

    except Exception as e:
        logger.debug(f"Silence detection failed: {e}")

    return None


def _find_lowest_energy_point(
    audio_1d: np.ndarray,
    sr: int,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> Optional[int]:
    """Find the frame with lowest RMS energy. Returns sample index."""
    if len(audio_1d) < frame_length:
        return None

    try:
        rms = _compute_rms(audio_1d, frame_length, hop_length)
        min_frame = int(np.argmin(rms))
        return min_frame * hop_length
    except Exception as e:
        logger.debug(f"Energy analysis failed: {e}")
        return None


def _apply_minimal_fade_and_concat(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    sr: int,
    fade_samples: int,
) -> np.ndarray:
    """Concatenate two segments with a minimal linear fade at the boundary."""
    actual_fade = min(fade_samples, seg_a.shape[1], seg_b.shape[1])

    if actual_fade <= 0:
        return np.concatenate([seg_a, seg_b], axis=1)

    fade_out = np.linspace(1.0, 0.0, actual_fade)
    fade_in = np.linspace(0.0, 1.0, actual_fade)

    seg_a_faded = seg_a.copy()
    seg_a_faded[:, -actual_fade:] *= fade_out

    seg_b_faded = seg_b.copy()
    seg_b_faded[:, :actual_fade] *= fade_in

    return np.concatenate([seg_a_faded, seg_b_faded], axis=1)


# =============================================================================
# Method 3: Spectral Crossfade (FFT-based)
# =============================================================================

def _spectral_crossfade_segments(
    segments: List[np.ndarray],
    sr: int,
    device: str,
    overlap_ms: float = 100.0,
) -> np.ndarray:
    """
    FFT-based phase-coherent crossfade.

    For each boundary, extracts overlap regions, computes STFT,
    interpolates magnitude and unwrapped phase separately, then
    reconstructs via ISTFT.
    """
    if len(segments) <= 1:
        return segments[0] if segments else np.zeros((1, 0))

    overlap_samples = int(overlap_ms / 1000.0 * sr)
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft)

    use_gpu = device == "cuda" and torch.cuda.is_available()
    if use_gpu:
        window = window.cuda()

    result = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]
        actual_overlap = min(overlap_samples, result.shape[1], seg.shape[1])

        if actual_overlap < n_fft:
            # Not enough samples for spectral analysis — fall back to crossfade
            result = _crossfade_segments([result, seg], sr, fade_duration_ms=20.0)
            continue

        # Process each channel
        channels = []
        for ch in range(result.shape[0]):
            tail = result[ch, -actual_overlap:]
            head = seg[ch, :actual_overlap]

            # Convert to torch tensors
            tail_t = torch.from_numpy(tail.copy()).float()
            head_t = torch.from_numpy(head.copy()).float()
            if use_gpu:
                tail_t = tail_t.cuda()
                head_t = head_t.cuda()

            # STFT
            tail_stft = torch.stft(tail_t, n_fft, hop_length=hop_length,
                                   window=window, return_complex=True)
            head_stft = torch.stft(head_t, n_fft, hop_length=hop_length,
                                   window=window, return_complex=True)

            # Interpolate magnitude and phase
            n_frames = min(tail_stft.shape[1], head_stft.shape[1])
            tail_stft = tail_stft[:, :n_frames]
            head_stft = head_stft[:, :n_frames]

            tail_mag = tail_stft.abs()
            head_mag = head_stft.abs()
            tail_phase = tail_stft.angle()
            head_phase = head_stft.angle()

            # Linear interpolation weights across frames
            alpha = torch.linspace(0, 1, n_frames, device=tail_stft.device).unsqueeze(0)

            # Interpolate magnitude (linear) and phase (circular via unit vectors)
            interp_mag = tail_mag * (1 - alpha) + head_mag * alpha
            tail_unit = torch.exp(1j * tail_phase)
            head_unit = torch.exp(1j * head_phase)
            interp_unit = tail_unit * (1 - alpha) + head_unit * alpha
            interp_phase = interp_unit.angle()

            # Reconstruct
            interp_stft = interp_mag * torch.exp(1j * interp_phase)
            reconstructed = torch.istft(interp_stft, n_fft, hop_length=hop_length,
                                        window=window, length=actual_overlap)

            if use_gpu:
                reconstructed = reconstructed.cpu()
            channels.append(reconstructed.numpy())

        overlap_audio = np.stack(channels, axis=0)

        # Build: result[:-overlap] + blended_overlap + seg[overlap:]
        result = np.concatenate([
            result[:, :-actual_overlap],
            overlap_audio,
            seg[:, actual_overlap:],
        ], axis=1)

    return result


# =============================================================================
# Method 4: AI Audio Inpainting
# =============================================================================

# Module-level model cache
_ai_model = None
_ai_model_device = None


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using torchaudio (replaces librosa.resample)."""
    if orig_sr == target_sr:
        return audio
    tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def _ai_inpainting_segments(
    segments: List[np.ndarray],
    sr: int,
    device: str,
    context_ms: float = 500.0,
    fill_ms: float = 100.0,
) -> np.ndarray:
    """
    Neural network audio inpainting at segment boundaries.

    Uses a pre-trained AudioLDM2 model to generate smooth transition audio
    between segments. Falls back to spectral crossfade if dependencies
    are not available.
    """
    global _ai_model, _ai_model_device

    if not AI_INPAINTING_AVAILABLE:
        logger.warning(
            "AI inpainting dependencies not installed. "
            "Install with: pip install diffusers transformers accelerate. "
            "Falling back to spectral crossfade."
        )
        return _spectral_crossfade_segments(segments, sr, device)

    if len(segments) <= 1:
        return segments[0] if segments else np.zeros((1, 0))

    # Lazy-load model (cached across calls)
    if _ai_model is None or _ai_model_device != device:
        try:
            logger.info("Loading AudioLDM2 model for AI inpainting (first use)...")
            _ai_model = AudioLDM2Pipeline.from_pretrained(
                "cvssp/audioldm2",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            if device == "cuda" and torch.cuda.is_available():
                _ai_model = _ai_model.to("cuda")
            _ai_model_device = device
            logger.info("AudioLDM2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}. Falling back to spectral crossfade.")
            return _spectral_crossfade_segments(segments, sr, device)

    fill_samples = int(fill_ms / 1000.0 * sr)
    result = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]

        try:
            # Generate transition audio using the model
            prompt = "smooth audio transition, continuous sound"
            with torch.no_grad():
                generated = _ai_model(
                    prompt,
                    audio_length_in_s=fill_ms / 1000.0 + 0.1,
                    num_inference_steps=10,
                    guidance_scale=2.5,
                ).audios[0]

            # Flatten if needed
            gen_audio = generated[0] if len(generated.shape) > 1 else generated

            # Resample from model's output rate (16kHz) to source rate
            gen_audio = _resample_audio(gen_audio, 16000, sr)

            # Trim to fill_samples
            gen_audio = gen_audio[:fill_samples]

            # Expand to match channels
            if result.shape[0] > 1:
                gen_audio = np.tile(gen_audio, (result.shape[0], 1))
            else:
                gen_audio = gen_audio[np.newaxis, :]

            # Crossfade: tail of result → generated → head of seg
            fade_len = min(fill_samples // 4, result.shape[1], seg.shape[1])
            fade_len = max(fade_len, 10)
            if gen_audio.shape[1] < fade_len * 2:
                fade_len = max(gen_audio.shape[1] // 4, 10)

            t_out = np.linspace(1, 0, fade_len)
            t_in = np.linspace(0, 1, fade_len)

            result_copy = result.copy()
            result_copy[:, -fade_len:] *= t_out

            gen_copy = gen_audio.copy()
            gen_copy[:, :fade_len] *= t_in
            if gen_copy.shape[1] > fade_len:
                gen_copy[:, -fade_len:] *= t_out

            seg_copy = seg.copy()
            seg_copy[:, :fade_len] *= t_in

            result = np.concatenate([result_copy, gen_copy, seg_copy], axis=1)

        except Exception as e:
            logger.warning(f"AI inpainting failed at boundary {i}: {e}. Using crossfade fallback.")
            result = _crossfade_segments([result, seg], sr, fade_duration_ms=20.0)

    return result


# =============================================================================
# NEW: In-Place Boundary Post-Processing (Building-Block Architecture)
# =============================================================================
# These functions operate on an already-concatenated audio file and modify
# samples IN-PLACE at boundary timestamps. No samples are added or removed,
# so audio duration is unchanged and A/V sync is guaranteed.
# =============================================================================


def _apply_micro_fade(
    audio: np.ndarray,
    center_sample: int,
    fade_samples: int = 5,
) -> np.ndarray:
    """Apply tiny fade-out/fade-in (~5 samples) at boundary to prevent clicks.

    Always applied regardless of continuity method. This is the Descript-style
    micro-fade that eliminates pops from waveform discontinuities at cut points.

    Args:
        audio: shape (channels, samples)
        center_sample: boundary position
        fade_samples: number of samples per side (default 5 ≈ 0.1ms at 48kHz)
    """
    n_samples = audio.shape[1]
    for i in range(fade_samples):
        alpha = i / fade_samples
        # Fade out: samples before boundary ramp down to ~0
        idx_left = center_sample - fade_samples + i
        if 0 <= idx_left < n_samples:
            audio[:, idx_left] *= alpha
        # Fade in: samples after boundary ramp up from ~0
        idx_right = center_sample + i
        if 0 <= idx_right < n_samples:
            audio[:, idx_right] *= alpha
    return audio


def process_audio_at_boundaries(
    audio_path: str,
    boundary_timestamps: List[float],
    method: ContinuityMethod,
    device: str = "cpu",
) -> str:
    """
    Post-process audio at boundary timestamps using the selected continuity method.

    Operates on an already-concatenated audio file. Modifies samples in-place
    at each boundary — no samples added or removed, duration unchanged.

    Args:
        audio_path: Path to the concatenated WAV file
        boundary_timestamps: List of timestamps (seconds) where segments were joined
        method: Which continuity tier to apply
        device: "cpu" or "cuda"

    Returns:
        Path to the processed WAV file (new temp file)
    """
    if not boundary_timestamps:
        return audio_path

    # Map deprecated methods to active ones
    if method == ContinuityMethod.SMART_SILENCE:
        method = ContinuityMethod.CROSSFADE

    # Load concatenated audio
    audio, sr = _load_audio(audio_path)
    original_shape = audio.shape

    logger.info(
        f"Processing {len(boundary_timestamps)} boundaries with method={method.value}, "
        f"sr={sr}, channels={audio.shape[0]}, samples={audio.shape[1]}"
    )

    # Extract room tone once for all boundaries (if needed)
    room_tone = None
    if method == ContinuityMethod.ROOM_TONE:
        room_tone = _extract_room_tone(audio, sr)
        logger.info(f"Extracted room tone: {room_tone.shape[1]} samples ({room_tone.shape[1]/sr*1000:.0f}ms)")

    for idx, ts in enumerate(boundary_timestamps):
        center = int(ts * sr)

        if center <= 0 or center >= audio.shape[1]:
            logger.debug(f"Boundary {idx} at {ts:.3f}s is out of range, skipping")
            continue

        # Clamp center to valid range
        center = max(1, min(center, audio.shape[1] - 1))

        logger.debug(f"Boundary {idx}: ts={ts:.3f}s, sample={center}")

        # Always apply micro-fade first (prevents clicks regardless of method)
        audio = _apply_micro_fade(audio, center)

        # --- Apply selected tier method (skip for NONE — micro-fade is enough) ---
        if method == ContinuityMethod.NONE:
            continue

        try:
            if method == ContinuityMethod.ROOM_TONE:
                audio = _blend_room_tone_inplace(audio, center, sr, room_tone)
            elif method == ContinuityMethod.CROSSFADE:
                audio = _blend_crossfade_inplace(audio, center, sr)
            elif method == ContinuityMethod.SPECTRAL:
                audio = _blend_spectral_inplace(audio, center, sr, device)
            elif method == ContinuityMethod.WSOLA:
                audio = _blend_wsola_inplace(audio, center, sr)
            elif method == ContinuityMethod.AI_INPAINTING:
                audio = _blend_mel_inpaint_inplace(audio, center, sr, device)
            elif method == ContinuityMethod.ENCODEC:
                audio = _blend_encodec_inplace(audio, center, sr, device)
            else:
                audio = _blend_crossfade_inplace(audio, center, sr)
        except Exception as e:
            logger.warning(f"Boundary {idx} method {method.value} failed: {e}. Using crossfade fallback.")
            try:
                audio = _blend_crossfade_inplace(audio, center, sr)
            except Exception:
                pass

    # Assert sample count preserved
    assert audio.shape == original_shape, (
        f"Sample count changed! {original_shape} → {audio.shape}"
    )

    # Write to temp WAV file
    fd, output_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    if audio.shape[0] == 1:
        sf.write(output_path, audio[0], sr)
    else:
        sf.write(output_path, audio.T, sr)

    logger.info(f"Audio continuity processing complete: {output_path}")
    return output_path


def _match_envelope(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    window_ms: float = 50.0,
) -> np.ndarray:
    """
    Match RMS energy levels across a boundary to prevent volume jumps.

    Computes RMS of window_ms on each side. If difference > 1dB,
    applies a gradual gain ramp to the quieter side.
    """
    window_samples = int(window_ms / 1000.0 * sr)

    left_start = max(0, center_sample - window_samples)
    right_end = min(audio.shape[1], center_sample + window_samples)

    left_region = audio[:, left_start:center_sample]
    right_region = audio[:, center_sample:right_end]

    if left_region.shape[1] == 0 or right_region.shape[1] == 0:
        return audio

    left_rms = np.sqrt(np.mean(left_region ** 2))
    right_rms = np.sqrt(np.mean(right_region ** 2))

    if left_rms < 1e-10 or right_rms < 1e-10:
        return audio

    ratio_db = 20 * np.log10(max(left_rms, right_rms) / min(left_rms, right_rms))

    if ratio_db <= 1.0:
        return audio

    if left_rms < right_rms:
        target_gain = right_rms / left_rms
        ramp = np.linspace(1.0, target_gain, left_region.shape[1])
        audio[:, left_start:center_sample] *= ramp
    else:
        target_gain = left_rms / right_rms
        ramp = np.linspace(target_gain, 1.0, right_region.shape[1])
        audio[:, center_sample:right_end] *= ramp

    return audio


# =============================================================================
# In-Place Blend: Crossfade (Fast)
# =============================================================================

def _blend_crossfade_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    window_ms: float = 40.0,
) -> np.ndarray:
    """
    Apply in-place fade-out/fade-in at the boundary center.

    Fades out the last N ms before the boundary and fades in the first
    N ms after. Uses equal-power cos² curve. The brief energy dip at the
    boundary eliminates clicks/pops from the waveform discontinuity.
    """
    logger.info(f"Crossfade: center={center_sample}, window={window_ms}ms")
    half_samples = int(window_ms / 1000.0 * sr)
    half_samples = max(half_samples, 8)

    left_start = max(0, center_sample - half_samples)
    right_end = min(audio.shape[1], center_sample + half_samples)
    actual_left = center_sample - left_start
    actual_right = right_end - center_sample

    if actual_left < 4 or actual_right < 4:
        return audio

    # Fade out left side approaching boundary (1 → 0)
    t_left = np.linspace(0, np.pi / 2, actual_left)
    audio[:, left_start:center_sample] *= np.cos(t_left) ** 2

    # Fade in right side leaving boundary (0 → 1)
    t_right = np.linspace(0, np.pi / 2, actual_right)
    audio[:, center_sample:right_end] *= np.sin(t_right) ** 2

    return audio


# =============================================================================
# In-Place Blend: Spectral (High Quality)
# =============================================================================

def _blend_spectral_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    device: str = "cpu",
    window_ms: float = 30.0,
) -> np.ndarray:
    """
    Apply in-place FFT phase-coherent blending at the boundary center.

    Uses numpy FFT (not torch) for speed. Extracts regions on each side of
    the boundary, interpolates magnitude and phase in the frequency domain,
    reconstructs, and blends back with equal-power transitions.
    """
    logger.info(f"Spectral: center={center_sample}, window={window_ms}ms")
    half = int(window_ms / 1000.0 * sr / 2)
    half = max(half, 64)

    left_start = max(0, center_sample - half)
    right_end = min(audio.shape[1], center_sample + half)
    actual_left = center_sample - left_start
    actual_right = right_end - center_sample

    if actual_left < 32 or actual_right < 32:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # Use same FFT size for both sides to ensure consistent array lengths
    n = max(actual_left, actual_right)

    for ch in range(audio.shape[0]):
        left = audio[ch, left_start:center_sample].copy()
        right = audio[ch, center_sample:right_end].copy()

        if len(left) == 0 or len(right) == 0:
            continue

        # Window before FFT to reduce spectral leakage
        left_win = np.hanning(len(left))
        right_win = np.hanning(len(right))
        left *= left_win
        right *= right_win

        # FFT — pure numpy, no torch overhead
        # Zero-pad both to same length n for consistent frequency bins
        L = np.fft.rfft(left, n=n)
        R = np.fft.rfft(right, n=n)

        # Interpolate magnitude and phase in frequency domain
        alpha = np.linspace(0, 1, len(L))
        mag = np.abs(L) * (1 - alpha) + np.abs(R) * alpha

        # Circular phase interpolation via unit vectors
        blend_unit = np.exp(1j * np.angle(L)) * (1 - alpha) + np.exp(1j * np.angle(R)) * alpha
        phase = np.angle(blend_unit)

        # Use full width for IRFFT so result covers both sides
        full_n = actual_left + actual_right
        result = np.fft.irfft(mag * np.exp(1j * phase), n=full_n)

        # Blend reconstructed signal back with equal-power curve
        t_l = np.sin(np.linspace(0, np.pi / 2, actual_left)) ** 2
        t_r = np.cos(np.linspace(0, np.pi / 2, actual_right)) ** 2

        audio[ch, left_start:center_sample] = (
            audio[ch, left_start:center_sample] * (1 - t_l) +
            result[:actual_left] * t_l
        )
        audio[ch, center_sample:right_end] = (
            audio[ch, center_sample:right_end] * (1 - t_r) +
            result[actual_left:actual_left + actual_right] * t_r
        )

    return audio


# =============================================================================
# In-Place Blend: WSOLA (Speech-Optimized)
# =============================================================================

def _blend_wsola_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    window_ms: float = 20.0,
    search_ms: float = 10.0,
) -> np.ndarray:
    """
    Waveform Similarity Overlap-Add (WSOLA) blending at boundary.

    Finds the pitch period via numpy autocorrelation, aligns the splice point
    to a pitch-period boundary, and applies equal-power crossfade there.
    Falls back to crossfade if pitch detection fails.

    Pure numpy — no scipy dependency.
    """
    logger.info(f"WSOLA: center={center_sample}, window={window_ms}ms, search={search_ms}ms")
    half_window = int(window_ms / 1000.0 * sr / 2)
    search_samples = int(search_ms / 1000.0 * sr)

    left_start = max(0, center_sample - half_window)
    right_end = min(audio.shape[1], center_sample + half_window)
    actual_left = center_sample - left_start
    actual_right = right_end - center_sample

    if actual_left < 64 or actual_right < 64:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # --- Pitch detection ---
    analysis_len = int(0.050 * sr)  # 50ms (wider window for more reliable YIN)
    analysis_start = max(0, center_sample - analysis_len // 2)
    analysis_end = min(audio.shape[1], center_sample + analysis_len // 2)
    mono = audio[:, analysis_start:analysis_end].mean(axis=0)

    if len(mono) < 64:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    pitch_period = None

    if LIBROSA_AVAILABLE and len(mono) >= 256:
        try:
            # librosa.yin() — YIN algorithm, much more robust than autocorrelation
            # for speech (handles unvoiced consonants, noisy recordings, etc.)
            f0_frames = librosa.yin(
                mono.astype(np.float32),
                fmin=50.0,
                fmax=500.0,
                sr=sr,
                frame_length=min(2048, len(mono) // 2),
            )
            voiced = f0_frames[f0_frames > 60.0]
            if len(voiced) > 0:
                f0_median = float(np.median(voiced))
                pp = int(round(sr / f0_median))
                if 16 <= pp <= int(0.025 * sr):
                    pitch_period = pp
        except Exception:
            pass

    if pitch_period is None:
        # Fallback: numpy autocorrelation
        corr = np.correlate(mono, mono, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] > 0:
            corr = corr / corr[0]
        min_lag = int(0.002 * sr)
        max_lag = min(int(0.020 * sr), len(corr) - 1)
        if min_lag >= max_lag:
            return _blend_crossfade_inplace(audio, center_sample, sr)
        pitch_region = corr[min_lag:max_lag + 1]
        if len(pitch_region) == 0:
            return _blend_crossfade_inplace(audio, center_sample, sr)
        peak_idx = int(np.argmax(pitch_region))
        if pitch_region[peak_idx] < 0.3:
            return _blend_crossfade_inplace(audio, center_sample, sr)
        pitch_period = min_lag + peak_idx

    # --- Find best splice point — vectorized cross-correlation ---
    s_start = max(-search_samples, -(actual_left - pitch_period))
    s_end = min(search_samples, actual_right - pitch_period)
    if s_end <= s_start:
        s_end = s_start + 1

    best_offset = 0
    best_cc = -1.0

    # Use mono mix for splice search
    mono_full = audio.mean(axis=0)
    for offset in range(s_start, s_end):
        splice = center_sample + offset
        if splice - pitch_period < 0 or splice + pitch_period >= audio.shape[1]:
            continue
        left_seg = mono_full[splice - pitch_period:splice]
        right_seg = mono_full[splice:splice + pitch_period]
        norm = np.sqrt(np.dot(left_seg, left_seg) * np.dot(right_seg, right_seg))
        if norm > 1e-10:
            cc = np.dot(left_seg, right_seg) / norm
            if cc > best_cc:
                best_cc = cc
                best_offset = offset

    # --- Apply pitch-aligned fade ---
    splice_center = center_sample + best_offset
    fade_len = max(pitch_period + pitch_period // 2, int(0.005 * sr))

    ls = max(0, splice_center - fade_len)
    re = min(audio.shape[1], splice_center + fade_len)
    al = splice_center - ls
    ar = re - splice_center

    if al < 4 or ar < 4:
        return audio

    t_left = np.linspace(0, np.pi / 2, al)
    t_right = np.linspace(0, np.pi / 2, ar)
    audio[:, ls:splice_center] *= np.cos(t_left) ** 2
    audio[:, splice_center:re] *= np.sin(t_right) ** 2

    return audio


# =============================================================================
# In-Place Blend: Mel-Spectrogram AI Inpainting
# =============================================================================

def _blend_mel_inpaint_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    device: str = "cpu",
    context_ms: float = 40.0,
    blend_ms: float = 20.0,
    fade_ms: float = 10.0,
) -> np.ndarray:
    """
    AI-style audio inpainting using mel-spectrogram interpolation.

    Converts audio on both sides of the boundary to mel spectrograms,
    interpolates in perceptual mel space, reconstructs with Griffin-Lim,
    and crossfades back into the original.

    This produces smoother transitions than time-domain methods because
    mel spectrograms represent audio in a perceptually-weighted frequency
    space — interpolation here creates natural-sounding spectral transitions.

    In-place: same sample count, no duration change, no A/V sync risk.
    """
    logger.info(f"Mel Inpainting: center={center_sample}, context={context_ms}ms, blend={blend_ms}ms")
    context_samples = int(context_ms / 1000.0 * sr)
    blend_samples = int(blend_ms / 1000.0 * sr)
    fade_samples = int(fade_ms / 1000.0 * sr)

    left_start = max(0, center_sample - context_samples)
    right_end = min(audio.shape[1], center_sample + context_samples)
    actual_left = center_sample - left_start
    actual_right = right_end - center_sample

    if actual_left < 256 or actual_right < 256:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # Work on mono for mel processing, apply result to all channels
    mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]

    # Extract context regions
    left_audio = mono[left_start:center_sample].copy()
    right_audio = mono[center_sample:right_end].copy()

    try:
        # Convert to torch tensors
        left_tensor = torch.from_numpy(left_audio).float().unsqueeze(0)
        right_tensor = torch.from_numpy(right_audio).float().unsqueeze(0)

        # Mel spectrogram transform
        n_fft = min(1024, actual_left, actual_right)
        # Ensure n_fft is even
        n_fft = n_fft - (n_fft % 2)
        hop_length = max(n_fft // 4, 1)
        n_mels = min(80, n_fft // 2)

        n_stft = n_fft // 2 + 1

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=1.0,
        )
        # Compute mel spectrograms
        mel_left = mel_transform(left_tensor)   # (1, n_mels, time_left)
        mel_right = mel_transform(right_tensor)  # (1, n_mels, time_right)

        # Number of mel frames to blend at the boundary
        blend_frames = max(4, blend_samples // hop_length)

        # Take the last `blend_frames` from left and first `blend_frames` from right
        left_end_frames = mel_left[:, :, -blend_frames:]
        right_start_frames = mel_right[:, :, :blend_frames]

        # Ensure same number of frames for interpolation
        n_frames = min(left_end_frames.shape[2], right_start_frames.shape[2])
        left_end_frames = left_end_frames[:, :, :n_frames]
        right_start_frames = right_start_frames[:, :, :n_frames]

        # Linear interpolation in mel space
        alpha = torch.linspace(0, 1, n_frames).unsqueeze(0).unsqueeze(0)
        blended_frames = left_end_frames * (1 - alpha) + right_start_frames * alpha

        # Build the full mel spectrogram for reconstruction:
        # [left context] [blended] [right context]
        full_mel = torch.cat([
            mel_left[:, :, :-blend_frames],
            blended_frames,
            mel_right[:, :, blend_frames:],
        ], dim=2)

        # Convert mel → linear spectrogram and reconstruct audio
        region_len = right_end - left_start

        if LIBROSA_AVAILABLE:
            # librosa path: mel_to_stft (pseudoinverse) + GriffinLim with momentum
            # Dramatically better than torchaudio's 32-iteration GL (no momentum)
            full_mel_np = full_mel.squeeze(0).numpy().astype(np.float32)
            linear_spec_np = librosa.feature.inverse.mel_to_stft(
                full_mel_np, sr=sr, n_fft=n_fft, power=1.0,
            )
            reconstructed = librosa.griffinlim(
                linear_spec_np,
                n_iter=128,
                hop_length=hop_length,
                win_length=n_fft,
                n_fft=n_fft,
                momentum=0.99,
                length=region_len,
            )
        else:
            # Fallback: torchaudio InverseMelScale + GriffinLim (lower quality)
            inverse_mel = torchaudio.transforms.InverseMelScale(
                n_stft=n_stft, n_mels=n_mels, sample_rate=sr,
            )
            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=n_fft, hop_length=hop_length, power=1.0, n_iter=32,
            )
            linear_spec = inverse_mel(full_mel)
            reconstructed = griffin_lim(linear_spec).squeeze(0).numpy()

        # The reconstructed audio covers the full context region
        # Trim or pad to match the original region length
        region_len = right_end - left_start
        if len(reconstructed) > region_len:
            reconstructed = reconstructed[:region_len]
        elif len(reconstructed) < region_len:
            # Pad with original audio
            padded = mono[left_start:right_end].copy()
            padded[:len(reconstructed)] = reconstructed
            reconstructed = padded

        # Crossfade the reconstructed audio back into the original
        # Only replace the center portion, fade at edges
        fade_in = min(fade_samples, actual_left // 2)
        fade_out = min(fade_samples, actual_right // 2)

        output = mono[left_start:right_end].copy()

        # Center region: pure reconstructed
        output[fade_in:-fade_out] = reconstructed[fade_in:len(output) - fade_out]

        # Left fade: original → reconstructed
        if fade_in > 0:
            t = np.linspace(0, np.pi / 2, fade_in)
            w = np.sin(t) ** 2
            output[:fade_in] = mono[left_start:left_start + fade_in] * (1 - w) + reconstructed[:fade_in] * w

        # Right fade: reconstructed → original
        if fade_out > 0:
            t = np.linspace(0, np.pi / 2, fade_out)
            w = np.cos(t) ** 2
            output[-fade_out:] = reconstructed[-fade_out:] * w + mono[right_end - fade_out:right_end] * (1 - w)

        # Apply to all channels
        for ch in range(audio.shape[0]):
            audio[ch, left_start:right_end] = output

    except Exception as e:
        logger.warning(f"Mel inpainting failed: {e}. Falling back to crossfade.")
        return _blend_crossfade_inplace(audio, center_sample, sr)

    return audio


# =============================================================================
# In-Place Blend: EnCodec Neural Codec Latent Blending (Premium)
# =============================================================================

def _blend_encodec_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    device: str = "cuda",
    context_ms: float = 80.0,
    bandwidth: float = 12.0,
) -> np.ndarray:
    """
    Meta EnCodec neural audio codec latent-space blending at boundary.

    Encodes both sides of the boundary together as a single continuous sequence
    so the decoder's LSTM has full causal context across the cut. Only a narrow
    window around the cut is replaced in the output — the rest of the audio is
    untouched original, avoiding broad codec-quality degradation.

    Strategy:
      1. Extract 80ms left + 80ms right (160ms total) centered on the cut.
      2. Encode the full 160ms as ONE sequence → the LSTM sees the whole region.
      3. In the embedding, blend the frames nearest the cut boundary.
      4. Decode → get 160ms of smooth neural audio.
      5. Replace only the 40ms window at the cut; crossfade 20ms on each edge.

    Operates at EnCodec's native 24kHz — resamples if needed.
    In-place: sample count unchanged, A/V sync safe.
    """
    global _encodec_model, _encodec_device

    if not ENCODEC_AVAILABLE:
        logger.warning("encodec not installed. Falling back to spectral crossfade.")
        return _blend_spectral_inplace(audio, center_sample, sr, device)

    logger.info(f"EnCodec: center={center_sample}, context={context_ms}ms, bandwidth={bandwidth}")
    context_samples = int(context_ms / 1000.0 * sr)

    left_start = max(0, center_sample - context_samples)
    right_end = min(audio.shape[1], center_sample + context_samples)
    actual_left = center_sample - left_start
    actual_right = right_end - center_sample

    if actual_left < 128 or actual_right < 128:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # Lazy-load EnCodec model
    if _encodec_model is None or _encodec_device != device:
        try:
            model = EncodecModel.encodec_model_24khz(pretrained=True)
            model.set_target_bandwidth(bandwidth)
            model.eval()
            if device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            _encodec_model = model
            _encodec_device = device
            logger.info(f"EnCodec 24kHz model loaded on {device}")
        except Exception as e:
            logger.error(f"EnCodec model load failed: {e}. Falling back to spectral.")
            return _blend_spectral_inplace(audio, center_sample, sr, device)

    ENCODEC_SR = 24000

    try:
        mono = audio.mean(axis=0) if audio.shape[0] > 1 else audio[0]
        region = mono[left_start:right_end].copy().astype(np.float32)
        region_len = len(region)
        # Where, within the region, is the cut boundary?
        center_in_region = center_sample - left_start

        # Resample to 24kHz for EnCodec
        if sr != ENCODEC_SR:
            region_24 = torchaudio.functional.resample(
                torch.from_numpy(region).unsqueeze(0), sr, ENCODEC_SR
            ).squeeze(0).numpy()
            center_in_region_24 = int(center_in_region * ENCODEC_SR / sr)
        else:
            region_24 = region
            center_in_region_24 = center_in_region

        # Encode the full region as ONE sequence (LSTM sees full context)
        region_t = torch.from_numpy(region_24).float().unsqueeze(0).unsqueeze(0)
        if _encodec_device == "cuda" and torch.cuda.is_available():
            region_t = region_t.cuda()

        with torch.no_grad():
            emb = _encodec_model.encoder(region_t)   # (1, D, T_enc)
            T_enc = emb.shape[2]

            # Identify the embedding frame nearest the cut boundary
            # EnCodec has a 320-sample hop at 24kHz (75 Hz frame rate)
            enc_hop = region_24.shape[0] / max(T_enc, 1)
            cut_frame = int(round(center_in_region_24 / enc_hop))
            cut_frame = max(1, min(T_enc - 2, cut_frame))

            # Blend 3 frames around the cut in latent space
            n_blend = min(3, cut_frame, T_enc - cut_frame)
            if n_blend >= 1:
                for i in range(n_blend):
                    alpha = (i + 1) / (n_blend + 1)
                    f_left = cut_frame - n_blend + i
                    f_right = cut_frame + i
                    if 0 <= f_left < T_enc and 0 <= f_right < T_enc:
                        blended = emb[:, :, f_left] * (1 - alpha) + emb[:, :, f_right] * alpha
                        emb[:, :, cut_frame - n_blend // 2 + i] = blended

            decoded = _encodec_model.decoder(emb)   # (1, 1, T_audio)
            decoded_np = decoded.squeeze(0).squeeze(0).cpu().float().numpy()

        # Resample decoded audio back to original sr
        if sr != ENCODEC_SR:
            decoded_np = torchaudio.functional.resample(
                torch.from_numpy(decoded_np).unsqueeze(0), ENCODEC_SR, sr
            ).squeeze(0).numpy()

        # Trim/pad decoded to match region length
        if len(decoded_np) > region_len:
            decoded_np = decoded_np[:region_len]
        elif len(decoded_np) < region_len:
            decoded_np = np.concatenate([decoded_np,
                                         np.zeros(region_len - len(decoded_np), dtype=np.float32)])

        # Only replace audio in a 40ms window at the cut — keep the rest original.
        # This prevents codec-quality artifacts from spreading across the full context.
        blend_zone_ms = 20.0   # 20ms each side of cut = 40ms total replaced
        blend_samples = int(blend_zone_ms / 1000.0 * sr)
        blend_start = max(0, center_in_region - blend_samples)
        blend_end = min(region_len, center_in_region + blend_samples)
        blend_len = blend_end - blend_start

        # Crossfade 20ms at each edge of the blend zone
        fade_samples = min(int(0.020 * sr), blend_len // 3)

        output_region = region.copy()
        blend_decoded = decoded_np[blend_start:blend_end]

        if fade_samples >= 2:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            # Left edge: original fades out, decoded fades in
            blend_decoded[:fade_samples] = (
                region[blend_start:blend_start + fade_samples] * fade_out +
                blend_decoded[:fade_samples] * fade_in
            )
            # Right edge: decoded fades out, original fades in
            blend_decoded[-fade_samples:] = (
                blend_decoded[-fade_samples:] * fade_out +
                region[blend_end - fade_samples:blend_end] * fade_in
            )

        output_region[blend_start:blend_end] = blend_decoded

        for ch in range(audio.shape[0]):
            audio[ch, left_start:right_end] = output_region

        logger.info(f"EnCodec: replaced {blend_len} samples ({blend_len/sr*1000:.1f}ms) at boundary")

    except Exception as e:
        logger.warning(f"EnCodec blending failed: {e}. Falling back to spectral crossfade.")
        return _blend_spectral_inplace(audio, center_sample, sr, device)

    return audio


# =============================================================================
# In-Place Blend: Room Tone Replacement (Recommended)
# =============================================================================

def _extract_room_tone(
    audio: np.ndarray,
    sr: int,
    target_ms: float = 300.0,
) -> np.ndarray:
    """
    Extract the quietest section of audio to use as room tone template.

    Slides a window across the mono-mixed audio, finds the section with
    lowest RMS energy. Returns that section with the original channel count.

    Args:
        audio: shape (channels, samples)
        sr: sample rate
        target_ms: length of room tone to extract

    Returns:
        Room tone array, shape (channels, target_samples)
    """
    target_samples = int(target_ms / 1000.0 * sr)
    target_samples = min(target_samples, audio.shape[1])

    if audio.shape[1] <= target_samples:
        return audio.copy()

    # Mono mix for energy analysis
    mono = audio.mean(axis=0)

    # Compute RMS in sliding windows (step = target_samples // 4 for speed)
    step = max(target_samples // 4, 1)
    best_start = 0
    best_rms = float('inf')

    for start in range(0, audio.shape[1] - target_samples, step):
        window = mono[start:start + target_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms < best_rms:
            best_rms = rms
            best_start = start

    room_tone = audio[:, best_start:best_start + target_samples].copy()
    logger.debug(f"Room tone extracted at {best_start/sr:.2f}s, RMS={best_rms:.6f}")
    return room_tone


def _blend_room_tone_inplace(
    audio: np.ndarray,
    center_sample: int,
    sr: int,
    room_tone: np.ndarray,
    replace_ms: float = 15.0,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """
    Replace audio at the boundary with room tone, crossfaded at edges.

    This creates a natural-sounding pause at the edit point by filling the
    boundary region with the recording's ambient noise instead of trying to
    blend two different speech sounds together.

    In-place: same number of samples, no duration change, no A/V sync risk.

    Args:
        audio: shape (channels, samples)
        center_sample: boundary position
        sr: sample rate
        room_tone: room tone template, shape (channels, N)
        replace_ms: total width of region to replace with room tone
        fade_ms: crossfade duration at each edge
    """
    logger.info(f"Room Tone: center={center_sample}, replace={replace_ms}ms, fade={fade_ms}ms")
    replace_half = int(replace_ms / 1000.0 * sr / 2)
    fade_samples = int(fade_ms / 1000.0 * sr)
    fade_samples = max(fade_samples, 4)

    # Region to replace
    left_start = max(0, center_sample - replace_half)
    right_end = min(audio.shape[1], center_sample + replace_half)
    region_len = right_end - left_start

    if region_len < fade_samples * 2 + 4:
        # Too short for room tone — fall back to crossfade
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # Tile room tone to fill region length
    rt_samples = room_tone.shape[1]
    if rt_samples == 0:
        return _blend_crossfade_inplace(audio, center_sample, sr)

    # Repeat and trim room tone to exact region length
    repeats = (region_len // rt_samples) + 1
    tiled = np.tile(room_tone, (1, repeats))[:, :region_len]

    # Build the replacement with crossfades at edges:
    # [fade: speech→room_tone] [pure room tone] [fade: room_tone→speech]
    fade_left = min(fade_samples, region_len // 3)
    fade_right = min(fade_samples, region_len // 3)

    # Create output region starting from room tone
    output = tiled.copy()

    # Left crossfade: speech (1→0) + room_tone (0→1)
    t = np.linspace(0, np.pi / 2, fade_left)
    fade_out = np.cos(t) ** 2  # speech: 1 → 0
    fade_in = np.sin(t) ** 2   # room tone: 0 → 1
    output[:, :fade_left] = (
        audio[:, left_start:left_start + fade_left] * fade_out +
        tiled[:, :fade_left] * fade_in
    )

    # Right crossfade: room_tone (1→0) + speech (0→1)
    t = np.linspace(0, np.pi / 2, fade_right)
    fade_out = np.cos(t) ** 2  # room tone: 1 → 0
    fade_in = np.sin(t) ** 2   # speech: 0 → 1
    output[:, -fade_right:] = (
        tiled[:, -fade_right:] * fade_out +
        audio[:, right_end - fade_right:right_end] * fade_in
    )

    # Write back in-place
    audio[:, left_start:right_end] = output

    return audio


# =============================================================================
# Bridge Audio Generation (Insert Mode)
# =============================================================================
# Instead of modifying existing samples in-place, these functions generate
# a standalone audio segment to INSERT between two kept segments.
# Duration is controlled by bridge_ms — this audio gets its own space,
# so no speech is damaged. A/V sync is maintained by also inserting
# RIFE-interpolated video frames of the same duration.
# =============================================================================


def generate_bridge_audio(
    full_audio: np.ndarray,
    seg_a_end_sample: int,
    seg_b_start_sample: int,
    method: ContinuityMethod,
    sr: int,
    bridge_ms: float = 80.0,
    room_tone: Optional[np.ndarray] = None,
    device: str = "cpu",
) -> np.ndarray:
    """
    Generate a bridge audio segment to insert between two kept segments.

    Unlike in-place methods, this creates NEW audio of exactly bridge_ms duration.
    No existing speech samples are modified.

    Args:
        full_audio: Full original audio, shape (channels, samples)
        seg_a_end_sample: Sample index where segment A ends
        seg_b_start_sample: Sample index where segment B starts
        method: Which continuity method to use
        sr: Sample rate
        bridge_ms: Duration of bridge audio in milliseconds
        room_tone: Pre-extracted room tone (for ROOM_TONE method)
        device: "cpu" or "cuda"

    Returns:
        Bridge audio array, shape (channels, bridge_samples)
    """
    # Per-method optimal bridge durations
    method_durations = {
        ContinuityMethod.ROOM_TONE: 250.0,
        ContinuityMethod.CROSSFADE: 250.0,
        ContinuityMethod.SPECTRAL: 250.0,
        ContinuityMethod.WSOLA: 250.0,
        ContinuityMethod.AI_INPAINTING: 250.0,
        ContinuityMethod.AI_REGENERATION: 300.0,
    }
    actual_bridge_ms = method_durations.get(method, bridge_ms)
    bridge_samples = int(actual_bridge_ms / 1000.0 * sr)
    channels = full_audio.shape[0]
    context_ms = 300.0  # How much context to use from each side
    context_samples = int(context_ms / 1000.0 * sr)

    # Extract context from each side
    a_start = max(0, seg_a_end_sample - context_samples)
    b_end = min(full_audio.shape[1], seg_b_start_sample + context_samples)
    tail_a = full_audio[:, a_start:seg_a_end_sample]  # Last N ms of segment A
    head_b = full_audio[:, seg_b_start_sample:b_end]   # First N ms of segment B

    logger.info(
        f"Generating bridge audio: method={method.value}, duration={bridge_ms}ms, "
        f"tail_a={tail_a.shape[1]} samples, head_b={head_b.shape[1]} samples"
    )

    if method == ContinuityMethod.ROOM_TONE:
        bridge = _bridge_room_tone(tail_a, head_b, bridge_samples, sr, room_tone)
    elif method == ContinuityMethod.CROSSFADE:
        bridge = _bridge_crossfade(tail_a, head_b, bridge_samples, sr)
    elif method == ContinuityMethod.SPECTRAL:
        bridge = _bridge_spectral(tail_a, head_b, bridge_samples, sr, device)
    elif method == ContinuityMethod.WSOLA:
        bridge = _bridge_wsola(tail_a, head_b, bridge_samples, sr)
    elif method == ContinuityMethod.AI_INPAINTING:
        bridge = _bridge_mel_inpaint(tail_a, head_b, bridge_samples, sr, device)
    elif method == ContinuityMethod.AI_REGENERATION:
        bridge = _bridge_ai_regeneration(tail_a, head_b, bridge_samples, sr, full_audio, seg_a_end_sample, seg_b_start_sample, device)
    else:
        bridge = _bridge_crossfade(tail_a, head_b, bridge_samples, sr)

    # Ensure correct shape
    if bridge.shape[0] != channels:
        if bridge.ndim == 1:
            bridge = np.tile(bridge[np.newaxis, :], (channels, 1))
        else:
            bridge = np.tile(bridge[:1], (channels, 1))

    # Ensure exact length
    if bridge.shape[1] > bridge_samples:
        bridge = bridge[:, :bridge_samples]
    elif bridge.shape[1] < bridge_samples:
        padded = np.zeros((channels, bridge_samples), dtype=bridge.dtype)
        padded[:, :bridge.shape[1]] = bridge
        bridge = padded

    return bridge


def extract_room_tone_from_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Public wrapper for room tone extraction."""
    return _extract_room_tone(audio, sr)


def _bridge_room_tone(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
    room_tone: Optional[np.ndarray],
    fade_ms: float = 50.0,
) -> np.ndarray:
    """Generate bridge audio filled with room tone, smoothly faded at edges."""
    channels = tail_a.shape[0]
    fade_samples = int(fade_ms / 1000.0 * sr)
    fade_samples = min(fade_samples, bridge_samples // 3)

    if room_tone is None or room_tone.shape[1] == 0:
        bridge = np.zeros((channels, bridge_samples))
    else:
        # Tile room tone to fill bridge
        repeats = (bridge_samples // room_tone.shape[1]) + 1
        bridge = np.tile(room_tone, (1, repeats))[:, :bridge_samples]

        # Match room tone volume to surrounding speech level
        tail_rms = np.sqrt(np.mean(tail_a[:, -min(tail_a.shape[1], int(0.05 * sr)):] ** 2))
        rt_rms = np.sqrt(np.mean(bridge ** 2))
        if rt_rms > 1e-10 and tail_rms > 1e-10:
            # Scale room tone to ~50% of speech level (audible natural ambient)
            target_rms = tail_rms * 0.5
            bridge *= target_rms / rt_rms

    # Crossfade: speech A → room tone at start
    if fade_samples > 0 and tail_a.shape[1] >= fade_samples:
        t = np.linspace(0, np.pi / 2, fade_samples)
        tail_end = tail_a[:, -fade_samples:]
        bridge[:, :fade_samples] = tail_end * np.cos(t) ** 2 + bridge[:, :fade_samples] * np.sin(t) ** 2

    # Crossfade: room tone → speech B at end
    if fade_samples > 0 and head_b.shape[1] >= fade_samples:
        t = np.linspace(0, np.pi / 2, fade_samples)
        head_start = head_b[:, :fade_samples]
        bridge[:, -fade_samples:] = bridge[:, -fade_samples:] * np.cos(t) ** 2 + head_start * np.sin(t) ** 2

    return np.clip(bridge, -1.0, 1.0)


def _bridge_crossfade(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
) -> np.ndarray:
    """Generate bridge audio: overlapping equal-power crossfade of tail_a into head_b.

    Uses cos²/sin² power-complementary curves so total power stays constant.
    A floor check prevents any section from dipping below 30% of speech level.
    """
    channels = tail_a.shape[0]

    # Use up to bridge_samples from each side
    a_len = min(bridge_samples, tail_a.shape[1])
    b_len = min(bridge_samples, head_b.shape[1])

    # Place tail_a at start, head_b at end of bridge-length buffers
    a_portion = np.zeros((channels, bridge_samples))
    b_portion = np.zeros((channels, bridge_samples))

    if a_len > 0:
        a_portion[:, :a_len] = tail_a[:, -a_len:]
    if b_len > 0:
        b_portion[:, -b_len:] = head_b[:, :b_len]

    # Equal-power crossfade: cos²/sin² curves across full bridge
    t = np.linspace(0, np.pi / 2, bridge_samples)
    fade_out = np.cos(t) ** 2  # 1 → 0
    fade_in = np.sin(t) ** 2   # 0 → 1

    bridge = a_portion * fade_out + b_portion * fade_in

    # Floor protection: ensure no 5ms frame drops below 30% of speech RMS
    speech_rms = np.sqrt(np.mean(
        np.concatenate([tail_a[:, -min(tail_a.shape[1], int(0.05 * sr)):],
                        head_b[:, :min(head_b.shape[1], int(0.05 * sr))]], axis=1) ** 2
    ))
    floor_level = speech_rms * 0.3

    frame_len = max(64, int(0.005 * sr))  # 5ms frames
    for start in range(0, bridge_samples - frame_len, frame_len):
        frame = bridge[:, start:start + frame_len]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < floor_level and frame_rms > 1e-10:
            boost = min(floor_level / frame_rms, 3.0)
            bridge[:, start:start + frame_len] *= boost
        elif frame_rms < 1e-10:
            # Dead silence — fill with low-level noise at floor
            noise = np.random.randn(channels, frame_len) * floor_level * 0.5
            bridge[:, start:start + frame_len] = noise

    return np.clip(bridge, -1.0, 1.0)


def _bridge_spectral(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
    device: str = "cpu",
) -> np.ndarray:
    """Generate spectrally-colored room tone bridge.

    Instead of interpolating between two different speech segments (which creates
    helium/chipmunk artifacts), this extracts the spectral envelope (formant
    structure) of the surrounding speech and applies it to white noise, creating
    a natural-sounding fill that matches the recording's acoustic characteristics.
    """
    channels = tail_a.shape[0]

    bridges = []
    for ch in range(channels):
        # Get spectral envelope from both sides
        a_ctx = tail_a[ch, -min(bridge_samples * 2, tail_a.shape[1]):]
        b_ctx = head_b[ch, :min(bridge_samples * 2, head_b.shape[1])]

        n = bridge_samples
        A = np.fft.rfft(a_ctx, n=n)
        B = np.fft.rfft(b_ctx, n=n)

        # Smooth spectral envelope (average magnitude, no phase — avoids helium)
        avg_mag = (np.abs(A) + np.abs(B)) / 2.0

        # Heavy smoothing to get just the formant shape, not individual harmonics
        kernel_size = max(5, len(avg_mag) // 20)
        kernel = np.ones(kernel_size) / kernel_size
        smooth_mag = np.convolve(avg_mag, kernel, mode='same')

        # Generate noise with this spectral shape (random phase)
        random_phase = np.random.uniform(-np.pi, np.pi, len(smooth_mag))
        colored_noise = np.fft.irfft(smooth_mag * np.exp(1j * random_phase), n=bridge_samples)

        # Scale to match speech RMS at ~40% (audible ambience)
        speech_rms = np.sqrt(np.mean(np.concatenate([a_ctx, b_ctx]) ** 2))
        noise_rms = np.sqrt(np.mean(colored_noise ** 2))
        if noise_rms > 1e-10 and speech_rms > 1e-10:
            colored_noise *= (speech_rms * 0.40) / noise_rms

        bridges.append(colored_noise)

    bridge = np.stack(bridges, axis=0)

    # Smooth crossfade edges with surrounding speech (50ms)
    fade_len = min(int(0.050 * sr), bridge_samples // 3)
    if fade_len > 0 and tail_a.shape[1] >= fade_len:
        t = np.linspace(0, np.pi / 2, fade_len)
        bridge[:, :fade_len] = tail_a[:, -fade_len:] * np.cos(t) ** 2 + bridge[:, :fade_len] * np.sin(t) ** 2
    if fade_len > 0 and head_b.shape[1] >= fade_len:
        t = np.linspace(0, np.pi / 2, fade_len)
        bridge[:, -fade_len:] = bridge[:, -fade_len:] * np.cos(t) ** 2 + head_b[:, :fade_len] * np.sin(t) ** 2

    return np.clip(bridge, -1.0, 1.0)


def _bridge_wsola(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
) -> np.ndarray:
    """Generate bridge audio via pitch-aligned overlapping crossfade (WSOLA-style).

    Detects pitch period, snaps crossfade boundaries to pitch period boundaries
    for smoother splices, then applies overlapping equal-power crossfade with
    floor protection (same as crossfade method but pitch-aware).
    """
    channels = tail_a.shape[0]

    # Detect pitch period from tail of A
    mono_a = tail_a.mean(axis=0) if channels > 1 else tail_a[0]
    analysis = mono_a[-min(int(0.030 * sr), len(mono_a)):]
    pitch_period = None

    if len(analysis) >= 64:
        corr = np.correlate(analysis, analysis, mode='full')
        corr = corr[len(corr) // 2:]
        if corr[0] > 0:
            corr = corr / corr[0]
        min_lag = int(0.002 * sr)
        max_lag = min(int(0.020 * sr), len(corr) - 1)
        if min_lag < max_lag:
            pitch_region = corr[min_lag:max_lag + 1]
            if len(pitch_region) > 0:
                peak_idx = int(np.argmax(pitch_region))
                if pitch_region[peak_idx] > 0.3:
                    pitch_period = min_lag + peak_idx

    # Use up to bridge_samples from each side
    a_len = min(bridge_samples, tail_a.shape[1])
    b_len = min(bridge_samples, head_b.shape[1])

    # Snap lengths to pitch period boundaries if detected
    if pitch_period and a_len > pitch_period:
        n_periods = max(1, a_len // pitch_period)
        a_len = n_periods * pitch_period
    if pitch_period and b_len > pitch_period:
        n_periods = max(1, b_len // pitch_period)
        b_len = n_periods * pitch_period

    # Place tail_a at start, head_b at end
    a_portion = np.zeros((channels, bridge_samples))
    b_portion = np.zeros((channels, bridge_samples))

    actual_a = min(a_len, tail_a.shape[1])
    actual_b = min(b_len, head_b.shape[1])

    if actual_a > 0:
        a_portion[:, :actual_a] = tail_a[:, -actual_a:]
    if actual_b > 0:
        b_portion[:, -actual_b:] = head_b[:, :actual_b]

    # Equal-power crossfade across full bridge
    t = np.linspace(0, np.pi / 2, bridge_samples)
    bridge = a_portion * np.cos(t) ** 2 + b_portion * np.sin(t) ** 2

    # Floor protection: ensure no 5ms frame drops below 30% of speech RMS
    speech_rms = np.sqrt(np.mean(
        np.concatenate([tail_a[:, -min(tail_a.shape[1], int(0.05 * sr)):],
                        head_b[:, :min(head_b.shape[1], int(0.05 * sr))]], axis=1) ** 2
    ))
    floor_level = speech_rms * 0.3

    frame_len = max(64, int(0.005 * sr))
    for start in range(0, bridge_samples - frame_len, frame_len):
        frame = bridge[:, start:start + frame_len]
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < floor_level and frame_rms > 1e-10:
            boost = min(floor_level / frame_rms, 3.0)
            bridge[:, start:start + frame_len] *= boost
        elif frame_rms < 1e-10:
            noise = np.random.randn(channels, frame_len) * floor_level * 0.5
            bridge[:, start:start + frame_len] = noise

    return np.clip(bridge, -1.0, 1.0)


def _bridge_mel_inpaint(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
    device: str = "cpu",
) -> np.ndarray:
    """Generate AI-quality bridge via time-varying spectral envelope on shaped noise.

    Uses STFT to extract the time-varying spectral envelope from both sides,
    creates a smooth spectral transition, and applies it to shaped noise.
    This produces a natural-sounding ambience that smoothly transitions between
    the acoustic characteristics of segment A and segment B — without echoing
    or helium artifacts because we never blend actual speech signals.

    Higher quality than Spectral: uses time-varying (frame-by-frame) spectral
    envelopes instead of a single averaged spectrum.
    """
    channels = tail_a.shape[0]
    mono_a = tail_a.mean(axis=0) if channels > 1 else tail_a[0]
    mono_b = head_b.mean(axis=0) if channels > 1 else head_b[0]

    try:
        n_fft = min(1024, len(mono_a), len(mono_b))
        n_fft = n_fft - (n_fft % 2)
        if n_fft < 64:
            return _bridge_crossfade(tail_a, head_b, bridge_samples, sr)

        hop_length = n_fft // 4
        window = torch.hann_window(n_fft)

        a_tensor = torch.from_numpy(mono_a.copy()).float()
        b_tensor = torch.from_numpy(mono_b.copy()).float()

        # STFT of both contexts
        stft_a = torch.stft(a_tensor, n_fft, hop_length=hop_length,
                           window=window, return_complex=True)
        stft_b = torch.stft(b_tensor, n_fft, hop_length=hop_length,
                           window=window, return_complex=True)

        # Number of frames for the bridge
        bridge_frames = max(4, bridge_samples // hop_length)

        # Get spectral envelopes (magnitude only) from each side
        a_mag = stft_a[:, -min(bridge_frames, stft_a.shape[1]):].abs()
        b_mag = stft_b[:, :min(bridge_frames, stft_b.shape[1])].abs()

        n_frames = min(a_mag.shape[1], b_mag.shape[1], bridge_frames)
        a_mag = a_mag[:, :n_frames]
        b_mag = b_mag[:, :n_frames]

        # Smooth each frame's magnitude to get envelope (remove harmonic detail)
        kernel_size = max(3, a_mag.shape[0] // 30)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smooth_a = torch.nn.functional.avg_pool1d(
            a_mag.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(0)
        smooth_b = torch.nn.functional.avg_pool1d(
            b_mag.unsqueeze(0), kernel_size, stride=1, padding=kernel_size // 2
        ).squeeze(0)

        # Interpolate spectral envelopes across time
        alpha = torch.linspace(0, 1, n_frames).unsqueeze(0)  # (1, n_frames)
        interp_mag = smooth_a * (1 - alpha) + smooth_b * alpha

        # Random phase (noise basis)
        random_phase = torch.rand_like(interp_mag) * 2 * np.pi - np.pi
        bridge_stft = interp_mag * torch.exp(1j * random_phase)

        # ISTFT to waveform
        reconstructed = torch.istft(bridge_stft, n_fft, hop_length=hop_length,
                                    window=window, length=bridge_samples)
        result = reconstructed.numpy()

        # Scale to ~40% of speech RMS (audible ambience)
        speech_rms = np.sqrt(np.mean(np.concatenate([mono_a[-int(0.05*sr):], mono_b[:int(0.05*sr)]]) ** 2))
        result_rms = np.sqrt(np.mean(result ** 2))
        if result_rms > 1e-10 and speech_rms > 1e-10:
            result *= (speech_rms * 0.40) / result_rms

        # Expand to channels
        bridge = np.tile(result[np.newaxis, :], (channels, 1))

        # Crossfade edges with surrounding speech (50ms)
        fade_len = min(int(0.050 * sr), bridge_samples // 3)
        if fade_len > 0 and tail_a.shape[1] >= fade_len:
            t = np.linspace(0, np.pi / 2, fade_len)
            bridge[:, :fade_len] = tail_a[:, -fade_len:] * np.cos(t) ** 2 + bridge[:, :fade_len] * np.sin(t) ** 2
        if fade_len > 0 and head_b.shape[1] >= fade_len:
            t = np.linspace(0, np.pi / 2, fade_len)
            bridge[:, -fade_len:] = bridge[:, -fade_len:] * np.cos(t) ** 2 + head_b[:, :fade_len] * np.sin(t) ** 2

        return np.clip(bridge, -1.0, 1.0)

    except Exception as e:
        print(f"[BRIDGE] AI Inpainting bridge failed: {e}", flush=True)
        return _bridge_crossfade(tail_a, head_b, bridge_samples, sr)


# ── AI Voice Regeneration (F5-TTS) ──────────────────────────────────────

_f5tts_model = None


def _load_f5tts():
    """Lazy-load F5-TTS model. Cached across calls."""
    global _f5tts_model
    if _f5tts_model is not None:
        return _f5tts_model
    from f5_tts.api import F5TTS
    print("[BRIDGE] Loading F5-TTS model for AI voice regeneration...", flush=True)
    _f5tts_model = F5TTS()
    print("[BRIDGE] F5-TTS model loaded.", flush=True)
    return _f5tts_model


def _bridge_ai_regeneration(
    tail_a: np.ndarray,
    head_b: np.ndarray,
    bridge_samples: int,
    sr: int,
    full_audio: np.ndarray,
    seg_a_end_sample: int,
    seg_b_start_sample: int,
    device: str = "cpu",
) -> np.ndarray:
    """Generate bridge using F5-TTS native speech infilling (edit_mask mode).

    Instead of generating random filler words, this uses F5-TTS's core
    speech-infilling capability: provide audio context from both sides of the
    edit point with a zero-filled gap, and let the model generate natural
    speech that bridges segment A into segment B seamlessly.
    """
    channels = tail_a.shape[0]
    try:
        import soundfile as sf

        f5tts = _load_f5tts()
        f5_device = f5tts.device

        # F5-TTS constants
        TARGET_SR = 24000  # F5-TTS internal sample rate
        HOP_LENGTH = 256
        N_MEL_CHANNELS = 100
        TARGET_RMS = 0.1

        mono = full_audio.mean(axis=0) if channels > 1 else full_audio[0]

        # ── Step 1: Extract ~3s context from each side ──
        ctx_duration = int(3.0 * sr)
        a_start = max(0, seg_a_end_sample - ctx_duration)
        b_end = min(len(mono), seg_b_start_sample + ctx_duration)
        ctx_a = mono[a_start:seg_a_end_sample].copy()
        ctx_b = mono[seg_b_start_sample:b_end].copy()

        if len(ctx_a) < int(0.3 * sr) or len(ctx_b) < int(0.3 * sr):
            raise ValueError("Context audio too short for speech infilling")

        print(f"[BRIDGE] F5-TTS infilling: ctx_a={len(ctx_a)/sr:.2f}s, "
              f"ctx_b={len(ctx_b)/sr:.2f}s, gap={bridge_samples/sr*1000:.0f}ms", flush=True)

        # ── Step 2: Transcribe both sides with WhisperX ──
        print("[BRIDGE] F5-TTS: Transcribing context with WhisperX...", flush=True)
        import whisperx

        whisper_model = whisperx.load_model("tiny", device="cpu", compute_type="int8")

        # Save contexts to temp files for WhisperX
        tmp_a = tempfile.mktemp(suffix="_ctxA.wav")
        tmp_b = tempfile.mktemp(suffix="_ctxB.wav")
        sf.write(tmp_a, ctx_a, sr)
        sf.write(tmp_b, ctx_b, sr)

        result_a = whisper_model.transcribe(tmp_a, language="en")
        text_a = " ".join(
            seg["text"].strip() for seg in result_a.get("segments", [])
        ).strip()

        result_b = whisper_model.transcribe(tmp_b, language="en")
        text_b = " ".join(
            seg["text"].strip() for seg in result_b.get("segments", [])
        ).strip()

        del whisper_model  # Free memory

        # Clean up temp files
        for p in [tmp_a, tmp_b]:
            try:
                os.unlink(p)
            except OSError:
                pass

        if not text_a or len(text_a) < 2:
            text_a = "speech"
        if not text_b or len(text_b) < 2:
            text_b = "continues"

        # Full text = words from A flowing into words from B
        target_text = f"{text_a} {text_b}"
        print(f"[BRIDGE] F5-TTS: target_text = '{target_text[:100]}'", flush=True)

        # ── Step 3: Resample to 24kHz and normalize RMS ──
        from torchaudio.functional import resample as torch_resample

        ctx_a_t = torch.from_numpy(ctx_a).float().unsqueeze(0)
        ctx_b_t = torch.from_numpy(ctx_b).float().unsqueeze(0)

        if sr != TARGET_SR:
            ctx_a_t = torch_resample(ctx_a_t, sr, TARGET_SR)
            ctx_b_t = torch_resample(ctx_b_t, sr, TARGET_SR)

        # Normalize RMS to match F5-TTS expected input level
        rms_a = torch.sqrt(torch.mean(ctx_a_t ** 2))
        if rms_a > 1e-10:
            ctx_a_t = ctx_a_t * TARGET_RMS / rms_a
        rms_b = torch.sqrt(torch.mean(ctx_b_t ** 2))
        if rms_b > 1e-10:
            ctx_b_t = ctx_b_t * TARGET_RMS / rms_b

        ctx_a_24k_len = ctx_a_t.shape[1]
        ctx_b_24k_len = ctx_b_t.shape[1]

        # ── Step 4: Build mel_cond with zero-filled gap ──
        ctx_a_t = ctx_a_t.to(f5_device)
        ctx_b_t = ctx_b_t.to(f5_device)

        with torch.inference_mode():
            mel_a = f5tts.ema_model.mel_spec(ctx_a_t)  # (1, n_mel, frames)
            mel_a = mel_a.permute(0, 2, 1)  # (1, frames, n_mel)
            mel_b = f5tts.ema_model.mel_spec(ctx_b_t)
            mel_b = mel_b.permute(0, 2, 1)

        # Gap duration in mel frames
        bridge_duration_sec = bridge_samples / sr
        gap_frames = round(bridge_duration_sec * TARGET_SR / HOP_LENGTH)
        if gap_frames < 4:
            gap_frames = 4

        zero_gap = torch.zeros(1, gap_frames, N_MEL_CHANNELS, device=f5_device)
        mel_cond = torch.cat([mel_a, zero_gap, mel_b], dim=1)

        print(f"[BRIDGE] F5-TTS: mel_a={mel_a.shape[1]} frames, gap={gap_frames} frames, "
              f"mel_b={mel_b.shape[1]} frames, total={mel_cond.shape[1]} frames", flush=True)

        # ── Step 5: Build edit_mask ──
        edit_mask = torch.cat([
            torch.ones(1, mel_a.shape[1], dtype=torch.bool, device=f5_device),
            torch.zeros(1, gap_frames, dtype=torch.bool, device=f5_device),
            torch.ones(1, mel_b.shape[1], dtype=torch.bool, device=f5_device),
        ], dim=-1)

        # Pad edit_mask to match mel_cond if needed
        if edit_mask.shape[-1] < mel_cond.shape[1]:
            import torch.nn.functional as F
            edit_mask = F.pad(
                edit_mask,
                (0, mel_cond.shape[1] - edit_mask.shape[-1]),
                value=True,
            )

        # ── Step 6: Tokenize text ──
        from f5_tts.model.utils import convert_char_to_pinyin
        text_list = convert_char_to_pinyin([target_text])

        # ── Step 7: Run CFM model with edit_mask ──
        duration = mel_cond.shape[1]

        print(f"[BRIDGE] F5-TTS: Running speech infilling (duration={duration} frames, "
              f"nfe_step=32)...", flush=True)

        with torch.inference_mode():
            generated, _ = f5tts.ema_model.sample(
                cond=mel_cond,
                text=text_list,
                duration=duration,
                steps=32,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
                edit_mask=edit_mask,
            )

        # ── Step 8: Vocoder: mel → waveform ──
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)  # (1, n_mel, frames)

        with torch.inference_mode():
            if f5tts.mel_spec_type == "vocos":
                full_wav = f5tts.vocoder.decode(gen_mel_spec).cpu().squeeze()
            elif f5tts.mel_spec_type == "bigvgan":
                full_wav = f5tts.vocoder(gen_mel_spec).squeeze(0).cpu().squeeze()
            else:
                full_wav = f5tts.vocoder.decode(gen_mel_spec).cpu().squeeze()

        if full_wav.ndim > 1:
            full_wav = full_wav.mean(dim=0)
        full_wav = full_wav.numpy()

        # Undo RMS normalization — scale back to original level
        orig_rms = np.sqrt(np.mean(np.concatenate([ctx_a[-int(0.1 * sr):],
                                                    ctx_b[:int(0.1 * sr)]]) ** 2))
        if orig_rms < TARGET_RMS and orig_rms > 1e-10:
            full_wav = full_wav * orig_rms / TARGET_RMS

        print(f"[BRIDGE] F5-TTS: Full output wav length={len(full_wav)} samples "
              f"(~{len(full_wav)/TARGET_SR*1000:.0f}ms)", flush=True)

        # ── Step 9: Extract only the bridge (gap) portion ──
        # The full output = ctx_a_portion + bridge_portion + ctx_b_portion
        # Convert mel frame positions to sample positions
        bridge_start_sample = ctx_a_24k_len
        bridge_end_sample = bridge_start_sample + int(gap_frames * HOP_LENGTH)

        # Clamp to output bounds
        bridge_start_sample = min(bridge_start_sample, len(full_wav))
        bridge_end_sample = min(bridge_end_sample, len(full_wav))

        bridge_wav = full_wav[bridge_start_sample:bridge_end_sample]

        if len(bridge_wav) < 100:
            raise ValueError(f"Bridge portion too short: {len(bridge_wav)} samples")

        print(f"[BRIDGE] F5-TTS: Extracted bridge={len(bridge_wav)} samples "
              f"(~{len(bridge_wav)/TARGET_SR*1000:.0f}ms at 24kHz)", flush=True)

        # ── Step 10: Resample to target sr ──
        if TARGET_SR != sr:
            bridge_t = torch.from_numpy(bridge_wav.copy()).float().unsqueeze(0)
            bridge_t = torch_resample(bridge_t, TARGET_SR, sr)
            bridge_wav = bridge_t.squeeze(0).numpy()

        # Trim or pad to exact bridge_samples length
        if len(bridge_wav) > bridge_samples:
            # Center-crop
            start = (len(bridge_wav) - bridge_samples) // 2
            bridge_wav = bridge_wav[start:start + bridge_samples]
        elif len(bridge_wav) < bridge_samples:
            padded = np.zeros(bridge_samples, dtype=bridge_wav.dtype)
            padded[:len(bridge_wav)] = bridge_wav
            bridge_wav = padded

        # ── Step 11: Volume match to surrounding speech ──
        tail_mono = tail_a.mean(axis=0) if channels > 1 else tail_a[0]
        head_mono = head_b.mean(axis=0) if channels > 1 else head_b[0]
        speech_rms = np.sqrt(np.mean(
            np.concatenate([
                tail_mono[-min(len(tail_mono), int(0.05 * sr)):],
                head_mono[:min(len(head_mono), int(0.05 * sr))]
            ]) ** 2
        ))
        wav_rms = np.sqrt(np.mean(bridge_wav ** 2))
        if wav_rms > 1e-10 and speech_rms > 1e-10:
            bridge_wav *= speech_rms / wav_rms

        # ── Step 12: Expand to channels + crossfade edges ──
        bridge = np.tile(bridge_wav[np.newaxis, :], (channels, 1))

        # 80ms crossfade at edges for smooth blending
        fade_len = min(int(0.080 * sr), bridge_samples // 3)
        if fade_len > 0 and tail_a.shape[1] >= fade_len:
            t = np.linspace(0, np.pi / 2, fade_len)
            bridge[:, :fade_len] = (
                tail_a[:, -fade_len:] * np.cos(t) ** 2
                + bridge[:, :fade_len] * np.sin(t) ** 2
            )
        if fade_len > 0 and head_b.shape[1] >= fade_len:
            t = np.linspace(0, np.pi / 2, fade_len)
            bridge[:, -fade_len:] = (
                bridge[:, -fade_len:] * np.cos(t) ** 2
                + head_b[:, :fade_len] * np.sin(t) ** 2
            )

        print("[BRIDGE] F5-TTS speech infilling bridge generated successfully.", flush=True)
        return np.clip(bridge, -1.0, 1.0)

    except Exception as e:
        import traceback
        print(f"[BRIDGE] AI Regeneration (infilling) failed: {e}", flush=True)
        traceback.print_exc()
        return _bridge_room_tone(tail_a, head_b, bridge_samples, sr, None)


# =============================================================================
# Method 13: Boundary Clone — F5-TTS word-level co-articulation fix
# =============================================================================

def apply_boundary_clone(
    audio_path: str,
    boundary_timestamps: List[float],
    segments_data: list,
    device: str = "cpu",
    clone_ms: float = 80.0,
    fade_ms: float = 20.0,
) -> str:
    """
    Fix co-articulation artifacts at edit boundaries by synthesising the boundary
    words in their new phonetic context and crossfading within each word.

    For each cut boundary between word A (pre-cut) and word B (post-cut):
      1. Identify the text of A and B from segments_data.
      2. Clone the speaker's voice using 3s of audio before the boundary.
      3. Synthesise "{word_A} {word_B}" with F5-TTS.
      4. Estimate the split point in the synthesis proportionally by character count.
      5. Crossfade: original tail of A → synthesised tail of A (last clone_ms of A).
      6. Crossfade: synthesised head of B → original head of B (first clone_ms of B).

    segments_data: list of dicts with keys word, start, end (flattened word list).
    Falls back to crossfade if F5-TTS is unavailable or synthesis fails.
    """
    if not boundary_timestamps or not segments_data:
        return audio_path

    audio, sr = _load_audio(audio_path)
    original_shape = audio.shape
    clone_samples = int(clone_ms / 1000.0 * sr)
    fade_samples = int(fade_ms / 1000.0 * sr)

    # Flatten all words sorted by start time
    words = sorted(segments_data, key=lambda w: w.get('start', 0))

    try:
        f5tts = _load_f5tts()
    except Exception as e:
        logger.warning(f"BOUNDARY_CLONE: F5-TTS load failed ({e}), falling back to crossfade")
        return process_audio_at_boundaries(audio_path, boundary_timestamps, ContinuityMethod.CROSSFADE, device)

    for ts in boundary_timestamps:
        center = int(ts * sr)
        if center <= 0 or center >= audio.shape[1]:
            continue

        # Find the word ending closest to and before ts (pre-cut word)
        pre_word = None
        pre_word_end = None
        for w in reversed(words):
            if w.get('end', 0) <= ts + 0.05:
                pre_word = w.get('word', '').strip()
                pre_word_end = int(w.get('end', ts) * sr)
                break

        # Find the word starting closest to and after ts (post-cut word)
        post_word = None
        post_word_start = None
        for w in words:
            if w.get('start', 0) >= ts - 0.05:
                post_word = w.get('word', '').strip()
                post_word_start = int(w.get('start', ts) * sr)
                break

        if not pre_word or not post_word:
            logger.debug(f"BOUNDARY_CLONE: could not find boundary words at {ts:.2f}s, skipping")
            _apply_micro_fade(audio, center)
            continue

        logger.info(f"BOUNDARY_CLONE: boundary {ts:.2f}s — '{pre_word}' | '{post_word}'")

        try:
            import soundfile as sf

            # Extract 3s voice reference ending at boundary
            ref_dur = int(3.0 * sr)
            ref_start = max(0, center - ref_dur)
            ref_audio = audio[:, ref_start:center]
            mono_ref = ref_audio.mean(axis=0) if ref_audio.ndim > 1 else ref_audio[0]

            fd, ref_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            sf.write(ref_wav, mono_ref, sr)

            # Build reference text from words that fall within the reference window
            ref_text_words = [
                w.get('word', '').strip()
                for w in words
                if w.get('start', 0) >= ref_start / sr and w.get('end', 0) <= ts + 0.01
            ]
            ref_text = ' '.join(ref_text_words).strip() or pre_word

            # Synthesise "pre_word post_word" in the speaker's voice
            gen_text = f"{pre_word} {post_word}"
            fd, synth_wav = tempfile.mkstemp(suffix='.wav')
            os.close(fd)

            f5tts.infer(
                ref_file=ref_wav,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave=synth_wav,
                remove_silence=False,
            )

            os.unlink(ref_wav)

            synth_audio, synth_sr = _load_audio(synth_wav)
            os.unlink(synth_wav)

            # Resample synthesised audio to match original sr
            if synth_sr != sr:
                import torch as _torch
                synth_tensor = _torch.from_numpy(synth_audio).float()
                synth_audio = torchaudio.functional.resample(synth_tensor, synth_sr, sr).numpy()

            synth_mono = synth_audio.mean(axis=0) if synth_audio.ndim > 1 else synth_audio.flatten()
            synth_len = len(synth_mono)

            # Volume-match synthesis to original
            orig_region = audio[:, max(0, center - int(0.2 * sr)):min(audio.shape[1], center + int(0.2 * sr))]
            orig_rms = np.sqrt(np.mean(orig_region ** 2)) if orig_region.size > 0 else 0.1
            synth_rms = np.sqrt(np.mean(synth_mono ** 2))
            if synth_rms > 1e-10:
                synth_mono = synth_mono * (orig_rms / synth_rms)

            # Estimate split point in synthesis by character proportion
            pre_chars = len(pre_word)
            post_chars = len(post_word)
            total_chars = max(pre_chars + post_chars, 1)
            split_sample = int(synth_len * pre_chars / total_chars)
            split_sample = max(clone_samples, min(synth_len - clone_samples, split_sample))

            # Synthesised tail of pre_word (last clone_ms of split)
            synth_pre_tail = synth_mono[max(0, split_sample - clone_samples):split_sample]
            # Synthesised head of post_word (first clone_ms after split)
            synth_post_head = synth_mono[split_sample:split_sample + clone_samples]

            # --- Apply to original audio in-place ---

            # Crossfade: original pre_word tail → synthetic tail
            # Region: pre_word_end - clone_samples → pre_word_end
            if pre_word_end is not None:
                pre_end = min(pre_word_end, audio.shape[1])
                pre_start_sample = max(0, pre_end - clone_samples)
                actual_clone = pre_end - pre_start_sample
                if actual_clone > fade_samples * 2 and len(synth_pre_tail) >= actual_clone:
                    synth_tail = synth_pre_tail[-actual_clone:]
                    t = np.linspace(0, np.pi / 2, actual_clone)
                    fade_out = np.cos(t) ** 2   # original fades out
                    fade_in = np.sin(t) ** 2    # synthetic fades in
                    for ch in range(audio.shape[0]):
                        audio[ch, pre_start_sample:pre_end] = (
                            audio[ch, pre_start_sample:pre_end] * fade_out +
                            synth_tail * fade_in
                        )

            # Crossfade: synthetic head → original post_word head
            # Region: post_word_start → post_word_start + clone_samples
            if post_word_start is not None:
                post_start = max(0, post_word_start)
                post_end_sample = min(audio.shape[1], post_start + clone_samples)
                actual_clone = post_end_sample - post_start
                if actual_clone > fade_samples * 2 and len(synth_post_head) >= actual_clone:
                    synth_head = synth_post_head[:actual_clone]
                    t = np.linspace(0, np.pi / 2, actual_clone)
                    fade_out = np.cos(t) ** 2   # synthetic fades out
                    fade_in = np.sin(t) ** 2    # original fades in
                    for ch in range(audio.shape[0]):
                        audio[ch, post_start:post_end_sample] = (
                            synth_head * fade_out +
                            audio[ch, post_start:post_end_sample] * fade_in
                        )

            logger.info(f"BOUNDARY_CLONE: applied synthesis blend at {ts:.2f}s")

        except Exception as e:
            logger.warning(f"BOUNDARY_CLONE: synthesis failed at {ts:.2f}s ({e}), using crossfade")
            _apply_micro_fade(audio, center)
            audio = _blend_crossfade_inplace(audio, center, sr)

    assert audio.shape == original_shape, f"Sample count changed: {original_shape} → {audio.shape}"

    fd, output_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    if audio.shape[0] == 1:
        sf.write(output_path, audio[0], sr)
    else:
        sf.write(output_path, audio.T, sr)

    logger.info(f"BOUNDARY_CLONE: complete → {output_path}")
    return output_path
