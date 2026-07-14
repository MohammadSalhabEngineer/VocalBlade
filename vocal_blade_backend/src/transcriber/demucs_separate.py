"""
Demucs v4 (htdemucs) source separation for audio continuity improvement.

Separates an audio file into vocals (speech) and background (everything else)
so continuity methods can be applied to the speech track only, while the
background gets a simple crossfade — preventing music tempo jumps at cuts.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Cached separator to avoid reloading the model between boundaries
_separator = None
_separator_device: str = ""


def _load_separator(device: str = "cuda"):
    """Lazy-load the Demucs htdemucs separator. Cached across calls."""
    global _separator, _separator_device
    if _separator is not None and _separator_device == device:
        return _separator

    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        model = get_model("htdemucs")
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                model = model.cuda()
        model.eval()
        _separator = (model, apply_model)
        _separator_device = device
        logger.info(f"Demucs htdemucs loaded on {device}")
        return _separator
    except Exception as e:
        logger.warning(f"Demucs load failed: {e}")
        return None


def separate_speech_background(
    audio_path: str,
    temp_files: list,
    device: str = "cuda",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Separate an audio file into vocals and background using Demucs htdemucs.

    Returns (vocals_path, background_path) as temp WAV files.
    Returns (None, None) if separation fails — caller should fall back to
    processing the original audio as-is.

    The background track is: drums + bass + other (everything except vocals).
    """
    sep = _load_separator(device)
    if sep is None:
        return None, None

    model, apply_model = sep

    try:
        import torch
        import torchaudio
        import soundfile as sf

        # Load audio via torchaudio
        waveform, sr = torchaudio.load(audio_path)  # (channels, samples)

        # Demucs expects (batch, channels, samples), sr=44100
        target_sr = model.samplerate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr

        # Ensure stereo (htdemucs requires 2 channels)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        wave_input = waveform.unsqueeze(0)  # (1, 2, samples)

        if device == "cuda" and torch.cuda.is_available():
            wave_input = wave_input.cuda()

        logger.info(f"Demucs: separating {Path(audio_path).name} ({waveform.shape[1]/sr:.1f}s)")

        with torch.no_grad():
            sources = apply_model(
                model, wave_input,
                shifts=1,
                overlap=0.25,
                progress=False,
                device=wave_input.device,
            )
        # sources: (batch=1, stems, channels, samples)
        # htdemucs stems: ['drums', 'bass', 'other', 'vocals']
        source_names = model.sources

        vocals_idx = source_names.index("vocals")
        vocals = sources[0, vocals_idx].cpu()  # (2, samples)

        # Background = sum of all other stems
        bg_indices = [i for i, n in enumerate(source_names) if n != "vocals"]
        background = sum(sources[0, i] for i in bg_indices).cpu()  # (2, samples)

        # Save to temp WAV files
        fd_v, vocals_path = tempfile.mkstemp(suffix="_vocals.wav")
        os.close(fd_v)
        temp_files.append(vocals_path)
        sf.write(vocals_path, vocals.T.numpy(), sr)

        fd_b, bg_path = tempfile.mkstemp(suffix="_background.wav")
        os.close(fd_b)
        temp_files.append(bg_path)
        sf.write(bg_path, background.T.numpy(), sr)

        logger.info(f"Demucs: separation complete → vocals={vocals_path}, bg={bg_path}")
        return vocals_path, bg_path

    except Exception as e:
        logger.error(f"Demucs separation failed: {e}", exc_info=True)
        return None, None


def mix_stems(
    vocals_path: str,
    background_path: str,
    output_path: str,
    vocals_gain: float = 1.0,
    bg_gain: float = 1.0,
) -> bool:
    """
    Remix vocals and background into a single audio file.
    Uses ffmpeg amix so length mismatches are handled gracefully.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", vocals_path,
                "-i", background_path,
                "-filter_complex",
                f"[0:a]volume={vocals_gain}[v];[1:a]volume={bg_gain}[b];[v][b]amix=inputs=2:duration=first:dropout_transition=0[out]",
                "-map", "[out]",
                "-acodec", "pcm_s16le",
                output_path,
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.warning(f"Demucs remix failed: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logger.error(f"Demucs remix error: {e}")
        return False


def apply_demucs_continuity(
    audio_path: str,
    boundaries: list,
    continuity_method,
    temp_files: list,
    device: str = "cuda",
) -> Optional[str]:
    """
    High-level wrapper: separate audio into speech + background, apply continuity
    to speech only, crossfade background at boundaries, then remix.

    Returns path to processed audio file, or None if the pipeline fails
    (caller should fall back to single-track processing).
    """
    from .audio_continuity import process_audio_at_boundaries

    # Step 1: Separate
    vocals_path, bg_path = separate_speech_background(audio_path, temp_files, device)
    if vocals_path is None:
        return None

    logger.info("Demucs: applying continuity to speech track")

    # Step 2: Apply full continuity method to vocals track
    try:
        processed_vocals = process_audio_at_boundaries(
            vocals_path, boundaries, continuity_method, device=device
        )
        temp_files.append(processed_vocals)
    except Exception as e:
        logger.warning(f"Demucs: vocals continuity failed: {e} — skipping")
        return None

    # Step 3: Apply simple crossfade to background track at each boundary
    from .audio_continuity import ContinuityMethod
    try:
        processed_bg = process_audio_at_boundaries(
            bg_path, boundaries, ContinuityMethod.CROSSFADE, device=device
        )
        temp_files.append(processed_bg)
    except Exception as e:
        logger.warning(f"Demucs: background continuity failed: {e} — using raw background")
        processed_bg = bg_path

    # Step 4: Remix vocals + background
    fd_out, remixed_path = tempfile.mkstemp(suffix="_demucs_remixed.wav")
    os.close(fd_out)
    temp_files.append(remixed_path)

    if not mix_stems(processed_vocals, processed_bg, remixed_path):
        return None

    logger.info(f"Demucs: remix complete → {remixed_path}")
    return remixed_path
