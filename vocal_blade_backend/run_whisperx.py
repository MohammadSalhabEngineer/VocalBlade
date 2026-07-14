#!/usr/bin/env python
"""
Simple WhisperX transcription with alignment and diarization.
Mirrors the flow in src/transcriber/resolvers.py but runs as a CLI script.
"""
import argparse
import json
import os
import time
import tempfile
import shutil
from pathlib import Path

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


def main():
    parser = argparse.ArgumentParser(description="WhisperX transcription with alignment + diarization")
    parser.add_argument("audio_path", type=Path, help="Path to audio file")
    parser.add_argument("--model", default="base", help="Whisper model size (e.g., tiny, base, small, medium, large-v3)")
    parser.add_argument("--language", default=None, help="Language code, e.g., en; leave empty for auto-detect")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for transcription")
    parser.add_argument("--return-words", action="store_true", help="Return word-level timings")
    parser.add_argument("--force-download", action="store_true", help="Force re-download of models if cache is corrupted")
    parser.add_argument("--clean-cache", action="store_true", help="Delete cached model files before loading")
    parser.add_argument("--cache-dir", type=Path, help="Optional HuggingFace cache directory to use")
    args = parser.parse_args()

    if not args.audio_path.exists():
        parser.error(f"Audio file not found: {args.audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[setup] device={device}, compute_type={compute_type}")

    # Configure cache directory if provided
    if args.cache_dir:
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(args.cache_dir)
        os.environ["HF_HOME"] = str(args.cache_dir)
        print(f"[cache] using custom cache dir: {args.cache_dir}")

    from huggingface_hub.constants import HF_HUB_CACHE

    repo_id = f"Systran/faster-whisper-{args.model}"
    repo_cache_dir = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"

    def wipe_repo_cache():
        if repo_cache_dir.exists():
            print(f"[cache] deleting cached files for {repo_id} at {repo_cache_dir}")
            shutil.rmtree(repo_cache_dir, ignore_errors=True)

    if args.clean_cache:
        wipe_repo_cache()

    print(f"[load] loading whisper model '{args.model}'...")

    def load_whisper_model(force_download: bool = False):
        if force_download:
            os.environ["HF_HUB_FORCE_DOWNLOAD"] = "1"
        return whisperx.load_model(
            args.model,
            device,
            compute_type=compute_type,
        )

    try:
        model = load_whisper_model(force_download=args.force_download)
    except OSError as e:
        if "Consistency check failed" in str(e):
            print("[load] cache consistency error detected; clearing repo cache and retrying...")
            wipe_repo_cache()
            print("[load] retrying with force download and isolated temp cache...")
            with tempfile.TemporaryDirectory(prefix="whisperx_cache_") as tmp_cache:
                os.environ["HF_HOME"] = tmp_cache  # isolate cache to avoid corrupted files
                os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(tmp_cache) / "hub")
                model = load_whisper_model(force_download=True)
        else:
            raise

    print(f"[audio] loading audio from {args.audio_path}...")
    audio = whisperx.load_audio(str(args.audio_path))

    print("[transcribe] starting transcription...")
    t0 = time.time()
    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)
    segments = result.get("segments", [])
    language = result.get("language")
    print(f"[transcribe] done in {time.time() - t0:.2f}s; detected language={language}; segments={len(segments)}")

    # Alignment
    if segments:
        print("[align] loading alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

        print("[align] aligning segments...")
        aligned = whisperx.align(
            segments,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=args.return_words,
        )
        segments = aligned.get("segments", segments)
        print(f"[align] alignment complete; segments={len(segments)}")
    else:
        print("[align] skipped (no segments from transcription)")

    # Diarization
    print("[diarize] running diarization pipeline...")
    diarize_model = DiarizationPipeline(use_auth_token=os.environ["HF_TOKEN"], device=device)
    diarize_segments = diarize_model(audio)
    try:
        diar_len = (
            len(diarize_segments["segments"])
            if isinstance(diarize_segments, dict) and "segments" in diarize_segments
            else len(diarize_segments)
        )
    except Exception:
        diar_len = "unknown"
    print(f"[diarize] diarization complete; diarization segments={diar_len}")

    print("[diarize] assigning speakers to words...")
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, {"segments": segments})
    segments = result_with_speakers.get("segments", segments)
    print("[diarize] speaker assignment done")

    # Output
    full_text = " ".join([s.get("text", "") for s in segments])
    print("\n[output] full text:")
    print(full_text)

    output_path = Path("segments_out.json")
    output_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    print(f"[output] wrote detailed segments to {output_path.resolve()}")


if __name__ == "__main__":
    main()
