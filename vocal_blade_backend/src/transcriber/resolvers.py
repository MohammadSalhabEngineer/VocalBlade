"""
Business logic for transcription operations.
Separated from schema definitions for better maintainability.
"""
import json
import os
import subprocess
import tempfile
import time
import uuid
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

import soundfile as sf
import torch
import whisperx
from django.core.files.uploadedfile import UploadedFile

from .types import (
    TranscriptionResponse,
    TranscriptionSegment,
    WordTiming,
    TranscriptionStatus,
    ModelSize,
    TranscriptionInput,
    SimpleTranscriptionInput,
    TimeRange,
    ExportResponse,
    ContinuityMethod,
)
from asgiref.sync import sync_to_async
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)


class TranscriptionService:
    """Service class for handling transcription operations."""

    def __init__(self):
        self._model_cache = {}

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def compute_type(self) -> str:
        return "float16" if self.device == "cuda" else "int8"

    def _get_model(self, model_size: str):
        """Get or create a cached model instance."""
        cache_key = (model_size, self.device)
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = whisperx.load_model(
                model_size,
                self.device,
                compute_type=self.compute_type
            )
        return self._model_cache[cache_key]

    def _save_uploaded_file(self, audio_file: UploadedFile) -> str:
        """Save uploaded file to temporary location."""
        suffix = Path(audio_file.name).suffix if audio_file.name else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            for chunk in audio_file.chunks():
                tmp_file.write(chunk)
            return tmp_file.name

    def _convert_segments(
        self,
        segments: List[Dict[str, Any]],
        include_words: bool = False
    ) -> List[TranscriptionSegment]:
        """Convert whisperx segments to GraphQL types."""
        result_segments = []

        for idx, segment in enumerate(segments):
            words = None
            if include_words and 'words' in segment:
                words = [
                    WordTiming(
                        word=w.get('word', ''),
                        start=w.get('start', 0.0),
                        end=w.get('end', 0.0),
                        score=w.get('score')
                    )
                    for w in segment['words']
                ]

            result_segments.append(
                TranscriptionSegment(
                    id=idx,
                    text=segment.get('text', ''),
                    start=segment.get('start', 0.0),
                    end=segment.get('end', 0.0),
                    score=segment.get('score'),
                    speaker=segment.get('speaker'),
                    words=words
                )
            )

        return result_segments

    async def transcribe_audio_full(
        self,
        audio_file: UploadedFile,
        params: TranscriptionInput
    ) -> TranscriptionResponse:
        """
        Full transcription with alignment and diarization.
        Runs in executor to avoid blocking the event loop.
        """
        return await sync_to_async(
            self._transcribe_audio_full_sync
        )(
            audio_file,
            params
        )

    def _transcribe_audio_full_sync(
        self,
        audio_file: UploadedFile,
        params: TranscriptionInput
    ) -> TranscriptionResponse:
        """Synchronous implementation of full transcription."""
        tmp_file_path = None
        start_time = time.time()

        try:
            # Save uploaded file
            tmp_file_path = self._save_uploaded_file(audio_file)

            # Load model and audio
            model = self._get_model(params.model.value)
            audio = whisperx.load_audio(tmp_file_path)

            # Transcribe
            result = model.transcribe(
                audio,
                batch_size=params.batch_size or 8,
                language=params.language
            )

            segments = result.get("segments", [])
            language = result.get("language", "")

            # Alignment
            if params.enable_alignment and segments:
                try:
                    model_a, metadata = whisperx.load_align_model(
                        language_code=language,
                        device=self.device
                    )
                    aligned_result = whisperx.align(
                        segments,
                        model_a,
                        metadata,
                        audio,
                        self.device,
                        return_char_alignments=params.return_word_timings
                    )
                    segments = aligned_result.get("segments", segments)
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")

            # Diarization
            if params.enable_diarization:
                try:
                    diarize_model = DiarizationPipeline(
                        use_auth_token=os.environ["HF_TOKEN"],
                        device=self.device,
                    )
                    diarize_segments = diarize_model(audio)
                    result_with_speakers = whisperx.assign_word_speakers(
                        diarize_segments,
                        {"segments": segments}
                    )
                    segments = result_with_speakers.get("segments", segments)
                except Exception as e:
                    logger.warning(f"Diarization failed: {e}")

            # Calculate duration from actual file (not last segment timestamp)
            # Using ffprobe ensures we get the real file length, preventing
            # truncation when the video extends beyond the last spoken word.
            try:
                dur_result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries',
                     'format=duration', '-of', 'csv=p=0', tmp_file_path],
                    capture_output=True, text=True, timeout=10
                )
                duration = float(dur_result.stdout.strip())
            except (ValueError, AttributeError, subprocess.TimeoutExpired):
                # Fallback to segment-based duration
                duration = max(
                    [s.get('end', 0) for s in segments],
                    default=0
                )

            # Convert segments
            converted_segments = self._convert_segments(
                segments,
                include_words=params.return_word_timings
            )

            # Create full text
            full_text = " ".join([s.text for s in converted_segments])

            processing_time = time.time() - start_time

            return TranscriptionResponse(
                status=TranscriptionStatus.SUCCESS,
                text=full_text,
                segments=converted_segments,
                language=language,
                language_score=result.get("language_score"),
                duration=duration,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return TranscriptionResponse(
                status=TranscriptionStatus.FAILED,
                error_message=str(e),
                error_code="TRANSCRIPTION_FAILED",
                error_details=json.dumps({"exception": type(e).__name__})
            )

        finally:
            # Cleanup temporary file
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")

    async def export_edited_media(
        self,
        media_file: UploadedFile,
        keep_ranges: List[TimeRange],
        continuity_method: ContinuityMethod = ContinuityMethod.SMART_SILENCE,
        video_interpolation_method=None,
        segments_json: str = None,
    ) -> ExportResponse:
        """Export edited media with only the kept time ranges."""
        from .types import VideoInterpolationMethod
        if video_interpolation_method is None:
            video_interpolation_method = VideoInterpolationMethod.NONE
        return await sync_to_async(self._export_edited_media_sync)(
            media_file, keep_ranges, continuity_method, video_interpolation_method,
            segments_json=segments_json,
        )

    def _export_edited_media_sync(
        self,
        media_file: UploadedFile,
        keep_ranges: List[TimeRange],
        continuity_method: ContinuityMethod = ContinuityMethod.SMART_SILENCE,
        video_interpolation_method=None,
        segments_json: str = None,
    ) -> ExportResponse:
        """Synchronous implementation of media export."""
        from django.conf import settings
        from .audio_continuity import process_audio_at_boundaries
        from .types import VideoInterpolationMethod
        import shutil

        if video_interpolation_method is None:
            video_interpolation_method = VideoInterpolationMethod.NONE

        tmp_input_path = None
        processed_audio_path = None
        concat_video_path = None
        temp_files = []  # Track all temp files for cleanup

        # Longer timeout for heavy methods
        timeout = 900 if (
            video_interpolation_method == VideoInterpolationMethod.RIFE or
            continuity_method in (ContinuityMethod.NEURAL_SMOOTH, ContinuityMethod.SMART_CUT, ContinuityMethod.SADTALKER_CUT)
        ) else (
            600 if continuity_method in (ContinuityMethod.AI_INPAINTING, ContinuityMethod.ENCODEC) else 300
        )

        try:
            # Save uploaded media
            tmp_input_path = self._save_uploaded_file(media_file)
            input_ext = Path(tmp_input_path).suffix or '.mp4'

            # Detect if file has video stream
            probe_result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v',
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0',
                 tmp_input_path],
                capture_output=True, text=True
            )
            has_video = 'video' in probe_result.stdout

            # Output path
            exports_dir = os.path.join(settings.MEDIA_ROOT, 'exports')
            os.makedirs(exports_dir, exist_ok=True)
            output_name = f"{uuid.uuid4()}{input_ext}"
            output_path = os.path.join(exports_dir, output_name)

            # No ranges = no edits → copy full file as-is
            if len(keep_ranges) == 0:
                shutil.copy2(tmp_input_path, output_path)
                file_url = f"/media/exports/{output_name}"
                return ExportResponse(
                    status=TranscriptionStatus.SUCCESS,
                    file_url=file_url,
                )

            # Single-range optimization (no boundaries to smooth)
            if len(keep_ranges) == 1:
                dur_result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'csv=p=0', tmp_input_path],
                    capture_output=True, text=True
                )
                total_dur = float(dur_result.stdout.strip()) if dur_result.stdout.strip() else 0
                r = keep_ranges[0]
                if r.start <= 0.1 and (total_dur == 0 or r.end >= total_dur - 0.1):
                    # Full file — just copy, no re-encode
                    shutil.copy2(tmp_input_path, output_path)
                    file_url = f"/media/exports/{output_name}"
                    return ExportResponse(
                        status=TranscriptionStatus.SUCCESS,
                        file_url=file_url,
                    )
                else:
                    # Single range — simple trim, no concat needed
                    cmd = ['ffmpeg', '-y', '-i', tmp_input_path,
                           '-ss', str(r.start), '-to', str(r.end),
                           '-c', 'copy', output_path]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                    if result.returncode != 0:
                        logger.error(f"ffmpeg trim error: {result.stderr}")
                        return ExportResponse(
                            status=TranscriptionStatus.FAILED,
                            error_message=f"FFmpeg trim failed: {result.stderr[:500]}"
                        )
                    file_url = f"/media/exports/{output_name}"
                    return ExportResponse(
                        status=TranscriptionStatus.SUCCESS,
                        file_url=file_url,
                    )

            # =================================================================
            # Multi-range: hard cuts + smart audio continuity
            # Phase 1: Snap boundaries to optimal cut points (WhisperX-aware)
            # Phase 2: Extract and concatenate segments
            # Phase 3: Apply micro-fade + audio boundary smoothing
            # Phase 4: Finalize
            # =================================================================
            logger.info(f"Processing export with {len(keep_ranges)} ranges, "
                        f"continuity={continuity_method.value}")

            # Phase 1: Snap cut-points to optimal low-energy/zero-crossing points
            from .audio_continuity import snap_boundaries
            keep_ranges = snap_boundaries(tmp_input_path, keep_ranges)

            # Phase 2: Extract segments and concatenate
            if has_video:
                concat_path, segment_paths = self._extract_concat_muxed(
                    tmp_input_path, keep_ranges, temp_files
                )
            else:
                concat_path, segment_paths = self._extract_concat_audio(
                    tmp_input_path, keep_ranges, temp_files
                )
            temp_files.append(concat_path)

            final_path = concat_path

            # Phase 2.5 + Phase 3 — video interpolation then audio continuity
            neural_smooth_applied = False
            if len(keep_ranges) > 1:
                boundaries = self._compute_boundaries(segment_paths, keep_ranges)

                # Phase 2.5a: Smart Edit (NEURAL_SMOOTH) — combined room-tone + face morph
                # Handles both audio and video simultaneously; skips Phase 3 when applied.
                if continuity_method == ContinuityMethod.NEURAL_SMOOTH and has_video:
                    try:
                        from .neural_smooth import apply_neural_smooth
                        smoothed = apply_neural_smooth(
                            final_path, boundaries, temp_files,
                            source_audio_path=tmp_input_path,
                            device=self.device,
                        )
                        if smoothed != final_path:
                            temp_files.append(smoothed)
                            final_path = smoothed
                            neural_smooth_applied = True
                    except Exception as e:
                        logger.error(f"Smart Edit (neural_smooth) failed: {e}", exc_info=True)

                # Phase 2.5a-2: Smart Cut (SMART_CUT) — FOMM video + speech crossfade audio
                elif continuity_method == ContinuityMethod.SMART_CUT and has_video:
                    try:
                        from .neural_smooth import apply_smart_cut
                        smoothed = apply_smart_cut(
                            final_path, boundaries, temp_files,
                            source_audio_path=tmp_input_path,
                            keep_ranges=keep_ranges,
                            device=self.device,
                        )
                        if smoothed != final_path:
                            temp_files.append(smoothed)
                            final_path = smoothed
                            neural_smooth_applied = True
                    except Exception as e:
                        logger.error(f"Smart Cut (fomm) failed: {e}", exc_info=True)

                # Phase 2.5a-3: SadTalker Cut — F5-TTS audio + SadTalker face reenactment video
                elif continuity_method == ContinuityMethod.SADTALKER_CUT and has_video:
                    try:
                        from .sadtalker_bridge import apply_sadtalker_cut
                        smoothed = apply_sadtalker_cut(
                            final_path, boundaries, temp_files,
                            source_audio_path=tmp_input_path,
                            keep_ranges=keep_ranges,
                            device=self.device,
                        )
                        if smoothed != final_path:
                            temp_files.append(smoothed)
                            final_path = smoothed
                            neural_smooth_applied = True
                    except Exception as e:
                        logger.error(f"SadTalker Cut failed: {e}", exc_info=True)

                # Phase 2.5b: Standard video interpolation (RIFE/BLEND), only when not Smart Edit
                elif has_video and video_interpolation_method != VideoInterpolationMethod.NONE:
                    try:
                        from .video_interpolation import interpolate_at_cuts
                        interpolated = interpolate_at_cuts(
                            final_path, boundaries, video_interpolation_method,
                            temp_files, device=self.device
                        )
                        if interpolated != final_path:
                            temp_files.append(interpolated)
                            final_path = interpolated
                    except Exception as e:
                        logger.error(f"Video interpolation failed: {e}", exc_info=True)

            # Phase 3: In-place audio boundary smoothing (skipped when Smart Edit handled it)
            if len(keep_ranges) > 1 and not neural_smooth_applied:
                try:
                    boundaries = self._compute_boundaries(segment_paths, keep_ranges)

                    # Parse segments_json once for BOUNDARY_CLONE
                    parsed_words = None
                    if continuity_method == ContinuityMethod.BOUNDARY_CLONE and segments_json:
                        try:
                            import json as _json
                            segs = _json.loads(segments_json)
                            parsed_words = []
                            for seg in segs:
                                for w in (seg.get('words') or []):
                                    parsed_words.append(w)
                        except Exception as _e:
                            logger.warning(f"BOUNDARY_CLONE: failed to parse segments_json: {_e}")

                    if has_video:
                        if continuity_method == ContinuityMethod.DEMUCS:
                            processed = self._apply_demucs_continuity_to_video(
                                final_path, boundaries, temp_files
                            )
                        elif continuity_method == ContinuityMethod.BOUNDARY_CLONE:
                            processed = self._apply_boundary_clone_to_video(
                                final_path, boundaries, parsed_words or [], temp_files
                            )
                        else:
                            processed = self._apply_audio_continuity_to_video(
                                final_path, segment_paths,
                                continuity_method, boundaries, temp_files
                            )
                        if processed:
                            temp_files.append(processed)
                            final_path = processed
                    else:
                        # Audio-only: extract to WAV, process, use as final
                        fd, extracted_audio = tempfile.mkstemp(suffix='.wav')
                        os.close(fd)
                        temp_files.append(extracted_audio)
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', concat_path,
                             '-vn', '-acodec', 'pcm_s16le', extracted_audio],
                            capture_output=True, text=True, timeout=120
                        )
                        if continuity_method == ContinuityMethod.DEMUCS:
                            from .demucs_separate import apply_demucs_continuity
                            from .types import ContinuityMethod as CM
                            processed_audio = apply_demucs_continuity(
                                extracted_audio, boundaries, CM.ROOM_TONE, temp_files, device=self.device
                            ) or process_audio_at_boundaries(
                                extracted_audio, boundaries, CM.ROOM_TONE, device=self.device
                            )
                        elif continuity_method == ContinuityMethod.BOUNDARY_CLONE:
                            from .audio_continuity import apply_boundary_clone
                            processed_audio = apply_boundary_clone(
                                extracted_audio, boundaries,
                                parsed_words or [], device=self.device
                            )
                        else:
                            processed_audio = process_audio_at_boundaries(
                                extracted_audio, boundaries, continuity_method, device=self.device
                            )
                        temp_files.append(processed_audio)
                        final_path = processed_audio
                except Exception as e:
                    logger.error(f"Audio continuity processing failed: {e}", exc_info=True)

            # Phase 4: Finalize
            if has_video and final_path.endswith('.mkv'):
                finalize_result = subprocess.run(
                    ['ffmpeg', '-y', '-i', final_path,
                     '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                     output_path],
                    capture_output=True, text=True, timeout=300
                )
                if finalize_result.returncode != 0:
                    logger.warning(f"Finalize failed: {finalize_result.stderr[:200]}")
                    shutil.copy2(final_path, output_path)
            else:
                shutil.copy2(final_path, output_path)

            file_url = f"/media/exports/{output_name}"
            return ExportResponse(
                status=TranscriptionStatus.SUCCESS,
                file_url=file_url,
            )

        except subprocess.TimeoutExpired:
            return ExportResponse(
                status=TranscriptionStatus.FAILED,
                error_message=f"Export timed out ({timeout // 60} minutes)"
            )
        except Exception as e:
            logger.error(f"Export error: {e}", exc_info=True)
            return ExportResponse(
                status=TranscriptionStatus.FAILED,
                error_message=str(e)
            )
        finally:
            # Cleanup all temporary files
            for path in [tmp_input_path] + temp_files:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    def _extract_concat_muxed(
        self,
        input_path: str,
        keep_ranges: List[TimeRange],
        temp_files: List[str],
    ) -> tuple:
        """Building-block export: extract muxed (audio+video) segments and concatenate.

        Each range becomes one segment with both audio and video,
        so A/V sync is guaranteed (they're never separated).

        Returns (concat_path, segment_paths) so caller can probe actual durations.
        """
        segment_paths = []

        for i, r in enumerate(keep_ranges):
            fd, seg_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)
            temp_files.append(seg_path)
            segment_paths.append(seg_path)

            seg_duration = r.end - r.start
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(r.start),
                '-i', input_path,
                '-t', str(seg_duration),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18',
                '-c:a', 'pcm_s16le',
                '-avoid_negative_ts', 'make_zero',
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Muxed segment {i} extraction failed: {result.stderr[:200]}")

        concat_path = self._concat_segments(segment_paths, temp_files)
        return concat_path, segment_paths

    def _extract_concat_audio(
        self,
        input_path: str,
        keep_ranges: List[TimeRange],
        temp_files: List[str],
    ) -> tuple:
        """Building-block export for audio-only files.

        Returns (concat_path, segment_paths) so caller can probe actual durations.
        """
        segment_paths = []
        # Detect input format
        ext = Path(input_path).suffix or '.mp3'

        for i, r in enumerate(keep_ranges):
            fd, seg_path = tempfile.mkstemp(suffix=ext)
            os.close(fd)
            temp_files.append(seg_path)
            segment_paths.append(seg_path)

            seg_duration = r.end - r.start
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(r.start),
                '-i', input_path,
                '-t', str(seg_duration),
                '-c:a', 'copy',
                seg_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Audio segment {i} extraction failed: {result.stderr[:200]}")

        concat_path = self._concat_segments(segment_paths, temp_files)
        return concat_path, segment_paths

    def _concat_segments(
        self,
        segment_paths: List[str],
        temp_files: List[str],
    ) -> str:
        """Concatenate segments using FFmpeg concat demuxer."""
        fd, list_path = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        temp_files.append(list_path)
        with open(list_path, 'w') as f:
            for sp in segment_paths:
                f.write(f"file '{sp}'\n")

        # Use .mkv for intermediate (supports PCM audio); .mp4 for audio-only
        suffix = '.mkv' if any(sp.endswith('.mkv') for sp in segment_paths) else '.mp4'
        fd, concat_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat', '-safe', '0',
            '-i', list_path,
            '-c', 'copy',
            '-fflags', '+genpts',
            '-avoid_negative_ts', 'make_zero',
            concat_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Concatenation failed: {result.stderr[:500]}")

        return concat_path

    def _probe_duration(self, path: str) -> float:
        """Get actual duration of a media file via ffprobe."""
        dur_result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'format=duration', '-of', 'csv=p=0', path],
            capture_output=True, text=True
        )
        try:
            return float(dur_result.stdout.strip())
        except (ValueError, AttributeError):
            return 0.0

    def _compute_boundaries(
        self,
        segment_paths: List[str],
        keep_ranges: List[TimeRange],
    ) -> List[float]:
        """Compute boundary timestamps from actual segment durations."""
        boundaries = []
        cumulative = 0.0
        for i, seg_path in enumerate(segment_paths[:-1]):
            dur = self._probe_duration(seg_path)
            if dur <= 0:
                dur = keep_ranges[i].end - keep_ranges[i].start
            cumulative += dur
            boundaries.append(cumulative)
        return boundaries

    def _apply_audio_continuity_to_video(
        self,
        video_path: str,
        segment_paths: List[str],
        continuity_method: ContinuityMethod,
        boundaries: List[float],
        temp_files: List[str],
    ) -> Optional[str]:
        """
        Extract audio from concatenated video, apply in-place boundary
        smoothing (micro-fade + method-specific blend), and remux with video.
        """
        from .audio_continuity import process_audio_at_boundaries

        try:
            if not boundaries:
                return None

            logger.info(
                f"Audio continuity: {len(boundaries)} boundaries at "
                f"{[f'{b:.3f}s' for b in boundaries]}"
            )

            # Extract audio from concatenated video
            fd, extracted_audio = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            temp_files.append(extracted_audio)

            extract_result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path,
                 '-vn', '-acodec', 'pcm_s16le', extracted_audio],
                capture_output=True, text=True, timeout=120
            )
            if extract_result.returncode != 0:
                logger.warning(
                    f"Audio extraction failed: {extract_result.stderr[:200]}"
                )
                return None

            # Apply in-place boundary smoothing
            processed_audio = process_audio_at_boundaries(
                extracted_audio, boundaries, continuity_method, device=self.device
            )
            temp_files.append(processed_audio)

            # Remux processed audio with original video
            fd, remuxed_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)

            remux_result = subprocess.run(
                ['ffmpeg', '-y',
                 '-i', video_path,
                 '-i', processed_audio,
                 '-map', '0:v:0',
                 '-map', '1:a:0',
                 '-c:v', 'copy',
                 '-c:a', 'pcm_s16le',
                 remuxed_path],
                capture_output=True, text=True, timeout=120
            )

            if remux_result.returncode != 0:
                logger.warning(
                    f"Audio remux failed: {remux_result.stderr[:200]}"
                )
                return None

            logger.info("Audio continuity applied and remuxed successfully")
            return remuxed_path

        except Exception as e:
            logger.error(
                f"Audio continuity for video failed: {e}", exc_info=True
            )
            return None


    def _apply_demucs_continuity_to_video(
        self,
        video_path: str,
        boundaries: List[float],
        temp_files: List[str],
    ) -> Optional[str]:
        """
        Run Demucs source separation on video audio, apply ROOM_TONE to vocals
        and CROSSFADE to background independently, then remux the remix with video.
        """
        from .demucs_separate import apply_demucs_continuity
        from .types import ContinuityMethod as CM

        try:
            if not boundaries:
                return None

            # Extract audio
            fd, extracted_audio = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            temp_files.append(extracted_audio)
            extract_result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', extracted_audio],
                capture_output=True, text=True, timeout=120
            )
            if extract_result.returncode != 0:
                return None

            # Demucs: separate, apply ROOM_TONE to vocals, CROSSFADE to background, remix
            processed_audio = apply_demucs_continuity(
                extracted_audio, boundaries, CM.ROOM_TONE, temp_files, device=self.device
            )
            if processed_audio is None:
                # Fallback to standard ROOM_TONE if Demucs fails
                from .audio_continuity import process_audio_at_boundaries
                processed_audio = process_audio_at_boundaries(
                    extracted_audio, boundaries, CM.ROOM_TONE, device=self.device
                )
            temp_files.append(processed_audio)

            # Remux
            fd, remuxed_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)
            remux_result = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-i', processed_audio,
                 '-map', '0:v:0', '-map', '1:a:0',
                 '-c:v', 'copy', '-c:a', 'pcm_s16le', remuxed_path],
                capture_output=True, text=True, timeout=120
            )
            if remux_result.returncode != 0:
                return None

            logger.info("Demucs continuity applied and remuxed successfully")
            return remuxed_path

        except Exception as e:
            logger.error(f"Demucs continuity for video failed: {e}", exc_info=True)
            return None

    def _apply_boundary_clone_to_video(
        self,
        video_path: str,
        boundaries: List[float],
        words: list,
        temp_files: List[str],
    ) -> Optional[str]:
        """Extract audio, apply BOUNDARY_CLONE synthesis, remux with video."""
        from .audio_continuity import apply_boundary_clone

        try:
            if not boundaries:
                return None

            fd, extracted_audio = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            temp_files.append(extracted_audio)
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', extracted_audio],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                return None

            processed_audio = apply_boundary_clone(
                extracted_audio, boundaries, words, device=self.device
            )
            temp_files.append(processed_audio)

            fd, remuxed_path = tempfile.mkstemp(suffix='.mkv')
            os.close(fd)
            r = subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-i', processed_audio,
                 '-map', '0:v:0', '-map', '1:a:0',
                 '-c:v', 'copy', '-c:a', 'pcm_s16le', remuxed_path],
                capture_output=True, text=True, timeout=120
            )
            if r.returncode != 0:
                return None

            logger.info("BOUNDARY_CLONE continuity applied and remuxed successfully")
            return remuxed_path

        except Exception as e:
            logger.error(f"BOUNDARY_CLONE for video failed: {e}", exc_info=True)
            return None

    # -----------------------------------------------------------------------
    # AI semantic editing
    # -----------------------------------------------------------------------

    async def semantic_edit(self, transcript: str, instruction: str, context) -> "SemanticEditResponse":
        from .types import SemanticEditResponse, SemanticOp
        return await sync_to_async(self._semantic_edit_sync)(transcript, instruction, context)

    def _semantic_edit_sync(self, transcript: str, instruction: str, context) -> "SemanticEditResponse":
        from .types import SemanticEditResponse, SemanticOp
        from .ai_semantic import semantic_edit as _semantic_edit
        import json

        speaker_names: Optional[Dict[str, str]] = None
        if context and context.speaker_names:
            try:
                speaker_names = json.loads(context.speaker_names)
            except Exception:
                pass

        result = _semantic_edit(transcript, instruction, speaker_names, self.device)

        if result.get("error_message"):
            return SemanticEditResponse(error_message=result["error_message"])

        raw_ops = result.get("operations") or []
        ops = []
        for op in raw_ops:
            ops.append(SemanticOp(
                type=op.get("type", ""),
                word_ids=op.get("word_ids"),
                target_word_id=op.get("target_word_id"),
                start=op.get("start"),
                end=op.get("end"),
            ))

        return SemanticEditResponse(
            operations=ops or None,
            explanation=result.get("explanation"),
        )

    async def ai_chat(self, transcript: str, message: str, history, context) -> "AIChatResponse":
        from .types import AIChatResponse
        return await sync_to_async(self._ai_chat_sync)(transcript, message, history, context)

    def _ai_chat_sync(self, transcript: str, message: str, history, context) -> "AIChatResponse":
        from .types import AIChatResponse, SemanticOp
        from .ai_semantic import ai_chat as _ai_chat
        import json

        speaker_names: Optional[Dict[str, str]] = None
        if context and context.speaker_names:
            try:
                speaker_names = json.loads(context.speaker_names)
            except Exception:
                pass

        hist_list = None
        if history:
            hist_list = [{"role": h.role, "content": h.content} for h in history]

        result = _ai_chat(transcript, message, hist_list, speaker_names, self.device)

        if result.get("error_message"):
            return AIChatResponse(error_message=result["error_message"])

        raw_ops = result.get("operations") or []
        ops = []
        for op in raw_ops:
            ops.append(SemanticOp(
                type=op.get("type", ""),
                word_ids=op.get("word_ids"),
                target_word_id=op.get("target_word_id"),
                start=op.get("start"),
                end=op.get("end"),
            ))

        return AIChatResponse(
            reply=result.get("reply") or "",
            operations=ops or None,
        )


# Global service instance
transcription_service = TranscriptionService()