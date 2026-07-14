"""
GraphQL schema definitions for the transcriber module.
Follows Strawberry GraphQL best practices with clear separation of concerns.
"""
from typing import Optional, List
import strawberry
from strawberry.file_uploads import Upload
from strawberry.types import Info
from django.core.files.uploadedfile import UploadedFile

from .types import (
    TranscriptionResponse,
    TranscriptionInput,
    SimpleTranscriptionInput,
    TranscriptionStatus,
    ModelSize,
    TimeRange,
    ExportResponse,
    ContinuityMethod,
    VideoInterpolationMethod,
    SemanticEditResponse,
    AIChatResponse,
    ChatMessage,
    SemanticEditContext,
)
from .resolvers import transcription_service


@strawberry.type
class TranscriberQuery:
    """Query type for transcriber module."""

    @strawberry.field
    def available_models(self) -> list[ModelSize]:
        """Get list of available Whisper models."""
        return list(ModelSize)

    @strawberry.field
    def supported_languages(self) -> list[str]:
        """Get list of supported languages for transcription."""
        # This is a subset of languages Whisper supports well
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja",
            "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
            "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
            "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
            "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
            "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
            "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
            "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
            "km", "sn", "yo", "so", "af", "oc", "ka", "be",
            "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
            "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
            "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
            "ba", "jw", "su"
        ]


@strawberry.type
class TranscriberMutation:
    """Mutation type for transcriber module."""

    @strawberry.mutation
    async def transcribe_audio(
        self,
        info: Info,
        audio: Upload, # type: ignore
        params: Optional[TranscriptionInput] = None
    ) -> TranscriptionResponse:
        """
        Transcribe audio with full features including alignment and diarization.

        Args:
            audio: Audio file to transcribe (supports wav, mp3, m4a, etc.)
            params: Optional transcription parameters

        Returns:
            TranscriptionResponse: Unified response with status and optional error info
        """
        # Get default params if not provided
        if params is None:
            params = TranscriptionInput()

        # Validate file
        audio_file: UploadedFile = audio

        # Check file size (max 100MB)
        if audio_file.size > 100 * 1024 * 1024:
            return TranscriptionResponse(
                status=TranscriptionStatus.FAILED,
                error_message="File size exceeds 100MB limit",
                error_code="FILE_TOO_LARGE"
            )

        # Check file extension
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.mp4', '.mpga', '.mpeg', '.webm'}
        file_name = audio_file.name or 'audio'
        extension = '.' + file_name.split('.')[-1].lower() if '.' in file_name else ''

        if extension and extension not in allowed_extensions:
            import json
            return TranscriptionResponse(
                status=TranscriptionStatus.FAILED,
                error_message=f"Unsupported file format: {extension}",
                error_code="UNSUPPORTED_FORMAT",
                error_details=json.dumps({"allowed_formats": list(allowed_extensions)})
            )

        # Process transcription
        result = await transcription_service.transcribe_audio_full(audio_file, params)
        return result

    @strawberry.mutation
    async def batch_transcribe_audio(
        self,
        info: Info,
        audio_files: list[Upload], # type: ignore
        params: Optional[TranscriptionInput] = None
    ) -> list[TranscriptionResponse]:
        """
        Batch transcribe multiple audio files.

        Args:
            audio_files: List of audio files to transcribe
            params: Optional transcription parameters (applied to all files)

        Returns:
            List of TranscriptionResponse results
        """
        if params is None:
            params = TranscriptionInput()

        results = []

        for audio in audio_files:
            audio_file: UploadedFile = audio

            # Basic validation
            if audio_file.size > 100 * 1024 * 1024:
                results.append(TranscriptionResponse(
                    status=TranscriptionStatus.FAILED,
                    error_message=f"File {audio_file.name} exceeds 100MB limit",
                    error_code="FILE_TOO_LARGE"
                ))
                continue

            # Process each file
            result = await transcription_service.transcribe_audio_full(audio_file, params)
            results.append(result)

        return results

    @strawberry.mutation
    async def export_edited_media(
        self,
        info: Info,
        media: Upload,  # type: ignore
        keep_ranges: list[TimeRange],
        continuity_method: Optional[ContinuityMethod] = ContinuityMethod.SMART_SILENCE,
        video_interpolation_method: Optional[VideoInterpolationMethod] = VideoInterpolationMethod.NONE,
        segments_json: Optional[str] = None,
    ) -> ExportResponse:
        """
        Export edited media by keeping only the specified time ranges.

        Uses hard cuts with smart audio boundary smoothing and optional
        video frame interpolation at cut boundaries.

        Args:
            media: Original media file
            keep_ranges: List of time ranges to keep in the output
            continuity_method: Audio continuity method for smoothing edit boundaries
            video_interpolation_method: Video frame interpolation at cut boundaries

        Returns:
            ExportResponse with download URL or error
        """
        media_file: UploadedFile = media

        # File size validation
        if media_file.size > 500 * 1024 * 1024:
            return ExportResponse(
                status=TranscriptionStatus.FAILED,
                error_message="File size exceeds 500MB limit for export"
            )

        # Empty keep_ranges = no edits → export full file as-is

        return await transcription_service.export_edited_media(
            media_file, keep_ranges,
            continuity_method or ContinuityMethod.SMART_SILENCE,
            video_interpolation_method or VideoInterpolationMethod.NONE,
            segments_json=segments_json,
        )

    @strawberry.mutation
    async def semantic_edit(
        self,
        info: Info,
        transcript: str,
        instruction: str,
        context: Optional[SemanticEditContext] = None,
    ) -> SemanticEditResponse:
        """
        Use Qwen 2.5 7B to analyse the transcript and produce edit operations
        (delete words, move words, delete time ranges) for the given instruction.

        transcript: JSON string of TranscriptionResponse segments (or segment list)
        instruction: Natural-language editing instruction
        context: Optional speaker name map (JSON string)
        """
        return await transcription_service.semantic_edit(transcript, instruction, context)

    @strawberry.mutation
    async def ai_chat(
        self,
        info: Info,
        transcript: str,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        context: Optional[SemanticEditContext] = None,
    ) -> AIChatResponse:
        """
        Conversational AI assistant that can answer questions about the transcript
        and/or propose edit operations.

        transcript: JSON string of TranscriptionResponse segments
        message: New user message
        history: Prior conversation turns
        context: Optional speaker name map
        """
        return await transcription_service.ai_chat(transcript, message, history, context)