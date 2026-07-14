"""
GraphQL type definitions for the transcriber module.
Following Strawberry GraphQL best practices with dataclasses and proper typing.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import strawberry


@strawberry.enum
class ModelSize(Enum):
    """Available Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"


@strawberry.enum
class TranscriptionStatus(Enum):
    """Status of transcription operation."""
    SUCCESS = "success"
    PROCESSING = "processing"
    FAILED = "failed"


@strawberry.enum
class ContinuityMethod(Enum):
    """Audio continuity method for smoothing edit boundaries during export."""
    CROSSFADE = "crossfade"           # Method 1: Fast time-domain equal-power crossfade
    SMART_SILENCE = "smart_silence"   # Method 2: Silence-aware cuts (recommended default)
    SPECTRAL = "spectral"             # Method 3: FFT-based phase-coherent crossfade
    AI_INPAINTING = "ai_inpainting"   # Method 4: Neural network audio inpainting
    WSOLA = "wsola"                   # Method 5: WSOLA pitch-aware speech blending
    ROOM_TONE = "room_tone"           # Method 6: Room tone replacement (recommended)
    AI_REGENERATION = "ai_regeneration"  # Method 7: AI voice regeneration (F5-TTS)
    ENCODEC = "encodec"                 # Method 8: Meta EnCodec neural codec latent blending
    NEURAL_SMOOTH = "neural_smooth"    # Method 9: Combined room tone + face morph (Smart Edit)
    SMART_CUT = "smart_cut"           # Method 10: FOMM video bridge + speech crossfade audio bridge
    SADTALKER_CUT = "sadtalker_cut"   # Method 11: SadTalker audio-driven face + F5-TTS voice clone bridge
    DEMUCS = "demucs"                 # Method 12: Demucs source separation + per-stem continuity
    BOUNDARY_CLONE = "boundary_clone" # Method 13: F5-TTS clone boundary words; crossfade within word
    NONE = "none"                     # No processing: raw concatenated segments


@strawberry.enum
class VideoInterpolationMethod(Enum):
    """Video frame interpolation method at edit cut boundaries."""
    NONE      = "none"       # Hard cut — default, no processing
    BLEND     = "blend"      # ffmpeg minterpolate (motion-compensated, no model needed)
    RIFE      = "rife"       # RIFE v4.25 neural optical flow interpolation (local weights)
    FRAME_PAD = "frame_pad"  # Freeze boundary frames + dissolve transition (no new frames generated)


@strawberry.type
class WordTiming:
    """Word-level timing information."""
    word: str
    start: float
    end: float
    score: Optional[float] = None


@strawberry.type
class TranscriptionSegment:
    """Represents a segment of transcribed audio."""
    id: int
    text: str
    start: float
    end: float
    score: Optional[float] = None
    speaker: Optional[str] = None
    words: Optional[List[WordTiming]] = None


@strawberry.type
class TranscriptionResponse:
    """Unified response for transcription operations."""
    status: TranscriptionStatus
    text: Optional[str] = None
    segments: Optional[List[TranscriptionSegment]] = None
    language: Optional[str] = None
    language_score: Optional[float] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Optional[str] = None  # JSON string for frontend compatibility


@strawberry.input
class TranscriptionInput:
    """Input parameters for transcription."""
    model: Optional[ModelSize] = ModelSize.BASE
    language: Optional[str] = None
    enable_diarization: Optional[bool] = True
    enable_alignment: Optional[bool] = True
    batch_size: Optional[int] = 8
    return_word_timings: Optional[bool] = False


@strawberry.input
class SimpleTranscriptionInput:
    """Input parameters for simple transcription."""
    model: Optional[ModelSize] = ModelSize.BASE
    language: Optional[str] = None
    batch_size: Optional[int] = 8


@strawberry.input
class TimeRange:
    """A time range in seconds."""
    start: float
    end: float


@strawberry.type
class ExportResponse:
    """Response for media export operations."""
    status: TranscriptionStatus
    file_url: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# AI Semantic Editing types
# ---------------------------------------------------------------------------

@strawberry.type
class SemanticOp:
    """A single edit operation proposed by the AI."""
    type: str                          # "delete_words" | "move_words" | "delete_range"
    word_ids: Optional[List[str]] = None
    target_word_id: Optional[str] = None
    start: Optional[float] = None      # for delete_range
    end: Optional[float] = None        # for delete_range


@strawberry.type
class SemanticEditResponse:
    """Response from the semanticEdit mutation."""
    operations: Optional[List[SemanticOp]] = None
    explanation: Optional[str] = None
    error_message: Optional[str] = None


@strawberry.input
class ChatMessage:
    """A single message in an AI chat history."""
    role: str     # "user" | "assistant"
    content: str


@strawberry.input
class SemanticEditContext:
    """Optional context passed alongside transcript for AI mutations."""
    speaker_names: Optional[str] = None   # JSON string: {"SPEAKER_0": "John"}


@strawberry.type
class AIChatResponse:
    """Response from the aiChat mutation."""
    reply: str = ""
    operations: Optional[List[SemanticOp]] = None
    error_message: Optional[str] = None