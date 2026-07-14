# VocalBlade Backend API Reference

## Overview

The backend uses **Django + Strawberry GraphQL**. All API calls go through a single endpoint:

```
POST /graphql/
```

## Queries

### `availableModels`
Returns available Whisper model sizes.

```graphql
{ availableModels }
```

Response: `["TINY", "BASE", "SMALL", "MEDIUM", "LARGE", "LARGE_V2", "LARGE_V3"]`

### `supportedLanguages`
Returns ISO language codes supported for transcription.

```graphql
{ supportedLanguages }
```

Response: `["en", "zh", "de", "es", ...]` (98 languages)

## Mutations

### `transcribeAudio`
Transcribe a single audio file with WhisperX.

```graphql
mutation($audio: Upload!) {
  transcribeAudio(audio: $audio, params: {
    model: BASE
    language: null          # null = auto-detect
    enableDiarization: true
    enableAlignment: true
    batchSize: 2
    returnWordTimings: false
  }) {
    status              # SUCCESS | PROCESSING | FAILED
    text                # Full transcribed text
    segments {
      id
      text
      start             # Start time in seconds
      end               # End time in seconds
      score             # Confidence score (0-1)
      speaker           # Speaker label (e.g., "SPEAKER_00")
      words {
        word
        start
        end
        score
      }
    }
    language            # Detected language code
    languageScore       # Language detection confidence
    duration            # Audio duration in seconds
    processingTime      # Time taken to process
    errorMessage        # Error description (if FAILED)
    errorCode           # Error code (if FAILED)
    errorDetails        # JSON string with details
  }
}
```

**File upload** uses the [GraphQL multipart request spec](https://github.com/jaydenseric/graphql-multipart-request-spec):

```bash
curl -X POST http://localhost:8000/graphql/ \
  -F operations='{"query":"mutation($audio: Upload!) { transcribeAudio(audio: $audio) { status text segments { id text start end speaker words { word start end } } language duration processingTime } }","variables":{"audio":null}}' \
  -F map='{"0":["variables.audio"]}' \
  -F 0=@test.wav
```

### `batchTranscribeAudio`
Transcribe multiple audio files. Same params, returns `[TranscriptionResponse]`.

## File Constraints
- Max file size: 100MB
- Allowed formats: `.wav`, `.mp3`, `.m4a`, `.mp4`, `.mpga`, `.mpeg`, `.webm`

## Authentication (JWT)
The backend includes JWT auth via `strawberry-django-auth`:
- `tokenAuth(username, password)` — login
- `register(email, username, password1, password2)` — register
- `refreshToken(refreshToken)` — refresh JWT
- `revokeToken(refreshToken)` — logout

Auth is optional for transcription in development mode.
