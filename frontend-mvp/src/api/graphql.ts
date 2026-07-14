const GRAPHQL_URL = '/graphql/';

export interface WordTiming {
  word: string;
  start: number;
  end: number;
  score: number | null;
}

export interface TranscriptionSegment {
  id: number;
  text: string;
  start: number;
  end: number;
  score: number | null;
  speaker: string | null;
  words: WordTiming[] | null;
}

export interface TranscriptionResponse {
  status: 'SUCCESS' | 'PROCESSING' | 'FAILED';
  text: string | null;
  segments: TranscriptionSegment[] | null;
  language: string | null;
  languageScore: number | null;
  duration: number | null;
  processingTime: number | null;
  errorMessage: string | null;
  errorCode: string | null;
  errorDetails: string | null;
}

export interface TranscriptionParams {
  model?: string;
  language?: string | null;
  enableDiarization?: boolean;
  enableAlignment?: boolean;
  batchSize?: number;
  returnWordTimings?: boolean;
}

export async function fetchAvailableModels(): Promise<string[]> {
  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: '{ availableModels }' }),
  });
  const json = await res.json();
  return json.data.availableModels;
}

export async function fetchSupportedLanguages(): Promise<string[]> {
  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: '{ supportedLanguages }' }),
  });
  const json = await res.json();
  return json.data.supportedLanguages;
}

export async function transcribeAudio(
  file: File,
  params?: TranscriptionParams
): Promise<TranscriptionResponse> {
  // GraphQL multipart upload spec (https://github.com/jaydenseric/graphql-multipart-request-spec)
  const paramsStr = params
    ? `, params: { ${Object.entries(params)
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => {
          if (k === 'model') return `${k}: ${v}`; // GraphQL enum — no quotes
          return `${k}: ${typeof v === 'string' ? `"${v}"` : v}`;
        })
        .join(', ')} }`
    : '';

  const operations = JSON.stringify({
    query: `mutation($audio: Upload!) {
      transcribeAudio(audio: $audio${paramsStr}) {
        status
        text
        segments {
          id
          text
          start
          end
          score
          speaker
          words {
            word
            start
            end
            score
          }
        }
        language
        languageScore
        duration
        processingTime
        errorMessage
        errorCode
        errorDetails
      }
    }`,
    variables: { audio: null },
  });

  const map = JSON.stringify({ '0': ['variables.audio'] });

  const formData = new FormData();
  formData.append('operations', operations);
  formData.append('map', map);
  formData.append('0', file);

  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();

  if (json.errors) {
    throw new Error(json.errors[0].message);
  }

  return json.data.transcribeAudio;
}

// --- Export edited media ---

export type ContinuityMethod = 'CROSSFADE' | 'SPECTRAL' | 'WSOLA' | 'ROOM_TONE' | 'SMART_SILENCE' | 'AI_INPAINTING' | 'AI_REGENERATION' | 'ENCODEC' | 'NEURAL_SMOOTH' | 'SMART_CUT' | 'SADTALKER_CUT' | 'DEMUCS' | 'BOUNDARY_CLONE' | 'J_CUT' | 'L_CUT' | 'NONE';
export type VideoInterpolationMethod = 'NONE' | 'BLEND' | 'RIFE' | 'FRAME_PAD' | 'FADE_TO_BLACK';

export interface ExportResponse {
  status: 'SUCCESS' | 'FAILED';
  fileUrl: string | null;
  errorMessage: string | null;
}

export async function exportEditedMedia(
  file: File,
  keepRanges: { start: number; end: number }[],
  continuityMethod: ContinuityMethod = 'CROSSFADE',
  videoInterpolationMethod: VideoInterpolationMethod = 'NONE',
  segmentsJson?: string,
): Promise<ExportResponse> {
  const keepRangesStr = keepRanges
    .map((r) => `{start: ${r.start}, end: ${r.end}}`)
    .join(', ');

  const segmentsArg = segmentsJson
    ? `, segmentsJson: ${JSON.stringify(segmentsJson)}`
    : '';

  const operations = JSON.stringify({
    query: `mutation($media: Upload!) {
      exportEditedMedia(media: $media, keepRanges: [${keepRangesStr}], continuityMethod: ${continuityMethod}, videoInterpolationMethod: ${videoInterpolationMethod}${segmentsArg}) {
        status
        fileUrl
        errorMessage
      }
    }`,
    variables: { media: null },
  });

  const map = JSON.stringify({ '0': ['variables.media'] });

  const formData = new FormData();
  formData.append('operations', operations);
  formData.append('map', map);
  formData.append('0', file);

  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    body: formData,
  });

  const json = await res.json();

  if (json.errors) {
    throw new Error(json.errors[0].message);
  }

  return json.data.exportEditedMedia;
}

// --- AI Semantic Editing ---

export type SemanticOp =
  | { type: 'delete_words'; wordIds: string[] }
  | { type: 'move_words'; wordIds: string[]; targetWordId: string }
  | { type: 'delete_range'; start: number; end: number };

export interface SemanticEditResponse {
  operations: SemanticOp[] | null;
  explanation: string | null;
  errorMessage: string | null;
}

export interface AIChatResponse {
  reply: string;
  operations: SemanticOp[] | null;
  errorMessage: string | null;
}

export async function semanticEdit(
  transcript: string,
  instruction: string,
  speakerNames?: Record<string, string>,
): Promise<SemanticEditResponse> {
  const contextArg = speakerNames
    ? `, context: { speakerNames: ${JSON.stringify(JSON.stringify(speakerNames))} }`
    : '';

  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `mutation SemanticEdit($transcript: String!, $instruction: String!) {
        semanticEdit(transcript: $transcript, instruction: $instruction${contextArg}) {
          operations { type wordIds targetWordId start end }
          explanation
          errorMessage
        }
      }`,
      variables: { transcript, instruction },
    }),
  });

  const json = await res.json();
  if (json.errors) throw new Error(json.errors[0].message);
  return json.data.semanticEdit;
}

export async function aiChat(
  transcript: string,
  message: string,
  history?: { role: string; content: string }[],
  speakerNames?: Record<string, string>,
): Promise<AIChatResponse> {
  const contextArg = speakerNames
    ? `, context: { speakerNames: ${JSON.stringify(JSON.stringify(speakerNames))} }`
    : '';

  const historyArg =
    history && history.length > 0
      ? `, history: [${history
          .map((h) => `{ role: ${JSON.stringify(h.role)}, content: ${JSON.stringify(h.content)} }`)
          .join(', ')}]`
      : '';

  const res = await fetch(GRAPHQL_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: `mutation AiChat($transcript: String!, $message: String!) {
        aiChat(transcript: $transcript, message: $message${historyArg}${contextArg}) {
          reply
          operations { type wordIds targetWordId start end }
          errorMessage
        }
      }`,
      variables: { transcript, message },
    }),
  });

  const json = await res.json();
  if (json.errors) throw new Error(json.errors[0].message);
  return json.data.aiChat;
}

export function semanticOpsToEditOps(
  ops: SemanticOp[],
): import('../utils/transcriptOps').EditOperation[] {
  return ops.map((op) => {
    if (op.type === 'delete_words') {
      return { type: 'delete' as const, wordIds: op.wordIds as import('../utils/transcriptOps').WordId[] };
    }
    if (op.type === 'move_words') {
      return {
        type: 'move' as const,
        wordIds: op.wordIds as import('../utils/transcriptOps').WordId[],
        targetWordId: op.targetWordId as import('../utils/transcriptOps').WordId,
      };
    }
    // delete_range
    return { type: 'delete_range' as const, start: op.start, end: op.end };
  });
}
