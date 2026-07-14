import { useReducer, useMemo } from 'react';
import type { TranscriptionParams, TranscriptionResponse, ContinuityMethod, VideoInterpolationMethod } from '../api/graphql';
import {
  type EditOperation,
  type WordId,
  type EffectiveWord,
  buildFlatWordList,
  applyEdits,
  getTextDirection,
  getWordIdsInRange,
} from '../utils/transcriptOps';

// --- State ---

export interface EditorState {
  // File
  file: File | null;
  fileType: 'audio' | 'video' | null;

  // Transcription
  transcriptionParams: TranscriptionParams;
  transcriptionStatus: 'idle' | 'transcribing' | 'done' | 'error';
  errorMessage: string;
  result: TranscriptionResponse | null;

  // Derived from result
  textDirection: 'ltr' | 'rtl';

  // Playback
  currentTime: number;
  isPlaying: boolean;

  // Selection
  selectedWordIds: Set<WordId>;
  selectionAnchor: WordId | null;

  // Edits
  editOperations: EditOperation[];
  undoStack: EditOperation[][];
  redoStack: EditOperation[][];

  // Clipboard (internal word IDs for paste)
  copiedWordIds: WordId[];
  pasteCounter: number;

  // Speaker names (original ID → custom name)
  speakerNames: Record<string, string>;

  // Audio continuity
  continuityMethod: ContinuityMethod;

  // Video interpolation
  videoInterpolationMethod: VideoInterpolationMethod;

  // Smart Edit (combined room-tone + face morph)
  smartEditEnabled: boolean;

  // Export
  exportStatus: 'idle' | 'exporting' | 'done' | 'error';
  exportUrl: string | null;
  exportError: string;

  // Smart Cut export
  smartCutStatus: 'idle' | 'exporting' | 'done' | 'error';
  smartCutUrl: string | null;
  smartCutError: string;

  // AI Panel
  aiPanelOpen: boolean;
  aiChatHistory: { role: 'user' | 'assistant'; content: string; ops?: EditOperation[] }[];
  aiStatus: 'idle' | 'thinking' | 'error';
  aiError: string;
  aiPendingOps: EditOperation[];
}

function detectFileType(file: File): 'audio' | 'video' {
  if (file.type.startsWith('video/')) return 'video';
  const ext = file.name.split('.').pop()?.toLowerCase();
  if (ext === 'mp4' || ext === 'webm' || ext === 'mpeg') return 'video';
  return 'audio';
}

const initialState: EditorState = {
  file: null,
  fileType: null,
  transcriptionParams: { model: 'BASE', enableDiarization: true, returnWordTimings: true },
  transcriptionStatus: 'idle',
  errorMessage: '',
  result: null,
  textDirection: 'ltr',
  currentTime: 0,
  isPlaying: false,
  selectedWordIds: new Set(),
  selectionAnchor: null,
  copiedWordIds: [],
  pasteCounter: 0,
  editOperations: [],
  undoStack: [],
  redoStack: [],
  speakerNames: {},
  continuityMethod: 'ROOM_TONE' as ContinuityMethod,
  videoInterpolationMethod: 'NONE' as VideoInterpolationMethod,
  smartEditEnabled: false,
  exportStatus: 'idle',
  exportUrl: null,
  exportError: '',
  smartCutStatus: 'idle',
  smartCutUrl: null,
  smartCutError: '',
  aiPanelOpen: false,
  aiChatHistory: [],
  aiStatus: 'idle',
  aiError: '',
  aiPendingOps: [],
};

// --- Actions ---

type EditorAction =
  | { type: 'SET_FILE'; file: File }
  | { type: 'SET_PARAMS'; params: TranscriptionParams }
  | { type: 'SET_STATUS'; status: EditorState['transcriptionStatus']; error?: string }
  | { type: 'SET_RESULT'; result: TranscriptionResponse }
  | { type: 'SET_CURRENT_TIME'; time: number }
  | { type: 'SET_PLAYING'; playing: boolean }
  | { type: 'SELECT_WORD'; wordId: WordId }
  | { type: 'SHIFT_SELECT_WORD'; wordId: WordId; flatWords: EffectiveWord[] }
  | { type: 'TOGGLE_SELECT_WORD'; wordId: WordId }
  | { type: 'SELECT_ALL'; flatWords: EffectiveWord[] }
  | { type: 'CLEAR_SELECTION' }
  | { type: 'DELETE_SELECTED' }
  | { type: 'MOVE_WORDS'; targetWordId: WordId }
  | { type: 'COPY_WORDS' }
  | { type: 'PASTE_WORDS' }
  | { type: 'RENAME_SPEAKER'; originalId: string; newName: string }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'SET_CONTINUITY_METHOD'; method: ContinuityMethod }
  | { type: 'SET_VIDEO_INTERPOLATION_METHOD'; method: VideoInterpolationMethod }
  | { type: 'SET_SMART_EDIT'; enabled: boolean }
  | { type: 'SET_EXPORT_STATUS'; status: EditorState['exportStatus']; url?: string; error?: string }
  | { type: 'SET_SMART_CUT_STATUS'; status: EditorState['smartCutStatus']; url?: string; error?: string }
  | { type: 'SET_AI_PANEL_OPEN'; open: boolean }
  | { type: 'AI_MESSAGE'; role: 'user' | 'assistant'; content: string; ops?: EditOperation[] }
  | { type: 'SET_AI_STATUS'; status: EditorState['aiStatus']; error?: string }
  | { type: 'SET_AI_PENDING'; ops: EditOperation[] }
  | { type: 'AI_APPLY_OPS' }
  | { type: 'AI_DISCARD_PENDING' }
  | { type: 'RESET' };

function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case 'SET_FILE':
      return {
        ...initialState,
        file: action.file,
        fileType: detectFileType(action.file),
        transcriptionParams: state.transcriptionParams,
      };

    case 'SET_PARAMS':
      return { ...state, transcriptionParams: action.params };

    case 'SET_STATUS':
      return {
        ...state,
        transcriptionStatus: action.status,
        errorMessage: action.error || '',
        ...(action.status === 'transcribing' ? { result: null } : {}),
      };

    case 'SET_RESULT':
      return {
        ...state,
        result: action.result,
        transcriptionStatus: action.result.status === 'FAILED' ? 'error' : 'done',
        errorMessage: action.result.errorMessage || '',
        textDirection: getTextDirection(action.result.language),
        // Reset editing state for new result
        selectedWordIds: new Set(),
        selectionAnchor: null,
        copiedWordIds: [],
        pasteCounter: 0,
        editOperations: [],
        undoStack: [],
        redoStack: [],
        exportStatus: 'idle', exportUrl: null, exportError: '',
        smartCutStatus: 'idle', smartCutUrl: null, smartCutError: '',
      };

    case 'SET_CURRENT_TIME':
      return { ...state, currentTime: action.time };

    case 'SET_PLAYING':
      return { ...state, isPlaying: action.playing };

    case 'SELECT_WORD':
      return {
        ...state,
        selectedWordIds: new Set([action.wordId]),
        selectionAnchor: action.wordId,
      };

    case 'SHIFT_SELECT_WORD': {
      if (!state.selectionAnchor) {
        return {
          ...state,
          selectedWordIds: new Set([action.wordId]),
          selectionAnchor: action.wordId,
        };
      }
      const rangeIds = getWordIdsInRange(action.flatWords, state.selectionAnchor, action.wordId);
      return {
        ...state,
        selectedWordIds: new Set(rangeIds),
      };
    }

    case 'TOGGLE_SELECT_WORD': {
      const next = new Set(state.selectedWordIds);
      if (next.has(action.wordId)) {
        next.delete(action.wordId);
      } else {
        next.add(action.wordId);
      }
      return {
        ...state,
        selectedWordIds: next,
        selectionAnchor: action.wordId,
      };
    }

    case 'SELECT_ALL': {
      const allIds = action.flatWords
        .filter((w) => !w.deleted)
        .map((w) => w.wordId);
      return {
        ...state,
        selectedWordIds: new Set(allIds),
      };
    }

    case 'CLEAR_SELECTION':
      return { ...state, selectedWordIds: new Set(), selectionAnchor: null };

    case 'DELETE_SELECTED': {
      if (state.selectedWordIds.size === 0) return state;
      const deleteOp: EditOperation = {
        type: 'delete',
        wordIds: Array.from(state.selectedWordIds),
      };
      return {
        ...state,
        undoStack: [...state.undoStack, state.editOperations],
        redoStack: [],
        editOperations: [...state.editOperations, deleteOp],
        selectedWordIds: new Set(),
        selectionAnchor: null,
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'MOVE_WORDS': {
      if (state.selectedWordIds.size === 0) return state;
      const moveOp: EditOperation = {
        type: 'move',
        wordIds: Array.from(state.selectedWordIds),
        targetWordId: action.targetWordId,
      };
      return {
        ...state,
        undoStack: [...state.undoStack, state.editOperations],
        redoStack: [],
        editOperations: [...state.editOperations, moveOp],
        selectedWordIds: new Set(),
        selectionAnchor: null,
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'COPY_WORDS': {
      if (state.selectedWordIds.size === 0) return state;
      return {
        ...state,
        copiedWordIds: Array.from(state.selectedWordIds),
      };
    }

    case 'PASTE_WORDS': {
      if (state.copiedWordIds.length === 0) return state;
      // Insert after the last selected word, or at end if nothing selected
      const anchor = state.selectionAnchor ?? null;
      const pasteOp: EditOperation = {
        type: 'paste',
        wordIds: state.copiedWordIds,
        insertAfter: anchor,
        pasteId: state.pasteCounter,
      };
      return {
        ...state,
        undoStack: [...state.undoStack, state.editOperations],
        redoStack: [],
        editOperations: [...state.editOperations, pasteOp],
        pasteCounter: state.pasteCounter + 1,
        selectedWordIds: new Set(),
        selectionAnchor: null,
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'RENAME_SPEAKER': {
      return {
        ...state,
        speakerNames: {
          ...state.speakerNames,
          [action.originalId]: action.newName,
        },
      };
    }

    case 'UNDO': {
      if (state.undoStack.length === 0) return state;
      const prevOps = state.undoStack[state.undoStack.length - 1];
      return {
        ...state,
        undoStack: state.undoStack.slice(0, -1),
        redoStack: [...state.redoStack, state.editOperations],
        editOperations: prevOps,
        selectedWordIds: new Set(),
        selectionAnchor: null,
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'REDO': {
      if (state.redoStack.length === 0) return state;
      const nextOps = state.redoStack[state.redoStack.length - 1];
      return {
        ...state,
        redoStack: state.redoStack.slice(0, -1),
        undoStack: [...state.undoStack, state.editOperations],
        editOperations: nextOps,
        selectedWordIds: new Set(),
        selectionAnchor: null,
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'SET_CONTINUITY_METHOD':
      return { ...state, continuityMethod: action.method, exportStatus: 'idle', exportUrl: null, exportError: '' };

    case 'SET_VIDEO_INTERPOLATION_METHOD':
      return { ...state, videoInterpolationMethod: action.method, exportStatus: 'idle', exportUrl: null, exportError: '' };

    case 'SET_SMART_EDIT':
      return { ...state, smartEditEnabled: action.enabled, exportStatus: 'idle', exportUrl: null, exportError: '' };

    case 'SET_EXPORT_STATUS':
      return {
        ...state,
        exportStatus: action.status,
        exportUrl: action.url || null,
        exportError: action.error || '',
      };

    case 'SET_SMART_CUT_STATUS':
      return {
        ...state,
        smartCutStatus: action.status,
        smartCutUrl: action.url || null,
        smartCutError: action.error || '',
      };

    case 'SET_AI_PANEL_OPEN':
      return { ...state, aiPanelOpen: action.open };

    case 'AI_MESSAGE':
      return {
        ...state,
        aiChatHistory: [
          ...state.aiChatHistory,
          { role: action.role, content: action.content, ops: action.ops },
        ],
      };

    case 'SET_AI_STATUS':
      return { ...state, aiStatus: action.status, aiError: action.error || '' };

    case 'SET_AI_PENDING':
      return { ...state, aiPendingOps: action.ops };

    case 'AI_APPLY_OPS': {
      if (state.aiPendingOps.length === 0) return state;
      return {
        ...state,
        undoStack: [...state.undoStack, state.editOperations],
        redoStack: [],
        editOperations: [...state.editOperations, ...state.aiPendingOps],
        aiPendingOps: [],
        exportStatus: 'idle', exportUrl: null, exportError: '',
      };
    }

    case 'AI_DISCARD_PENDING':
      return { ...state, aiPendingOps: [] };

    case 'RESET':
      return initialState;

    default:
      return state;
  }
}

// --- Hook ---

export function useEditorState() {
  const [state, dispatch] = useReducer(editorReducer, initialState);

  const flatWords = useMemo(() => {
    if (!state.result?.segments) return [];
    return applyEdits(buildFlatWordList(state.result.segments), state.editOperations);
  }, [state.result?.segments, state.editOperations]);

  const hasEdits = state.editOperations.length > 0;
  const canUndo = state.undoStack.length > 0;
  const canRedo = state.redoStack.length > 0;
  const hasSelection = state.selectedWordIds.size > 0;
  const isEditorMode = state.transcriptionStatus === 'done' && state.result?.segments != null;

  const actions = useMemo(() => ({
    setFile: (file: File) => dispatch({ type: 'SET_FILE', file }),
    setParams: (params: TranscriptionParams) => dispatch({ type: 'SET_PARAMS', params }),
    setStatus: (status: EditorState['transcriptionStatus'], error?: string) =>
      dispatch({ type: 'SET_STATUS', status, error }),
    setResult: (result: TranscriptionResponse) => dispatch({ type: 'SET_RESULT', result }),
    setCurrentTime: (time: number) => dispatch({ type: 'SET_CURRENT_TIME', time }),
    setPlaying: (playing: boolean) => dispatch({ type: 'SET_PLAYING', playing }),
    selectWord: (wordId: WordId) => dispatch({ type: 'SELECT_WORD', wordId }),
    shiftSelectWord: (wordId: WordId) =>
      dispatch({ type: 'SHIFT_SELECT_WORD', wordId, flatWords }),
    toggleSelectWord: (wordId: WordId) => dispatch({ type: 'TOGGLE_SELECT_WORD', wordId }),
    selectAll: () => dispatch({ type: 'SELECT_ALL', flatWords }),
    clearSelection: () => dispatch({ type: 'CLEAR_SELECTION' }),
    deleteSelected: () => dispatch({ type: 'DELETE_SELECTED' }),
    copyWords: () => dispatch({ type: 'COPY_WORDS' }),
    pasteWords: () => dispatch({ type: 'PASTE_WORDS' }),
    moveWords: (targetWordId: WordId) => dispatch({ type: 'MOVE_WORDS', targetWordId }),
    renameSpeaker: (originalId: string, newName: string) =>
      dispatch({ type: 'RENAME_SPEAKER', originalId, newName }),
    undo: () => dispatch({ type: 'UNDO' }),
    redo: () => dispatch({ type: 'REDO' }),
    setContinuityMethod: (method: ContinuityMethod) =>
      dispatch({ type: 'SET_CONTINUITY_METHOD', method }),
    setVideoInterpolationMethod: (method: VideoInterpolationMethod) =>
      dispatch({ type: 'SET_VIDEO_INTERPOLATION_METHOD', method }),
    setSmartEdit: (enabled: boolean) =>
      dispatch({ type: 'SET_SMART_EDIT', enabled }),
    setExportStatus: (status: EditorState['exportStatus'], url?: string, error?: string) =>
      dispatch({ type: 'SET_EXPORT_STATUS', status, url, error }),
    setSmartCutStatus: (status: EditorState['smartCutStatus'], url?: string, error?: string) =>
      dispatch({ type: 'SET_SMART_CUT_STATUS', status, url, error }),
    setAiPanelOpen: (open: boolean) => dispatch({ type: 'SET_AI_PANEL_OPEN', open }),
    addAiMessage: (role: 'user' | 'assistant', content: string, ops?: EditOperation[]) =>
      dispatch({ type: 'AI_MESSAGE', role, content, ops }),
    setAiStatus: (status: EditorState['aiStatus'], error?: string) =>
      dispatch({ type: 'SET_AI_STATUS', status, error }),
    setAiPending: (ops: EditOperation[]) => dispatch({ type: 'SET_AI_PENDING', ops }),
    aiApplyOps: () => dispatch({ type: 'AI_APPLY_OPS' }),
    aiDiscardPending: () => dispatch({ type: 'AI_DISCARD_PENDING' }),
    reset: () => dispatch({ type: 'RESET' }),
  }), [flatWords]);

  return {
    state,
    actions,
    flatWords,
    hasEdits,
    canUndo,
    canRedo,
    hasSelection,
    isEditorMode,
  };
}
