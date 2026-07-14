import { useEffect, useCallback, useRef } from 'react';
import type { EffectiveWord, WordId } from '../utils/transcriptOps';
import { getSelectedText } from '../utils/transcriptOps';

interface EditingToolbarProps {
  selectedCount: number;
  canUndo: boolean;
  canRedo: boolean;
  hasEdits: boolean;
  flatWords: EffectiveWord[];
  selectedWordIds: Set<WordId>;
  hasCopiedWords: boolean;
  onDelete: () => void;
  onCopy: () => void;
  onPaste: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
}

export default function EditingToolbar({
  selectedCount,
  canUndo,
  canRedo,
  hasEdits,
  flatWords,
  selectedWordIds,
  hasCopiedWords,
  onDelete,
  onCopy,
  onPaste,
  onUndo,
  onRedo,
  onSelectAll,
  onClearSelection,
}: EditingToolbarProps) {
  // Stable refs for copy handler — avoids re-registering the listener on every render
  const flatWordsRef = useRef(flatWords);
  const selectedWordIdsRef = useRef(selectedWordIds);
  const onCopyRef = useRef(onCopy);
  useEffect(() => { flatWordsRef.current = flatWords; }, [flatWords]);
  useEffect(() => { selectedWordIdsRef.current = selectedWordIds; }, [selectedWordIds]);
  useEffect(() => { onCopyRef.current = onCopy; }, [onCopy]);

  // Copy button handler (uses navigator.clipboard API for programmatic clicks)
  const handleCopy = useCallback(async () => {
    const text = getSelectedText(flatWordsRef.current, selectedWordIdsRef.current);
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    onCopyRef.current();
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isCtrl = e.ctrlKey || e.metaKey;

      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedCount > 0 && !isCtrl) {
        e.preventDefault();
        onDelete();
      } else if (isCtrl && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        onUndo();
      } else if (isCtrl && (e.key === 'Z' || (e.key === 'z' && e.shiftKey)) ) {
        e.preventDefault();
        onRedo();
      } else if (isCtrl && e.key === 'v') {
        e.preventDefault();
        onPaste();
      } else if (isCtrl && e.key === 'a') {
        e.preventDefault();
        onSelectAll();
      } else if (e.key === 'Escape') {
        onClearSelection();
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [selectedCount, onDelete, onPaste, onUndo, onRedo, onSelectAll, onClearSelection]);

  // Native copy event — register ONCE, read current values from refs
  // This avoids the race condition where the listener is unregistered/re-registered
  // between keypress and event dispatch (selectedWordIds is a new Set every render)
  useEffect(() => {
    const handleCopyEvent = (e: ClipboardEvent) => {
      if (selectedWordIdsRef.current.size === 0) return;
      const text = getSelectedText(flatWordsRef.current, selectedWordIdsRef.current);
      if (!text) return;
      e.preventDefault();
      e.clipboardData!.setData('text/plain', text);
      onCopyRef.current();
    };
    document.addEventListener('copy', handleCopyEvent);
    return () => document.removeEventListener('copy', handleCopyEvent);
  }, []); // Empty deps — register once, never re-register

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 border-b border-slate-700 rounded-t-lg">
      {/* Undo */}
      <button
        onClick={onUndo}
        disabled={!canUndo}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        title="Undo (Ctrl+Z)"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-300">
          <path d="M3 10h10a5 5 0 0 1 0 10H13" />
          <polyline points="3,10 7,6" />
          <polyline points="3,10 7,14" />
        </svg>
      </button>

      {/* Redo */}
      <button
        onClick={onRedo}
        disabled={!canRedo}
        className="p-1.5 rounded hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        title="Redo (Ctrl+Shift+Z)"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-slate-300">
          <path d="M21 10H11a5 5 0 0 0 0 10h2" />
          <polyline points="21,10 17,6" />
          <polyline points="21,10 17,14" />
        </svg>
      </button>

      <div className="w-px h-5 bg-slate-600 mx-1" />

      {/* Delete */}
      <button
        onClick={onDelete}
        disabled={selectedCount === 0}
        className="flex items-center gap-1.5 px-2 py-1 rounded text-xs hover:bg-red-500/20 hover:text-red-300 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-slate-300"
        title="Delete selected (Del)"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="3 6 5 6 21 6" />
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
        </svg>
        Delete
      </button>

      {/* Copy */}
      <button
        onClick={handleCopy}
        disabled={selectedCount === 0}
        className="flex items-center gap-1.5 px-2 py-1 rounded text-xs hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-slate-300"
        title="Copy selected (Ctrl+C)"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="9" y="9" width="13" height="13" rx="2" />
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
        </svg>
        Copy
      </button>

      {/* Paste */}
      <button
        onClick={onPaste}
        disabled={!hasCopiedWords}
        className="flex items-center gap-1.5 px-2 py-1 rounded text-xs hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-slate-300"
        title="Paste copied words (Ctrl+V)"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
          <rect x="8" y="2" width="8" height="4" rx="1" />
        </svg>
        Paste
      </button>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Selection info */}
      {selectedCount > 0 && (
        <span className="text-xs text-purple-300">
          {selectedCount} word{selectedCount !== 1 ? 's' : ''} selected
        </span>
      )}
      {hasEdits && selectedCount === 0 && (
        <span className="text-xs text-amber-400/70">Edited</span>
      )}
    </div>
  );
}
