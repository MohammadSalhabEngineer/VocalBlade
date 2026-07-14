import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import type { EffectiveWord, WordId } from '../utils/transcriptOps';
import { findActiveWordIndex } from '../utils/transcriptOps';
import WordSpan from './WordSpan';
import EditingToolbar from './EditingToolbar';

interface TranscriptDisplayProps {
  flatWords: EffectiveWord[];
  currentTime: number;
  selectedWordIds: Set<WordId>;
  textDirection: 'ltr' | 'rtl';
  canUndo: boolean;
  canRedo: boolean;
  hasEdits: boolean;
  speakerNames: Record<string, string>;
  onWordClick: (wordId: WordId, e: React.MouseEvent) => void;
  onSeek: (time: number) => void;
  hasCopiedWords: boolean;
  onDelete: () => void;
  onCopy: () => void;
  onPaste: () => void;
  onUndo: () => void;
  onRedo: () => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
  onMoveWords: (targetWordId: WordId) => void;
  onRenameSpeaker: (originalId: string, newName: string) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${m}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
}

// Group flat words back into speaker segments for display
interface DisplaySegment {
  speaker: string | null;
  startTime: number;
  endTime: number;
  words: EffectiveWord[];
}

function groupBySpeaker(flatWords: EffectiveWord[]): DisplaySegment[] {
  if (flatWords.length === 0) return [];

  const segments: DisplaySegment[] = [];
  let current: DisplaySegment = {
    speaker: flatWords[0].speaker,
    startTime: flatWords[0].start,
    endTime: flatWords[0].end,
    words: [flatWords[0]],
  };

  for (let i = 1; i < flatWords.length; i++) {
    const w = flatWords[i];
    if (w.speaker === current.speaker) {
      current.words.push(w);
      current.endTime = Math.max(current.endTime, w.end);
    } else {
      segments.push(current);
      current = {
        speaker: w.speaker,
        startTime: w.start,
        endTime: w.end,
        words: [w],
      };
    }
  }
  segments.push(current);
  return segments;
}

// Speaker colors
const SPEAKER_COLORS: Record<string, string> = {};
const COLOR_PALETTE = [
  'bg-purple-500/20 text-purple-300',
  'bg-blue-500/20 text-blue-300',
  'bg-green-500/20 text-green-300',
  'bg-amber-500/20 text-amber-300',
  'bg-pink-500/20 text-pink-300',
  'bg-cyan-500/20 text-cyan-300',
  'bg-rose-500/20 text-rose-300',
  'bg-teal-500/20 text-teal-300',
];

function getSpeakerColor(speaker: string): string {
  if (!SPEAKER_COLORS[speaker]) {
    SPEAKER_COLORS[speaker] = COLOR_PALETTE[Object.keys(SPEAKER_COLORS).length % COLOR_PALETTE.length];
  }
  return SPEAKER_COLORS[speaker];
}

// --- Speaker Badge with inline rename ---
function SpeakerBadge({
  speakerId,
  displayName,
  colorClass,
  onRename,
}: {
  speakerId: string;
  displayName: string;
  colorClass: string;
  onRename: (originalId: string, newName: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(displayName);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editing]);

  useEffect(() => {
    setValue(displayName);
  }, [displayName]);

  const commit = useCallback(() => {
    const trimmed = value.trim();
    if (trimmed && trimmed !== speakerId) {
      onRename(speakerId, trimmed);
    } else if (!trimmed) {
      // Reset to original if empty
      setValue(displayName);
    }
    setEditing(false);
  }, [value, speakerId, displayName, onRename]);

  if (editing) {
    return (
      <input
        ref={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
          if (e.key === 'Escape') {
            setValue(displayName);
            setEditing(false);
          }
          e.stopPropagation(); // Don't trigger global keyboard shortcuts
        }}
        onClick={(e) => e.stopPropagation()}
        className="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-600 text-white border border-blue-400 outline-none w-32"
      />
    );
  }

  return (
    <span
      className={`text-xs px-2 py-0.5 rounded-full font-medium cursor-pointer hover:ring-1 hover:ring-white/30 transition-all ${colorClass}`}
      onClick={(e) => { e.stopPropagation(); setEditing(true); }}
      title="Click to rename speaker"
    >
      {displayName}
    </span>
  );
}

export default function TranscriptDisplay({
  flatWords,
  currentTime,
  selectedWordIds,
  textDirection,
  canUndo,
  canRedo,
  hasEdits,
  speakerNames,
  hasCopiedWords,
  onWordClick,
  onSeek,
  onDelete,
  onCopy,
  onPaste,
  onUndo,
  onRedo,
  onSelectAll,
  onClearSelection,
  onMoveWords,
  onRenameSpeaker,
}: TranscriptDisplayProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [dragOverWordId, setDragOverWordId] = useState<string | null>(null);

  // Find active word
  const activeWordIndex = useMemo(
    () => findActiveWordIndex(flatWords, currentTime),
    [flatWords, currentTime]
  );
  const activeWordId = activeWordIndex >= 0 ? flatWords[activeWordIndex].wordId : null;

  // Group words by speaker for display
  const displaySegments = useMemo(() => groupBySpeaker(flatWords), [flatWords]);

  // Auto-scroll to active word
  useEffect(() => {
    if (!activeWordId) return;
    const el = document.querySelector(`[data-word-id="${activeWordId}"]`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, [activeWordId]);

  // Handle word click with modifier key detection
  const handleWordClick = useCallback(
    (wordId: WordId, e: React.MouseEvent) => {
      const w = flatWords.find((fw) => fw.wordId === wordId);
      if (w && !w.deleted) {
        onSeek(w.start);
      }
      onWordClick(wordId, e);
    },
    [flatWords, onSeek, onWordClick]
  );

  // --- Drag and drop handlers ---
  const handleDragStart = useCallback(
    (_wordId: string, e: React.DragEvent) => {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', 'vocalblade-move');
    },
    []
  );

  const handleDragOver = useCallback(
    (wordId: string, _e: React.DragEvent) => {
      if (!selectedWordIds.has(wordId)) {
        setDragOverWordId(wordId);
      }
    },
    [selectedWordIds]
  );

  const handleDragLeave = useCallback(() => {
    setDragOverWordId(null);
  }, []);

  const handleDrop = useCallback(
    (wordId: string, _e: React.DragEvent) => {
      setDragOverWordId(null);
      if (!selectedWordIds.has(wordId) && selectedWordIds.size > 0) {
        onMoveWords(wordId);
      }
    },
    [selectedWordIds, onMoveWords]
  );

  // Click on container background clears selection
  const handleContainerClick = useCallback(
    (e: React.MouseEvent) => {
      if (!(e.target as HTMLElement).closest('[data-word-id]') &&
          !(e.target as HTMLElement).closest('input')) {
        onClearSelection();
      }
    },
    [onClearSelection]
  );

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <EditingToolbar
        selectedCount={selectedWordIds.size}
        canUndo={canUndo}
        canRedo={canRedo}
        hasEdits={hasEdits}
        flatWords={flatWords}
        selectedWordIds={selectedWordIds}
        hasCopiedWords={hasCopiedWords}
        onDelete={onDelete}
        onCopy={onCopy}
        onPaste={onPaste}
        onUndo={onUndo}
        onRedo={onRedo}
        onSelectAll={onSelectAll}
        onClearSelection={onClearSelection}
      />

      {/* Scrollable transcript */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-3 py-2 space-y-3"
        onClick={handleContainerClick}
        onDragOver={(e) => e.preventDefault()}
      >
        {displaySegments.map((seg, segIdx) => (
          <div key={segIdx} data-segment className="group">
            {/* Segment header: speaker + time */}
            <div className="flex items-center gap-2 mb-1.5" dir="ltr">
              {seg.speaker && (
                <SpeakerBadge
                  speakerId={seg.speaker}
                  displayName={speakerNames[seg.speaker] || seg.speaker}
                  colorClass={getSpeakerColor(seg.speaker)}
                  onRename={onRenameSpeaker}
                />
              )}
              <span className="text-[10px] font-mono text-slate-500">
                {formatTime(seg.startTime)} - {formatTime(seg.endTime)}
              </span>
            </div>

            {/* Words */}
            <div className="flex flex-wrap gap-0.5" dir={textDirection}>
              {seg.words.map((w) => (
                <WordSpan
                  key={w.wordId}
                  word={w.word}
                  wordId={w.wordId}
                  isActive={w.wordId === activeWordId}
                  isSelected={selectedWordIds.has(w.wordId)}
                  isDeleted={w.deleted}
                  isDragOver={w.wordId === dragOverWordId}
                  onWordClick={handleWordClick}
                  onDragStart={handleDragStart}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                />
              ))}
            </div>
          </div>
        ))}

        {flatWords.length === 0 && (
          <div className="text-center text-slate-500 py-8">
            No transcript data available
          </div>
        )}
      </div>
    </div>
  );
}
