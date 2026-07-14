import { memo, useCallback } from 'react';

interface WordSpanProps {
  word: string;
  wordId: string;
  isActive: boolean;
  isSelected: boolean;
  isDeleted: boolean;
  isDragOver: boolean;
  onWordClick: (wordId: string, e: React.MouseEvent) => void;
  onDragStart: (wordId: string, e: React.DragEvent) => void;
  onDragOver: (wordId: string, e: React.DragEvent) => void;
  onDragLeave: () => void;
  onDrop: (wordId: string, e: React.DragEvent) => void;
}

function WordSpanInner({
  word,
  wordId,
  isActive,
  isSelected,
  isDeleted,
  isDragOver,
  onWordClick,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
}: WordSpanProps) {
  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onWordClick(wordId, e);
    },
    [wordId, onWordClick]
  );

  const handleDragStart = useCallback(
    (e: React.DragEvent) => onDragStart(wordId, e),
    [wordId, onDragStart]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      onDragOver(wordId, e);
    },
    [wordId, onDragOver]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      onDrop(wordId, e);
    },
    [wordId, onDrop]
  );

  let className =
    'inline-block px-1 py-0.5 rounded text-sm cursor-pointer transition-all duration-100 select-none ';

  if (isDeleted) {
    className += 'line-through opacity-30 bg-red-500/10 text-red-300 ';
  } else if (isActive && isSelected) {
    className += 'bg-blue-500/40 text-white font-medium ring-2 ring-purple-400 ';
  } else if (isActive) {
    className += 'bg-blue-500/30 text-white font-medium scale-105 ';
  } else if (isSelected) {
    className += 'bg-purple-500/25 text-purple-200 ring-1 ring-purple-400/60 ';
  } else {
    className += 'text-slate-300 hover:bg-slate-700/50 ';
  }

  // Drop indicator: blue left border when dragging over
  if (isDragOver && !isDeleted) {
    className += 'border-l-2 border-blue-400 ';
  }

  return (
    <span
      className={className}
      onClick={handleClick}
      data-word-id={wordId}
      draggable={isSelected && !isDeleted}
      onDragStart={handleDragStart}
      onDragOver={handleDragOver}
      onDragLeave={onDragLeave}
      onDrop={handleDrop}
    >
      {word}
    </span>
  );
}

export default memo(WordSpanInner);
