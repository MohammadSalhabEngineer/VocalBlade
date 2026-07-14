import React, { useRef, useEffect, useState } from 'react';
import type { TranscriptionSegment } from '../api/graphql';
import type { EditOperation } from '../utils/transcriptOps';

export interface AIChatEntry {
  role: 'user' | 'assistant';
  content: string;
  ops?: EditOperation[];
}

interface AIPanelProps {
  segments: TranscriptionSegment[];
  speakerNames: Record<string, string>;
  chatHistory: AIChatEntry[];
  aiStatus: 'idle' | 'thinking' | 'error';
  aiError: string;
  pendingOps: EditOperation[];
  onClose: () => void;
  onSendMessage: (msg: string) => void;
  onQuickAction: (instruction: string) => void;
  onSilenceRemove: (thresholdSec: number) => void;
  onRemoveSpeaker: (speakerId: string) => void;
  onApplyPending: () => void;
  onDiscardPending: () => void;
}

export default function AIPanel({
  segments,
  speakerNames,
  chatHistory,
  aiStatus,
  aiError,
  pendingOps,
  onClose,
  onSendMessage,
  onQuickAction,
  onSilenceRemove,
  onRemoveSpeaker,
  onApplyPending,
  onDiscardPending,
}: AIPanelProps) {
  const [input, setInput] = useState('');
  const [silenceThreshold, setSilenceThreshold] = useState(3);
  const [selectedSpeaker, setSelectedSpeaker] = useState('');
  const chatBottomRef = useRef<HTMLDivElement>(null);

  const speakerIds = Object.keys(speakerNames).length > 0
    ? Object.keys(speakerNames)
    : Array.from(new Set(segments.map((s) => s.speaker).filter(Boolean))) as string[];

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, aiStatus]);

  function handleSend() {
    const trimmed = input.trim();
    if (!trimmed || aiStatus === 'thinking') return;
    setInput('');
    onSendMessage(trimmed);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  const hasPending = pendingOps.length > 0;
  const pendingWordOps = pendingOps.filter((op) => op.type === 'delete' || op.type === 'move').length;
  const pendingRangeOps = pendingOps.filter((op) => op.type === 'delete_range').length;

  const previewLabel = (() => {
    const parts: string[] = [];
    if (pendingWordOps > 0) parts.push(`${pendingWordOps} word op${pendingWordOps !== 1 ? 's' : ''}`);
    if (pendingRangeOps > 0) parts.push(`${pendingRangeOps} range cut${pendingRangeOps !== 1 ? 's' : ''}`);
    return parts.join(', ');
  })();

  return (
    <div className="flex flex-col h-full w-80 bg-gray-900 border-l border-gray-700 text-sm text-gray-100">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700 shrink-0">
        <span className="font-semibold text-white flex items-center gap-2">
          <span>✨</span> AI Assistant
        </span>
        <button
          onClick={onClose}
          className="text-gray-400 hover:text-white transition-colors text-lg leading-none"
          aria-label="Close AI panel"
        >
          ×
        </button>
      </div>

      {/* Pending ops banner */}
      {hasPending && (
        <div className="mx-3 mt-3 p-3 bg-blue-900/50 border border-blue-600 rounded-lg shrink-0">
          <div className="text-blue-300 text-xs font-medium mb-2">
            Pending: {previewLabel}
          </div>
          <div className="flex gap-2">
            <button
              onClick={onApplyPending}
              className="flex-1 py-1.5 bg-blue-600 hover:bg-blue-500 rounded text-white text-xs font-medium transition-colors"
            >
              Apply
            </button>
            <button
              onClick={onDiscardPending}
              className="flex-1 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-gray-200 text-xs transition-colors"
            >
              Discard
            </button>
          </div>
        </div>
      )}

      {/* Quick Actions */}
      <div className="px-3 pt-3 pb-2 border-b border-gray-700 shrink-0">
        <div className="text-xs text-gray-400 font-medium mb-2 uppercase tracking-wide">Quick Actions</div>
        <div className="flex flex-col gap-1.5">
          {/* Remove filler words */}
          <button
            disabled={aiStatus === 'thinking'}
            onClick={() =>
              onQuickAction(
                'Remove all filler words and verbal tics such as "um", "uh", "like", "you know", "kind of", "sort of", "basically", "literally", "right", "I mean", "well".',
              )
            }
            className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-gray-200 text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            ✂ Remove filler words
          </button>

          {/* Remove silences */}
          <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded">
            <span className="text-gray-300 text-xs shrink-0">⏸ Silences &gt;</span>
            <input
              type="number"
              min={1}
              max={60}
              value={silenceThreshold}
              onChange={(e) => setSilenceThreshold(Math.max(1, Number(e.target.value)))}
              className="w-12 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-center text-white"
            />
            <span className="text-gray-400 text-xs shrink-0">s</span>
            <button
              onClick={() => onSilenceRemove(silenceThreshold)}
              className="ml-auto px-2 py-1 bg-blue-700 hover:bg-blue-600 rounded text-white text-xs transition-colors"
            >
              Apply
            </button>
          </div>

          {/* Remove speaker */}
          {speakerIds.length > 0 && (
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded">
              <span className="text-gray-300 text-xs shrink-0">👤 Remove</span>
              <select
                value={selectedSpeaker}
                onChange={(e) => setSelectedSpeaker(e.target.value)}
                className="flex-1 bg-gray-700 border border-gray-600 rounded px-1 py-0.5 text-xs text-white"
              >
                <option value="">— speaker —</option>
                {speakerIds.map((id) => (
                  <option key={id} value={id}>
                    {speakerNames[id] || id}
                  </option>
                ))}
              </select>
              <button
                disabled={!selectedSpeaker}
                onClick={() => selectedSpeaker && onRemoveSpeaker(selectedSpeaker)}
                className="px-2 py-1 bg-red-800 hover:bg-red-700 rounded text-white text-xs transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Remove
              </button>
            </div>
          )}

          {/* Summarize */}
          <button
            disabled={aiStatus === 'thinking'}
            onClick={() => onSendMessage('Summarize this transcript in 3 bullet points.')}
            className="w-full text-left px-3 py-2 bg-gray-800 hover:bg-gray-700 rounded text-gray-200 text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            📝 Summarize transcript
          </button>
        </div>
      </div>

      {/* Chat history */}
      <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-3 min-h-0">
        {chatHistory.length === 0 && (
          <p className="text-gray-500 text-xs text-center mt-4">
            Ask anything about your transcript, or use the quick actions above.
          </p>
        )}
        {chatHistory.map((entry, i) => (
          <div
            key={i}
            className={`flex flex-col gap-1 ${entry.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[90%] rounded-lg px-3 py-2 text-xs leading-relaxed ${
                entry.role === 'user'
                  ? 'bg-blue-700 text-white rounded-br-none'
                  : 'bg-gray-800 text-gray-200 rounded-bl-none'
              }`}
            >
              {entry.content}
            </div>
            {entry.role === 'assistant' && entry.ops && entry.ops.length > 0 && (
              <div className="text-xs text-blue-400 pl-1">
                {entry.ops.length} edit op{entry.ops.length !== 1 ? 's' : ''} proposed
              </div>
            )}
          </div>
        ))}

        {aiStatus === 'thinking' && (
          <div className="flex items-start">
            <div className="bg-gray-800 rounded-lg rounded-bl-none px-3 py-2 text-xs text-gray-400 animate-pulse">
              Thinking…
            </div>
          </div>
        )}

        {aiStatus === 'error' && aiError && (
          <div className="text-xs text-red-400 bg-red-900/30 border border-red-700 rounded px-3 py-2">
            {aiError}
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* Input */}
      <div className="px-3 pb-3 pt-2 border-t border-gray-700 shrink-0">
        <div className="flex gap-2 items-end">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything… (Enter to send)"
            rows={2}
            disabled={aiStatus === 'thinking'}
            className="flex-1 bg-gray-800 border border-gray-600 rounded px-3 py-2 text-xs text-white placeholder-gray-500 resize-none focus:outline-none focus:border-blue-500 transition-colors disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || aiStatus === 'thinking'}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed self-stretch"
          >
            ↑
          </button>
        </div>
      </div>
    </div>
  );
}
