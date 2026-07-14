import { useCallback, useMemo, useRef, useEffect } from 'react';
import AudioUpload from './components/AudioUpload';
import TranscriptionParamsPanel from './components/TranscriptionParams';
import TranscriptDisplay from './components/TranscriptDisplay';
import StatusBar from './components/StatusBar';
import MediaPlayer from './components/MediaPlayer';
import EditorLayout from './components/EditorLayout';
import ExportButton from './components/ExportButton';
import SmartCutButton from './components/SmartCutButton';
import ContinuitySelector from './components/ContinuitySelector';
import AIPanel from './components/AIPanel';
import { transcribeAudio, semanticEdit, aiChat, semanticOpsToEditOps } from './api/graphql';
import { useEditorState } from './hooks/useEditorState';
import { useMediaSync } from './hooks/useMediaSync';
import {
  computePlaybackSchedule,
  findSilenceGaps,
} from './utils/transcriptOps';
import type { WordId, EditOperation } from './utils/transcriptOps';

function App() {
  const { state, actions, flatWords, hasEdits, canUndo, canRedo, isEditorMode } =
    useEditorState();

  // Playback schedule — ordered time ranges from effective word list
  const playbackSchedule = useMemo(() => computePlaybackSchedule(flatWords), [flatWords]);

  const seekRef = useRef<((time: number) => void) | null>(null);
  const pauseRef = useRef<(() => void) | null>(null);
  const scheduleIdxRef = useRef(0);
  const lastSeekRef = useRef<{ target: number; ts: number } | null>(null);

  useEffect(() => { scheduleIdxRef.current = 0; }, [playbackSchedule]);

  const handleTimeUpdate = useCallback(
    (time: number) => {
      actions.setCurrentTime(time);
      if (!hasEdits) return;
      if (playbackSchedule.length === 0) return;

      let idx = scheduleIdxRef.current;
      if (idx >= playbackSchedule.length) return;

      const seg = playbackSchedule[idx];

      if (time >= seg.start - 0.05 && time < seg.end - 0.02) return;

      if (time >= seg.end - 0.02) {
        const nextIdx = idx + 1;
        if (nextIdx >= playbackSchedule.length) {
          scheduleIdxRef.current = 0;
          seekRef.current?.(playbackSchedule[0].start);
          pauseRef.current?.();
          return;
        }
        scheduleIdxRef.current = nextIdx;
        const next = playbackSchedule[nextIdx];
        if (Math.abs(next.start - seg.end) > 0.1) {
          const last = lastSeekRef.current;
          if (last && Math.abs(last.target - next.start) < 0.05 && performance.now() - last.ts < 500) return;
          lastSeekRef.current = { target: next.start, ts: performance.now() };
          seekRef.current?.(next.start);
        }
        return;
      }

      for (let i = 0; i < playbackSchedule.length; i++) {
        const s = playbackSchedule[i];
        if (time >= s.start - 0.05 && time < s.end + 0.05) {
          scheduleIdxRef.current = i;
          return;
        }
      }

      const target = playbackSchedule[scheduleIdxRef.current];
      if (target && Math.abs(time - target.start) > 0.1) {
        const last = lastSeekRef.current;
        if (!last || Math.abs(last.target - target.start) > 0.05 || performance.now() - last.ts > 500) {
          lastSeekRef.current = { target: target.start, ts: performance.now() };
          seekRef.current?.(target.start);
        }
      }
    },
    [actions, playbackSchedule, hasEdits]
  );

  const mediaSync = useMediaSync({
    onTimeUpdate: handleTimeUpdate,
    onPlayStateChange: actions.setPlaying,
  });

  useEffect(() => { seekRef.current = mediaSync.seekTo; }, [mediaSync.seekTo]);
  useEffect(() => { pauseRef.current = () => mediaSync.mediaRef.current?.pause(); }, [mediaSync.mediaRef]);

  // --- Handlers ---

  const handleFileSelected = useCallback(
    (f: File) => { actions.setFile(f); },
    [actions]
  );

  const handleTranscribe = useCallback(async () => {
    if (!state.file) return;
    actions.setStatus('transcribing');
    try {
      const response = await transcribeAudio(state.file, state.transcriptionParams);
      actions.setResult(response);
    } catch (err) {
      actions.setStatus('error', err instanceof Error ? err.message : 'Unknown error');
    }
  }, [state.file, state.transcriptionParams, actions]);

  const handleWordClick = useCallback(
    (wordId: WordId, e: React.MouseEvent) => {
      if (e.shiftKey) {
        actions.shiftSelectWord(wordId);
      } else if (e.ctrlKey || e.metaKey) {
        actions.toggleSelectWord(wordId);
      } else {
        actions.selectWord(wordId);
      }
    },
    [actions]
  );

  const handleCopy = useCallback(() => { actions.copyWords(); }, [actions]);

  // --- AI Panel handlers ---

  const buildTranscriptJson = useCallback(() => {
    if (!state.result?.segments) return '[]';
    return JSON.stringify(state.result.segments);
  }, [state.result?.segments]);

  const handleAiSendMessage = useCallback(async (msg: string) => {
    const transcript = buildTranscriptJson();
    actions.addAiMessage('user', msg);
    actions.setAiStatus('thinking');

    try {
      const history = state.aiChatHistory.map((h) => ({ role: h.role, content: h.content }));
      const resp = await aiChat(transcript, msg, history, state.speakerNames);

      if (resp.errorMessage) {
        actions.setAiStatus('error', resp.errorMessage);
        return;
      }

      const editOps = resp.operations ? semanticOpsToEditOps(resp.operations) : undefined;
      actions.addAiMessage('assistant', resp.reply || '(no reply)', editOps);
      actions.setAiStatus('idle');

      if (editOps && editOps.length > 0) {
        actions.setAiPending(editOps);
      }
    } catch (err) {
      actions.setAiStatus('error', err instanceof Error ? err.message : 'AI request failed');
    }
  }, [buildTranscriptJson, state.aiChatHistory, state.speakerNames, actions]);

  const handleAiQuickAction = useCallback(async (instruction: string) => {
    const transcript = buildTranscriptJson();
    actions.addAiMessage('user', instruction);
    actions.setAiStatus('thinking');

    try {
      const resp = await semanticEdit(transcript, instruction, state.speakerNames);

      if (resp.errorMessage) {
        actions.setAiStatus('error', resp.errorMessage);
        return;
      }

      const editOps = resp.operations ? semanticOpsToEditOps(resp.operations) : undefined;
      const reply = resp.explanation || 'Done.';
      actions.addAiMessage('assistant', reply, editOps);
      actions.setAiStatus('idle');

      if (editOps && editOps.length > 0) {
        actions.setAiPending(editOps);
      }
    } catch (err) {
      actions.setAiStatus('error', err instanceof Error ? err.message : 'AI request failed');
    }
  }, [buildTranscriptJson, state.speakerNames, actions]);

  const handleSilenceRemove = useCallback((thresholdSec: number) => {
    if (!state.result?.segments) return;
    const gaps = findSilenceGaps(state.result.segments, thresholdSec);
    if (gaps.length === 0) {
      actions.addAiMessage('assistant', `No silences longer than ${thresholdSec}s found.`);
      return;
    }
    const ops: EditOperation[] = gaps.map((g) => ({ type: 'delete_range', start: g.start, end: g.end }));
    actions.addAiMessage('assistant', `Found ${gaps.length} silence gap${gaps.length !== 1 ? 's' : ''} longer than ${thresholdSec}s.`, ops);
    actions.setAiPending(ops);
  }, [state.result?.segments, actions]);

  const handleRemoveSpeaker = useCallback((speakerId: string) => {
    if (!state.result?.segments) return;
    const wordIds: WordId[] = [];
    state.result.segments.forEach((seg, si) => {
      if (seg.speaker === speakerId) {
        (seg.words || []).forEach((_, wi) => {
          wordIds.push(`${si}-${wi}` as WordId);
        });
      }
    });

    if (wordIds.length === 0) {
      actions.addAiMessage('assistant', `No words found for speaker "${state.speakerNames[speakerId] || speakerId}".`);
      return;
    }

    const ops: EditOperation[] = [{ type: 'delete', wordIds }];
    const label = state.speakerNames[speakerId] || speakerId;
    actions.addAiMessage('assistant', `Marked ${wordIds.length} words from ${label} for deletion.`, ops);
    actions.setAiPending(ops);
  }, [state.result?.segments, state.speakerNames, actions]);

  const isProcessing = state.transcriptionStatus === 'transcribing';

  // --- Editor mode (after transcription) ---
  if (isEditorMode && state.file && state.fileType) {
    const statusError = state.exportStatus === 'error' ? state.exportError : state.errorMessage;

    return (
      <div className="h-screen bg-slate-900 flex flex-col">
        {/* Header */}
        <header className="flex-shrink-0 border-b border-slate-700 bg-slate-800/80">
          <div className="flex items-center justify-between px-4 py-2.5">
            <div className="flex items-center gap-3">
              <h1 className="text-lg font-bold text-white">VocalBlade</h1>
              <span className="text-xs text-slate-500 hidden sm:inline">AI-Powered Video Editor</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={actions.reset}
                className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded hover:bg-slate-700 transition-colors"
              >
                New
              </button>

              <ContinuitySelector
                audioContinuityMethod={state.continuityMethod}
                onAudioChange={actions.setContinuityMethod}
                videoInterpolationMethod={state.videoInterpolationMethod}
                onVideoChange={actions.setVideoInterpolationMethod}
                hasVideo={state.fileType === 'video'}
                smartEditEnabled={state.smartEditEnabled}
                onSmartEditChange={actions.setSmartEdit}
              />

              <SmartCutButton
                file={state.file}
                segments={state.result!.segments!}
                editOperations={state.editOperations}
                hasEdits={hasEdits}
                duration={state.result?.duration}
                smartCutStatus={state.smartCutStatus}
                smartCutUrl={state.smartCutUrl}
                onStatusChange={actions.setSmartCutStatus}
              />

              <ExportButton
                file={state.file}
                segments={state.result!.segments!}
                editOperations={state.editOperations}
                hasEdits={hasEdits}
                exportStatus={state.exportStatus}
                exportUrl={state.exportUrl}
                continuityMethod={state.continuityMethod}
                videoInterpolationMethod={state.videoInterpolationMethod}
                smartEditEnabled={state.smartEditEnabled}
                duration={state.result?.duration}
                onExportStatusChange={actions.setExportStatus}
              />

              {/* AI Panel toggle */}
              <button
                onClick={() => actions.setAiPanelOpen(!state.aiPanelOpen)}
                title="AI Assistant"
                className={`text-sm px-2.5 py-1.5 rounded transition-colors ${
                  state.aiPanelOpen
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-300 hover:text-white hover:bg-slate-700'
                }`}
              >
                ✨ AI
              </button>
            </div>
          </div>
        </header>

        {/* Editor */}
        <EditorLayout
          mediaPanel={
            <MediaPlayer
              file={state.file}
              fileType={state.fileType}
              isPlaying={state.isPlaying}
              currentTime={state.currentTime}
              onMediaRef={mediaSync.setMediaElement}
              onTogglePlayPause={mediaSync.togglePlayPause}
              onSeek={mediaSync.seekTo}
              onVolumeChange={mediaSync.setVolume}
              onPlaybackRateChange={mediaSync.setPlaybackRate}
            />
          }
          transcriptPanel={
            <TranscriptDisplay
              flatWords={flatWords}
              currentTime={state.currentTime}
              selectedWordIds={state.selectedWordIds}
              textDirection={state.textDirection}
              canUndo={canUndo}
              canRedo={canRedo}
              hasEdits={hasEdits}
              speakerNames={state.speakerNames}
              hasCopiedWords={state.copiedWordIds.length > 0}
              onWordClick={handleWordClick}
              onSeek={mediaSync.seekTo}
              onDelete={actions.deleteSelected}
              onCopy={handleCopy}
              onPaste={actions.pasteWords}
              onUndo={actions.undo}
              onRedo={actions.redo}
              onSelectAll={actions.selectAll}
              onClearSelection={actions.clearSelection}
              onMoveWords={actions.moveWords}
              onRenameSpeaker={actions.renameSpeaker}
            />
          }
          statusBar={
            <StatusBar
              status={state.transcriptionStatus}
              message={statusError}
              processingTime={state.result?.processingTime}
              duration={state.result?.duration}
              language={state.result?.language}
            />
          }
          aiPanel={
            state.aiPanelOpen ? (
              <AIPanel
                segments={state.result!.segments!}
                speakerNames={state.speakerNames}
                chatHistory={state.aiChatHistory}
                aiStatus={state.aiStatus}
                aiError={state.aiError}
                pendingOps={state.aiPendingOps}
                onClose={() => actions.setAiPanelOpen(false)}
                onSendMessage={handleAiSendMessage}
                onQuickAction={handleAiQuickAction}
                onSilenceRemove={handleSilenceRemove}
                onRemoveSpeaker={handleRemoveSpeaker}
                onApplyPending={actions.aiApplyOps}
                onDiscardPending={actions.aiDiscardPending}
              />
            ) : undefined
          }
        />
      </div>
    );
  }

  // --- Upload / Transcription mode ---
  return (
    <div className="min-h-screen bg-slate-900">
      <header className="border-b border-slate-700 bg-slate-800/50">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-white">VocalBlade</h1>
          <span className="text-xs text-slate-500">AI-Powered Video Editor</span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6 space-y-6">
        <StatusBar
          status={state.transcriptionStatus}
          message={state.errorMessage}
          processingTime={state.result?.processingTime}
          duration={state.result?.duration}
          language={state.result?.language}
        />

        <AudioUpload onFileSelected={handleFileSelected} disabled={isProcessing} />

        {state.file && (
          <div className="bg-slate-800 rounded-xl p-4 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-slate-200">{state.file.name}</p>
                <p className="text-xs text-slate-400">
                  {(state.file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button
                onClick={handleTranscribe}
                disabled={isProcessing}
                className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white px-6 py-2 rounded-lg font-medium text-sm transition-colors"
              >
                {isProcessing ? (
                  <span className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Transcribing...
                  </span>
                ) : (
                  'Transcribe'
                )}
              </button>
            </div>

            <TranscriptionParamsPanel
              params={state.transcriptionParams}
              onChange={actions.setParams}
              disabled={isProcessing}
            />
          </div>
        )}

        {state.transcriptionStatus === 'error' && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
            <p className="text-sm text-red-300">{state.errorMessage}</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
