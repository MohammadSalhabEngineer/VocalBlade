import { useCallback, useState } from 'react';
import type { TranscriptionSegment } from '../api/graphql';
import { exportEditedMedia } from '../api/graphql';
import type { EditOperation } from '../utils/transcriptOps';
import { computeKeepRanges } from '../utils/transcriptOps';

interface SmartCutButtonProps {
  file: File;
  segments: TranscriptionSegment[];
  editOperations: EditOperation[];
  hasEdits: boolean;
  duration: number | null | undefined;
  smartCutStatus: 'idle' | 'exporting' | 'done' | 'error';
  smartCutUrl: string | null;
  onStatusChange: (status: 'idle' | 'exporting' | 'done' | 'error', url?: string, error?: string) => void;
}

export default function SmartCutButton({
  file,
  segments,
  editOperations,
  hasEdits,
  duration,
  smartCutStatus,
  smartCutUrl,
  onStatusChange,
}: SmartCutButtonProps) {
  const [showMenu, setShowMenu] = useState(false);

  const handleSmartCut = useCallback(async () => {
    setShowMenu(false);
    onStatusChange('exporting');

    try {
      const keepRanges = computeKeepRanges(segments, editOperations, duration ?? undefined);

      if (keepRanges.length === 0 && editOperations.length > 0) {
        onStatusChange('error', undefined, 'Nothing to export — all content was deleted');
        return;
      }

      const result = await exportEditedMedia(file, keepRanges, 'SMART_CUT', 'NONE');

      if (result.status === 'FAILED') {
        onStatusChange('error', undefined, result.errorMessage || 'Smart Cut failed');
      } else if (result.fileUrl) {
        onStatusChange('done', result.fileUrl);
      } else {
        onStatusChange('error', undefined, 'Smart Cut completed but no file URL returned');
      }
    } catch (err) {
      onStatusChange('error', undefined, err instanceof Error ? err.message : 'Smart Cut failed');
    }
  }, [file, segments, editOperations, duration, onStatusChange]);

  const handleDownload = useCallback(() => {
    if (!smartCutUrl) return;
    const a = document.createElement('a');
    a.href = smartCutUrl;
    a.download = `smartcut_${file.name}`;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setShowMenu(false);
  }, [smartCutUrl, file.name]);

  const isExporting = smartCutStatus === 'exporting';

  return (
    <div className="relative">
      <button
        onClick={() => {
          if (smartCutStatus === 'done' && smartCutUrl) {
            handleDownload();
          } else if (!isExporting) {
            if (hasEdits) {
              handleSmartCut();
            } else {
              setShowMenu(!showMenu);
            }
          }
        }}
        disabled={isExporting}
        title="Smart Cut: FOMM neural face animation + speech crossfade audio"
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
          smartCutStatus === 'done'
            ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
            : isExporting
            ? 'bg-slate-600 text-slate-400 cursor-wait'
            : hasEdits
            ? 'bg-indigo-600 hover:bg-indigo-500 text-white'
            : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
        }`}
      >
        {isExporting ? (
          <>
            <div className="w-3.5 h-3.5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
            FOMM...
          </>
        ) : smartCutStatus === 'done' ? (
          <>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download
          </>
        ) : (
          <>
            {/* Film strip icon */}
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
              <line x1="7" y1="2" x2="7" y2="22" />
              <line x1="17" y1="2" x2="17" y2="22" />
              <line x1="2" y1="12" x2="22" y2="12" />
              <line x1="2" y1="7" x2="7" y2="7" />
              <line x1="2" y1="17" x2="7" y2="17" />
              <line x1="17" y1="17" x2="22" y2="17" />
              <line x1="17" y1="7" x2="22" y2="7" />
            </svg>
            Smart Cut
          </>
        )}
      </button>

      {/* No-edits dropdown */}
      {showMenu && !hasEdits && (
        <div className="absolute top-full right-0 mt-1 bg-slate-700 rounded-lg shadow-xl border border-slate-600 overflow-hidden z-20 w-56">
          <button
            onClick={handleSmartCut}
            className="block w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-600"
          >
            Smart Cut original (no edits)
          </button>
          <div className="px-3 py-2 text-[10px] text-slate-500 border-t border-slate-600">
            FOMM neural face animation + speech crossfade at cut points.
          </div>
        </div>
      )}
    </div>
  );
}
