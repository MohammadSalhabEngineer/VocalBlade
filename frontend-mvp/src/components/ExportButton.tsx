import { useCallback, useState } from 'react';
import type { TranscriptionSegment, ContinuityMethod, VideoInterpolationMethod } from '../api/graphql';
import { exportEditedMedia } from '../api/graphql';
import type { EditOperation } from '../utils/transcriptOps';
import { computeKeepRanges } from '../utils/transcriptOps';

interface ExportButtonProps {
  file: File;
  segments: TranscriptionSegment[];
  editOperations: EditOperation[];
  hasEdits: boolean;
  exportStatus: 'idle' | 'exporting' | 'done' | 'error';
  exportUrl: string | null;
  continuityMethod: ContinuityMethod;
  videoInterpolationMethod: VideoInterpolationMethod;
  smartEditEnabled: boolean;
  duration: number | null | undefined;
  onExportStatusChange: (status: 'idle' | 'exporting' | 'done' | 'error', url?: string, error?: string) => void;
}

export default function ExportButton({
  file,
  segments,
  editOperations,
  hasEdits,
  exportStatus,
  exportUrl,
  continuityMethod,
  videoInterpolationMethod,
  smartEditEnabled,
  duration,
  onExportStatusChange,
}: ExportButtonProps) {
  const [showMenu, setShowMenu] = useState(false);

  const handleExport = useCallback(async () => {
    setShowMenu(false);

    onExportStatusChange('exporting');

    try {
      const keepRanges = computeKeepRanges(segments, editOperations, duration ?? undefined);

      // Empty ranges + edits = everything deleted; empty ranges + no edits = full file copy
      if (keepRanges.length === 0 && editOperations.length > 0) {
        onExportStatusChange('error', undefined, 'Nothing to export - all content was deleted');
        return;
      }

      const effectiveContinuity: ContinuityMethod = smartEditEnabled ? 'NEURAL_SMOOTH' : continuityMethod;
      const effectiveVideo: VideoInterpolationMethod = smartEditEnabled ? 'NONE' : videoInterpolationMethod;
      const segmentsJson = effectiveContinuity === 'BOUNDARY_CLONE'
        ? JSON.stringify(segments)
        : undefined;
      const result = await exportEditedMedia(file, keepRanges, effectiveContinuity, effectiveVideo, segmentsJson);

      if (result.status === 'FAILED') {
        onExportStatusChange('error', undefined, result.errorMessage || 'Export failed');
      } else if (result.fileUrl) {
        onExportStatusChange('done', result.fileUrl);
      } else {
        onExportStatusChange('error', undefined, 'Export completed but no file URL was returned');
      }
    } catch (err) {
      onExportStatusChange('error', undefined, err instanceof Error ? err.message : 'Export failed');
    }
  }, [file, segments, editOperations, continuityMethod, videoInterpolationMethod, smartEditEnabled, duration, onExportStatusChange]);

  const handleDownload = useCallback(() => {
    if (!exportUrl) return;
    const a = document.createElement('a');
    a.href = exportUrl;
    a.download = `edited_${file.name}`;
    a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setShowMenu(false);
    // Stay on "done" so user can download again; they can re-export by making new edits
  }, [exportUrl, file.name]);

  const isExporting = exportStatus === 'exporting';

  return (
    <div className="relative">
      <button
        onClick={() => {
          if (exportStatus === 'done' && exportUrl) {
            handleDownload();
          } else if (!isExporting) {
            if (hasEdits) {
              handleExport();
            } else {
              setShowMenu(!showMenu);
            }
          }
        }}
        disabled={isExporting}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
          exportStatus === 'done'
            ? 'bg-green-600 hover:bg-green-500 text-white'
            : isExporting
            ? 'bg-slate-600 text-slate-400 cursor-wait'
            : hasEdits
            ? 'bg-blue-600 hover:bg-blue-500 text-white'
            : 'bg-slate-700 hover:bg-slate-600 text-slate-300'
        }`}
      >
        {isExporting ? (
          <>
            <div className="w-3.5 h-3.5 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
            {smartEditEnabled ? 'Smart Edit...' :
             videoInterpolationMethod === 'RIFE' ? 'RIFE Neural...' :
             videoInterpolationMethod === 'BLEND' ? 'Motion Blend...' :
             continuityMethod === 'ENCODEC' ? 'Neural Codec...' :
             continuityMethod === 'AI_INPAINTING' ? 'AI Inpainting...' :
             continuityMethod === 'AI_REGENERATION' ? 'AI Regenerating...' :
             continuityMethod === 'BOUNDARY_CLONE' ? 'Cloning Voice...' :
             videoInterpolationMethod === 'FRAME_PAD' ? 'Frame Padding...' :
             'Exporting...'}
          </>
        ) : exportStatus === 'done' ? (
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
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Export
          </>
        )}
      </button>

      {/* No-edits dropdown */}
      {showMenu && !hasEdits && (
        <div className="absolute top-full right-0 mt-1 bg-slate-700 rounded-lg shadow-xl border border-slate-600 overflow-hidden z-20 w-48">
          <button
            onClick={handleExport}
            className="block w-full text-left px-3 py-2 text-xs text-slate-300 hover:bg-slate-600"
          >
            Export original (no edits)
          </button>
          <div className="px-3 py-2 text-[10px] text-slate-500 border-t border-slate-600">
            Make edits to the transcript first, then export to create an edited version.
          </div>
        </div>
      )}
    </div>
  );
}
