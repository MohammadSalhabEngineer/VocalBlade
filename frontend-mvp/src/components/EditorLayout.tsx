import type { ReactNode } from 'react';

interface EditorLayoutProps {
  mediaPanel: ReactNode;
  transcriptPanel: ReactNode;
  statusBar: ReactNode;
  aiPanel?: ReactNode;
}

export default function EditorLayout({ mediaPanel, transcriptPanel, statusBar, aiPanel }: EditorLayoutProps) {
  return (
    <div className="flex flex-col h-[calc(100vh-52px)]">
      {/* Main content area */}
      <div className="flex-1 flex flex-col lg:flex-row gap-0 min-h-0">
        {/* Left: Media player */}
        <div className="lg:w-[58%] flex-shrink-0 p-3 pb-1 lg:pb-3">
          {mediaPanel}
        </div>

        {/* Right: Transcript editor + optional AI panel */}
        <div className="flex-1 min-h-0 flex flex-row border-l border-slate-700/50">
          <div className="flex-1 min-w-0 flex flex-col min-h-0">
            {transcriptPanel}
          </div>
          {aiPanel && (
            <div className="flex-shrink-0 min-h-0 overflow-hidden flex flex-col">
              {aiPanel}
            </div>
          )}
        </div>
      </div>

      {/* Status bar */}
      <div className="flex-shrink-0 border-t border-slate-700/50">
        {statusBar}
      </div>
    </div>
  );
}
