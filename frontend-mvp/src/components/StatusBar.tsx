interface StatusBarProps {
  status: 'idle' | 'uploading' | 'transcribing' | 'done' | 'error';
  message?: string;
  processingTime?: number | null;
  duration?: number | null;
  language?: string | null;
}

export default function StatusBar({ status, message, processingTime, duration, language }: StatusBarProps) {
  const configs = {
    idle: { color: 'bg-slate-700', text: 'Ready - upload an audio file to get started' },
    uploading: { color: 'bg-yellow-500/20', text: 'Uploading and processing...' },
    transcribing: { color: 'bg-blue-500/20', text: 'Transcribing audio with WhisperX...' },
    done: { color: 'bg-green-500/20', text: 'Transcription complete!' },
    error: { color: 'bg-red-500/20', text: message || 'An error occurred' },
  };

  const config = configs[status];

  return (
    <div className={`${config.color} rounded-lg px-4 py-3 flex items-center justify-between`}>
      <div className="flex items-center gap-3">
        {(status === 'uploading' || status === 'transcribing') && (
          <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        )}
        <span className="text-sm">{status === 'error' ? message : config.text}</span>
      </div>
      {status === 'done' && (
        <div className="flex gap-4 text-xs text-slate-400">
          {language && <span>Language: {language}</span>}
          {duration != null && <span>Duration: {duration.toFixed(1)}s</span>}
          {processingTime != null && <span>Processed in: {processingTime.toFixed(1)}s</span>}
        </div>
      )}
    </div>
  );
}
