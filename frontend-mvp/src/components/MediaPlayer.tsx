import { useEffect, useRef, useState, useCallback } from 'react';

interface MediaPlayerProps {
  file: File;
  fileType: 'audio' | 'video';
  isPlaying: boolean;
  currentTime: number;
  onMediaRef: (el: HTMLVideoElement | HTMLAudioElement | null) => void;
  onTogglePlayPause: () => void;
  onSeek: (time: number) => void;
  onVolumeChange: (volume: number) => void;
  onPlaybackRateChange: (rate: number) => void;
}

function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

const SPEEDS = [0.5, 0.75, 1, 1.25, 1.5, 2];

export default function MediaPlayer({
  file,
  fileType,
  isPlaying,
  currentTime,
  onMediaRef,
  onTogglePlayPause,
  onSeek,
  onVolumeChange,
  onPlaybackRateChange,
}: MediaPlayerProps) {
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [speed, setSpeed] = useState(1);
  const [showSpeedMenu, setShowSpeedMenu] = useState(false);
  const progressRef = useRef<HTMLDivElement>(null);

  // Stable local ref for the media element
  const localMediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);

  // Create object URL per file
  const [objectUrl, setObjectUrl] = useState<string>('');
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setObjectUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Wire up the media element to the parent sync hook via useEffect
  // This avoids callback ref instability
  const onMediaRefRef = useRef(onMediaRef);
  useEffect(() => { onMediaRefRef.current = onMediaRef; }, [onMediaRef]);

  const setLocalRef = useCallback((el: HTMLVideoElement | HTMLAudioElement | null) => {
    localMediaRef.current = el;
    // Notify parent immediately on mount
    onMediaRefRef.current(el);
  }, []);

  // Reload media element when source URL changes
  const prevUrlRef = useRef<string>('');
  useEffect(() => {
    const el = localMediaRef.current;
    if (!el || !objectUrl) return;
    // Only reload if URL actually changed (not on initial mount)
    if (prevUrlRef.current && prevUrlRef.current !== objectUrl) {
      el.load();
      el.currentTime = 0;
    }
    prevUrlRef.current = objectUrl;
  }, [objectUrl]);

  // Listen for metadata load
  useEffect(() => {
    const el = localMediaRef.current;
    if (!el) return;

    const handleMetadata = () => setDuration(el.duration);
    // If metadata already loaded (e.g., cached)
    if (el.readyState >= 1) {
      setDuration(el.duration);
    }
    el.addEventListener('loadedmetadata', handleMetadata);
    return () => el.removeEventListener('loadedmetadata', handleMetadata);
  }, [objectUrl]); // re-attach when src changes

  const handleProgressClick = useCallback(
    (e: React.MouseEvent) => {
      const bar = progressRef.current;
      if (!bar || duration === 0) return;
      const rect = bar.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      onSeek(fraction * duration);
    },
    [duration, onSeek]
  );

  const handleVolumeChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = parseFloat(e.target.value);
      setVolume(v);
      onVolumeChange(v);
    },
    [onVolumeChange]
  );

  const handleSpeedChange = useCallback(
    (rate: number) => {
      setSpeed(rate);
      onPlaybackRateChange(rate);
      setShowSpeedMenu(false);
    },
    [onPlaybackRateChange]
  );

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="flex flex-col bg-slate-900 rounded-xl overflow-hidden">
      {/* Media element */}
      {fileType === 'video' ? (
        <video
          ref={setLocalRef as any}
          src={objectUrl || undefined}
          className="w-full aspect-video bg-black"
          preload="metadata"
          playsInline
        />
      ) : (
        <div className="w-full aspect-video bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4 opacity-40">&#9836;</div>
            <p className="text-slate-400 text-sm">{file.name}</p>
            <p className="text-slate-500 text-xs mt-1">
              {duration > 0 ? formatTimestamp(duration) : 'Loading...'}
            </p>
          </div>
          <audio ref={setLocalRef as any} src={objectUrl || undefined} preload="metadata" />
        </div>
      )}

      {/* Progress bar */}
      <div
        ref={progressRef}
        className="h-1.5 bg-slate-700 cursor-pointer group relative"
        onClick={handleProgressClick}
      >
        <div
          className="h-full bg-blue-500 transition-[width] duration-75"
          style={{ width: `${progress}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-blue-400 rounded-full opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
          style={{ left: `${progress}%`, transform: `translate(-50%, -50%)` }}
        />
      </div>

      {/* Controls bar */}
      <div className="flex items-center gap-3 px-3 py-2 bg-slate-800/80">
        {/* Play/Pause */}
        <button
          onClick={onTogglePlayPause}
          className="text-white hover:text-blue-400 transition-colors p-1"
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
        </button>

        {/* Time display */}
        <span className="text-xs text-slate-400 font-mono min-w-[80px]">
          {formatTimestamp(currentTime)} / {formatTimestamp(duration)}
        </span>

        <div className="flex-1" />

        {/* Volume */}
        <div className="flex items-center gap-1.5">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="text-slate-400">
            <polygon points="11,5 6,9 2,9 2,15 6,15 11,19" />
            {volume > 0 && <path d="M14,8 C15.5,9.5 15.5,14.5 14,16" fill="none" stroke="currentColor" strokeWidth="1.5" />}
            {volume > 0.5 && <path d="M17,5 C20,8 20,16 17,19" fill="none" stroke="currentColor" strokeWidth="1.5" />}
          </svg>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            onChange={handleVolumeChange}
            className="w-16 h-1 accent-blue-500"
          />
        </div>

        {/* Speed */}
        <div className="relative">
          <button
            onClick={() => setShowSpeedMenu(!showSpeedMenu)}
            className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded transition-colors"
          >
            {speed}x
          </button>
          {showSpeedMenu && (
            <div className="absolute bottom-full right-0 mb-1 bg-slate-700 rounded-lg shadow-xl border border-slate-600 overflow-hidden z-10">
              {SPEEDS.map((s) => (
                <button
                  key={s}
                  onClick={() => handleSpeedChange(s)}
                  className={`block w-full text-left px-3 py-1.5 text-xs hover:bg-slate-600 transition-colors ${
                    s === speed ? 'text-blue-400 bg-slate-600/50' : 'text-slate-300'
                  }`}
                >
                  {s}x
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
