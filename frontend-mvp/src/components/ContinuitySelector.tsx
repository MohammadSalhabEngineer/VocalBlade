import { useState, useRef, useEffect } from 'react';
import type { ContinuityMethod, VideoInterpolationMethod } from '../api/graphql';

interface ContinuitySelectorProps {
  audioContinuityMethod: ContinuityMethod;
  onAudioChange: (method: ContinuityMethod) => void;
  videoInterpolationMethod: VideoInterpolationMethod;
  onVideoChange: (method: VideoInterpolationMethod) => void;
  hasVideo?: boolean;
  smartEditEnabled: boolean;
  onSmartEditChange: (enabled: boolean) => void;
}

type MethodEntry<V> = {
  value: V;
  label: string;
  description: string;
  badge: string;
  badgeColor: string;
  icon: string;
};

// ── Visible audio methods ──────────────────────────────────────────────────
const AUDIO_METHODS: MethodEntry<ContinuityMethod>[] = [
  {
    value: 'ROOM_TONE',
    label: 'Room Tone',
    description: 'Natural micro-pause at edit boundaries',
    badge: 'Recommended',
    badgeColor: 'emerald',
    icon: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z',
  },
  {
    value: 'CROSSFADE',
    label: 'Crossfade',
    description: 'Smooth equal-power audio blend',
    badge: 'Fast',
    badgeColor: 'sky',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z',
  },
  {
    value: 'SPECTRAL',
    label: 'Spectral',
    description: 'FFT phase-coherent audio blending',
    badge: 'HQ',
    badgeColor: 'violet',
    icon: 'M2 12h2l3-7 4 14 4-10 3 6h4',
  },
  {
    value: 'SMART_SILENCE',
    label: 'Smart Silence',
    description: 'Cut at lowest-energy point in boundary region',
    badge: 'Clean',
    badgeColor: 'teal',
    icon: 'M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zM19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8',
  },
  {
    value: 'AI_REGENERATION',
    label: 'Voice Clone Bridge',
    description: 'F5-TTS voice cloning fills the edit gap',
    badge: 'AI',
    badgeColor: 'rose',
    icon: 'M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zM19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8',
  },
  {
    value: 'BOUNDARY_CLONE',
    label: 'Boundary Clone',
    description: 'Synthesise boundary words; crossfade within word',
    badge: 'New',
    badgeColor: 'indigo',
    icon: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z',
  },
  {
    value: 'J_CUT',
    label: 'J-Cut',
    description: 'Next clip audio starts before the video transition',
    badge: 'Cut',
    badgeColor: 'amber',
    icon: 'M9 18V5l12-2v13M9 18c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z',
  },
  {
    value: 'L_CUT',
    label: 'L-Cut',
    description: 'Previous clip audio continues after the video cut',
    badge: 'Cut',
    badgeColor: 'amber',
    icon: 'M3 18V5l12 2v13M3 18c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z',
  },
  {
    value: 'NONE',
    label: 'None (Raw)',
    description: 'No audio processing — hard cut',
    badge: 'Raw',
    badgeColor: 'slate',
    icon: 'M18 6L6 18M6 6l12 12',
  },
];

// Hidden: WSOLA, ENCODEC, AI_INPAINTING, DEMUCS, NEURAL_SMOOTH, SADTALKER_CUT

// ── SMART_CUT — combined A/V method shown in the video panel ──────────────
const SMART_CUT_ENTRY: MethodEntry<ContinuityMethod> = {
  value: 'SMART_CUT',
  label: 'FOMM Smart Cut',
  description: 'Neural face keypoint morph + speech crossfade',
  badge: 'Neural',
  badgeColor: 'fuchsia',
  icon: 'M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3zM19 10v2a7 7 0 0 1-14 0v-2M12 19v4M8 23h8',
};

// ── Visible video methods ──────────────────────────────────────────────────
const VIDEO_METHODS: MethodEntry<VideoInterpolationMethod>[] = [
  {
    value: 'NONE',
    label: 'None (Hard Cut)',
    description: 'No visual transition — direct splice',
    badge: 'Default',
    badgeColor: 'slate',
    icon: 'M18 6L6 18M6 6l12 12',
  },
  {
    value: 'FRAME_PAD',
    label: 'Frame Pad',
    description: 'Freeze boundary frames + dissolve transition',
    badge: 'New',
    badgeColor: 'sky',
    icon: 'M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.89L15 14M3 8h12v8H3z',
  },
  {
    value: 'BLEND',
    label: 'Motion Blend',
    description: 'ffmpeg motion-compensated interpolation',
    badge: 'Fast',
    badgeColor: 'sky',
    icon: 'M15 10l4.553-2.069A1 1 0 0121 8.82V15.18a1 1 0 01-1.447.89L15 14M3 8h12v8H3z',
  },
  {
    value: 'RIFE',
    label: 'RIFE Neural',
    description: 'Neural optical flow frame synthesis',
    badge: 'AI',
    badgeColor: 'violet',
    icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
  },
  {
    value: 'FADE_TO_BLACK',
    label: 'Fade to Black',
    description: 'Dip to black frame at each edit boundary',
    badge: 'FX',
    badgeColor: 'slate',
    icon: 'M12 3a9 9 0 100 18A9 9 0 0012 3z',
  },
];

// ── Badge / border helpers ─────────────────────────────────────────────────
const BADGE_CLASSES: Record<string, string> = {
  rose: 'bg-rose-500/15 text-rose-400 ring-1 ring-rose-500/20',
  emerald: 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/20',
  sky: 'bg-sky-500/15 text-sky-400 ring-1 ring-sky-500/20',
  violet: 'bg-violet-500/15 text-violet-400 ring-1 ring-violet-500/20',
  amber: 'bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/20',
  slate: 'bg-slate-500/15 text-slate-400 ring-1 ring-slate-500/20',
  fuchsia: 'bg-fuchsia-500/15 text-fuchsia-400 ring-1 ring-fuchsia-500/20',
  teal: 'bg-teal-500/15 text-teal-400 ring-1 ring-teal-500/20',
  indigo: 'bg-indigo-500/15 text-indigo-400 ring-1 ring-indigo-500/20',
};

const ACTIVE_BORDER: Record<string, string> = {
  rose: 'border-rose-500/40',
  emerald: 'border-emerald-500/40',
  sky: 'border-sky-500/40',
  violet: 'border-violet-500/40',
  amber: 'border-amber-500/40',
  slate: 'border-slate-500/40',
  fuchsia: 'border-fuchsia-500/40',
  teal: 'border-teal-500/40',
  indigo: 'border-indigo-500/40',
};

// ── Shared row renderers ───────────────────────────────────────────────────

function MethodRow({
  label, description, badge, badgeColor, icon, isActive, onClick,
}: {
  label: string; description: string; badge: string; badgeColor: string; icon: string;
  isActive: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-2.5 py-2 rounded-lg transition-all duration-100 border ${
        isActive
          ? `bg-slate-700/50 ${ACTIVE_BORDER[badgeColor]}`
          : 'border-transparent hover:bg-slate-700/30'
      }`}
    >
      <div className="flex items-center gap-2">
        <div className={`flex-shrink-0 w-6 h-6 rounded-md flex items-center justify-center ${
          isActive ? 'bg-slate-600/60' : 'bg-slate-700/40'
        }`}>
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
            stroke={isActive ? 'currentColor' : '#94a3b8'}
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            className={isActive ? 'text-white' : ''}
          >
            <path d={icon} />
          </svg>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className={`text-xs font-medium ${isActive ? 'text-white' : 'text-slate-300'}`}>
              {label}
            </span>
            <span className={`text-[9px] font-medium px-1.5 py-px rounded-full ${BADGE_CLASSES[badgeColor]}`}>
              {badge}
            </span>
          </div>
          <p className={`text-[10px] mt-0.5 ${isActive ? 'text-slate-400' : 'text-slate-500'}`}>
            {description}
          </p>
        </div>
        {isActive && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-emerald-400 flex-shrink-0">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
      </div>
    </button>
  );
}

// ── Component ──────────────────────────────────────────────────────────────

export default function ContinuitySelector({
  audioContinuityMethod,
  onAudioChange,
  videoInterpolationMethod,
  onVideoChange,
  hasVideo = false,
  smartEditEnabled,
  onSmartEditChange,
}: ContinuitySelectorProps) {
  const [openPanel, setOpenPanel] = useState<'audio' | 'video' | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!openPanel) return;
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpenPanel(null);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [openPanel]);

  const isSmartCutActive = audioContinuityMethod === 'SMART_CUT' && !smartEditEnabled;
  const activeAudio = AUDIO_METHODS.find(m => m.value === audioContinuityMethod);
  const activeVideo = VIDEO_METHODS.find(m => m.value === videoInterpolationMethod);

  const audioButtonLabel = smartEditEnabled
    ? 'Smart Edit'
    : activeAudio?.label ?? SMART_CUT_ENTRY.label;

  const videoButtonLabel = isSmartCutActive
    ? SMART_CUT_ENTRY.label
    : (activeVideo?.label ?? 'None');

  return (
    <div className="relative flex items-center gap-1" ref={containerRef}>

      {/* Smart Edit toggle */}
      <button
        onClick={() => onSmartEditChange(!smartEditEnabled)}
        className={`flex items-center gap-1.5 text-xs px-2 py-1.5 rounded-md transition-all duration-150 border ${
          smartEditEnabled
            ? 'bg-gradient-to-r from-violet-600/30 to-emerald-600/30 border-violet-500/40 text-white'
            : 'border-transparent text-slate-400 hover:text-white hover:bg-slate-700/60'
        }`}
        title="Smart Edit: room tone + neural face blend (best quality)"
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3l1.5 4.5L18 9l-4.5 1.5L12 15l-1.5-4.5L6 9l4.5-1.5L12 3z" />
          <path d="M5 3l.75 2.25L8 6l-2.25.75L5 9l-.75-2.25L2 6l2.25-.75L5 3z" />
          <path d="M19 15l.75 2.25L22 18l-2.25.75L19 21l-.75-2.25L16 18l2.25-.75L19 15z" />
        </svg>
        <span className="hidden sm:inline font-medium">Smart Edit</span>
        {smartEditEnabled && (
          <span className="text-[9px] font-medium px-1.5 py-px rounded-full bg-violet-500/20 text-violet-300 ring-1 ring-violet-500/30">
            ON
          </span>
        )}
      </button>

      {/* ── Audio dropdown ── */}
      <div className="relative">
        <button
          onClick={() => setOpenPanel(openPanel === 'audio' ? null : 'audio')}
          className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2 py-1.5 rounded-md hover:bg-slate-700/60 transition-all duration-150"
          title="Audio continuity method"
        >
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d={activeAudio?.icon ?? AUDIO_METHODS[0].icon} />
          </svg>
          <span className="hidden sm:inline">{audioButtonLabel}</span>
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
            className={`transition-transform duration-150 ${openPanel === 'audio' ? 'rotate-180' : ''}`}
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>

        {openPanel === 'audio' && (
          <div className="absolute top-full right-0 mt-1.5 bg-slate-800/95 backdrop-blur-xl rounded-xl shadow-2xl shadow-black/40 border border-slate-700/60 overflow-hidden z-20 w-64 max-h-[80vh] overflow-y-auto">
            <div className="px-3 py-2 border-b border-slate-700/50">
              <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Audio Continuity</span>
            </div>
            <div className={`transition-opacity duration-150 ${smartEditEnabled ? 'opacity-40 pointer-events-none' : ''}`}>
              <div className="p-1.5 space-y-0.5">
                {AUDIO_METHODS.map(m => (
                  <MethodRow
                    key={m.value}
                    label={m.label}
                    description={m.description}
                    badge={m.badge}
                    badgeColor={m.badgeColor}
                    icon={m.icon}
                    isActive={m.value === audioContinuityMethod}
                    onClick={() => { onAudioChange(m.value); setOpenPanel(null); }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── Video dropdown (only when hasVideo) ── */}
      {hasVideo && (
        <div className="relative">
          <button
            onClick={() => setOpenPanel(openPanel === 'video' ? null : 'video')}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2 py-1.5 rounded-md hover:bg-slate-700/60 transition-all duration-150"
            title="Video continuity method"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="7" width="20" height="15" rx="2" ry="2" />
              <polyline points="17 2 12 7 7 2" />
            </svg>
            <span className="hidden sm:inline">{videoButtonLabel}</span>
            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"
              className={`transition-transform duration-150 ${openPanel === 'video' ? 'rotate-180' : ''}`}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>

          {openPanel === 'video' && (
            <div className="absolute top-full right-0 mt-1.5 bg-slate-800/95 backdrop-blur-xl rounded-xl shadow-2xl shadow-black/40 border border-slate-700/60 overflow-hidden z-20 w-64 max-h-[80vh] overflow-y-auto">
              <div className="px-3 py-2 border-b border-slate-700/50">
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">Video Continuity</span>
              </div>
              <div className={`transition-opacity duration-150 ${smartEditEnabled ? 'opacity-40 pointer-events-none' : ''}`}>
                {/* Combined A/V method */}
                <div className="p-1.5 pb-0 space-y-0.5">
                  <MethodRow
                    label={SMART_CUT_ENTRY.label}
                    description={SMART_CUT_ENTRY.description}
                    badge={SMART_CUT_ENTRY.badge}
                    badgeColor={SMART_CUT_ENTRY.badgeColor}
                    icon={SMART_CUT_ENTRY.icon}
                    isActive={isSmartCutActive}
                    onClick={() => {
                      onAudioChange('SMART_CUT');
                      onVideoChange('NONE');
                      setOpenPanel(null);
                    }}
                  />
                </div>

                {/* Frame interpolation methods */}
                <div className="px-3 py-1.5 border-t border-slate-700/50 mt-1">
                  <span className="text-[9px] font-semibold text-slate-600 uppercase tracking-widest">Frame Interpolation</span>
                </div>
                <div className="p-1.5 space-y-0.5">
                  {VIDEO_METHODS.map(m => (
                    <MethodRow
                      key={m.value}
                      label={m.label}
                      description={m.description}
                      badge={m.badge}
                      badgeColor={m.badgeColor}
                      icon={m.icon}
                      isActive={m.value === videoInterpolationMethod && !isSmartCutActive}
                      onClick={() => {
                        if (audioContinuityMethod === 'SMART_CUT') onAudioChange('ROOM_TONE');
                        onVideoChange(m.value);
                        setOpenPanel(null);
                      }}
                    />
                  ))}
                </div>

              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
}
