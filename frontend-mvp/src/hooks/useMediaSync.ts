import { useRef, useCallback, useEffect } from 'react';

interface UseMediaSyncOptions {
  onTimeUpdate: (time: number) => void;
  onPlayStateChange: (playing: boolean) => void;
}

export function useMediaSync({ onTimeUpdate, onPlayStateChange }: UseMediaSyncOptions) {
  const mediaRef = useRef<HTMLVideoElement | HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const lastDispatchRef = useRef<number>(0);

  // Store callbacks in refs so setMediaElement has a STABLE identity
  const onTimeUpdateRef = useRef(onTimeUpdate);
  const onPlayStateChangeRef = useRef(onPlayStateChange);
  useEffect(() => { onTimeUpdateRef.current = onTimeUpdate; }, [onTimeUpdate]);
  useEffect(() => { onPlayStateChangeRef.current = onPlayStateChange; }, [onPlayStateChange]);

  const stopTimeLoop = useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    }
  }, []);

  const startTimeLoop = useCallback(() => {
    stopTimeLoop();
    const loop = () => {
      const el = mediaRef.current;
      if (!el) return;
      const now = performance.now();
      if (now - lastDispatchRef.current >= 50) {
        onTimeUpdateRef.current(el.currentTime);
        lastDispatchRef.current = now;
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [stopTimeLoop]);

  // Cleanup stored as ref — not on the DOM element
  const cleanupRef = useRef<(() => void) | null>(null);

  // STABLE: only depends on startTimeLoop/stopTimeLoop which are also stable
  const setMediaElement = useCallback(
    (el: HTMLVideoElement | HTMLAudioElement | null) => {
      // Clean up previous element
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      stopTimeLoop();

      mediaRef.current = el;
      if (!el) return;

      const handlePlay = () => {
        onPlayStateChangeRef.current(true);
        startTimeLoop();
      };
      const handlePause = () => {
        onPlayStateChangeRef.current(false);
        stopTimeLoop();
        onTimeUpdateRef.current(el.currentTime);
      };
      const handleEnded = () => {
        onPlayStateChangeRef.current(false);
        stopTimeLoop();
      };
      const handleSeeked = () => {
        onTimeUpdateRef.current(el.currentTime);
      };

      el.addEventListener('play', handlePlay);
      el.addEventListener('pause', handlePause);
      el.addEventListener('ended', handleEnded);
      el.addEventListener('seeked', handleSeeked);

      cleanupRef.current = () => {
        el.removeEventListener('play', handlePlay);
        el.removeEventListener('pause', handlePause);
        el.removeEventListener('ended', handleEnded);
        el.removeEventListener('seeked', handleSeeked);
        stopTimeLoop();
      };
    },
    [startTimeLoop, stopTimeLoop]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (cleanupRef.current) cleanupRef.current();
      stopTimeLoop();
    };
  }, [stopTimeLoop]);

  const seekTo = useCallback((time: number) => {
    const el = mediaRef.current;
    if (el) el.currentTime = time;
  }, []);

  const togglePlayPause = useCallback(() => {
    const el = mediaRef.current;
    if (!el) return;
    if (el.paused) el.play(); else el.pause();
  }, []);

  const setVolume = useCallback((volume: number) => {
    const el = mediaRef.current;
    if (el) el.volume = Math.max(0, Math.min(1, volume));
  }, []);

  const setPlaybackRate = useCallback((rate: number) => {
    const el = mediaRef.current;
    if (el) el.playbackRate = rate;
  }, []);

  return {
    mediaRef,
    setMediaElement,
    seekTo,
    togglePlayPause,
    setVolume,
    setPlaybackRate,
  };
}
