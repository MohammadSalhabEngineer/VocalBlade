import type { TranscriptionSegment } from '../api/graphql';

// --- RTL Detection ---

const RTL_LANGUAGES = new Set(['ar', 'he', 'fa', 'ur', 'ps', 'sd', 'yi']);

export function getTextDirection(language: string | null | undefined): 'ltr' | 'rtl' {
  if (!language) return 'ltr';
  return RTL_LANGUAGES.has(language.toLowerCase()) ? 'rtl' : 'ltr';
}

// --- Word ID helpers ---

export type WordId = string; // "segIdx-wordIdx"

export function makeWordId(segIdx: number, wordIdx: number): WordId {
  return `${segIdx}-${wordIdx}`;
}

export function parseWordId(id: WordId): { segIdx: number; wordIdx: number } {
  const [s, w] = id.split('-').map(Number);
  return { segIdx: s, wordIdx: w };
}

// --- Edit operations ---

export type EditOperation =
  | { type: 'delete'; wordIds: WordId[] }
  | { type: 'move'; wordIds: WordId[]; targetWordId: WordId }
  | { type: 'paste'; wordIds: WordId[]; insertAfter: WordId | null; pasteId: number }
  | { type: 'delete_range'; start: number; end: number }; // absolute time range (silence removal)

export function isWordDeleted(wordId: WordId, editOps: EditOperation[]): boolean {
  for (const op of editOps) {
    if (op.type === 'delete' && op.wordIds.includes(wordId)) {
      return true;
    }
  }
  return false;
}

// --- Effective words (flat list with original timing, after edits) ---

export interface EffectiveWord {
  word: string;
  start: number;
  end: number;
  score: number | null;
  speaker: string | null;
  wordId: WordId;
  deleted: boolean;
  originalIndex: number; // position in pre-edit word list (-1 for pasted clones)
}

export function buildFlatWordList(segments: TranscriptionSegment[]): EffectiveWord[] {
  const words: EffectiveWord[] = [];
  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    if (seg.words && seg.words.length > 0) {
      for (let wi = 0; wi < seg.words.length; wi++) {
        const w = seg.words[wi];
        words.push({
          word: w.word,
          start: w.start,
          end: w.end,
          score: w.score,
          speaker: seg.speaker,
          wordId: makeWordId(si, wi),
          deleted: false,
          originalIndex: words.length,
        });
      }
    } else {
      // Segment-level: treat the whole segment as one "word"
      words.push({
        word: seg.text,
        start: seg.start,
        end: seg.end,
        score: seg.score,
        speaker: seg.speaker,
        wordId: makeWordId(si, 0),
        deleted: false,
        originalIndex: words.length,
      });
    }
  }
  return words;
}

export function applyEdits(flatWords: EffectiveWord[], editOps: EditOperation[]): EffectiveWord[] {
  let result = flatWords.map((w) => ({ ...w }));

  for (const op of editOps) {
    if (op.type === 'delete') {
      const deleteSet = new Set(op.wordIds);
      result = result.map((w) => ({
        ...w,
        deleted: w.deleted || deleteSet.has(w.wordId),
      }));
    } else if (op.type === 'move') {
      const moveSet = new Set(op.wordIds);
      const movedWords = result.filter((w) => moveSet.has(w.wordId));
      const remaining = result.filter((w) => !moveSet.has(w.wordId));
      const targetIdx = remaining.findIndex((w) => w.wordId === op.targetWordId);
      if (targetIdx !== -1) {
        remaining.splice(targetIdx, 0, ...movedWords);
      }
      result = remaining;
    } else if (op.type === 'delete_range') {
      // Mark any word whose midpoint falls within the time range as deleted
      result = result.map((w) => {
        const mid = (w.start + w.end) / 2;
        return { ...w, deleted: w.deleted || (mid >= op.start && mid <= op.end) };
      });
    } else if (op.type === 'paste') {
      // Clone referenced words and insert after anchor
      const clones: EffectiveWord[] = [];
      for (let ci = 0; ci < op.wordIds.length; ci++) {
        const src = result.find((w) => w.wordId === op.wordIds[ci]);
        if (src) {
          clones.push({
            ...src,
            wordId: `paste-${op.pasteId}-${ci}` as WordId,
            deleted: false,
            originalIndex: -1,
          });
        }
      }
      if (clones.length > 0) {
        const insertIdx = op.insertAfter
          ? result.findIndex((w) => w.wordId === op.insertAfter)
          : -1;
        const position = insertIdx >= 0 ? insertIdx + 1 : result.length;
        result.splice(position, 0, ...clones);
      }
    }
  }

  return result;
}

// --- Time ranges ---

export interface TimeRange {
  start: number;
  end: number;
}

// --- Playback schedule for live preview (handles deletes AND moves) ---

/**
 * Build an ordered list of time ranges from the effective word list.
 * Words already have deletes applied and moves reordered by applyEdits().
 * Adjacent words in the original timeline are merged; gaps or reorderings
 * produce separate ranges that the player follows in sequence.
 */
export function computePlaybackSchedule(flatWords: EffectiveWord[]): TimeRange[] {
  if (flatWords.length === 0) return [];

  const ranges: TimeRange[] = [];
  let cur: TimeRange | null = null;
  let lastOrigIdx = -1;

  for (const w of flatWords) {
    if (w.deleted) {
      // Deleted word → end current range (player will seek past this gap)
      if (cur) {
        ranges.push(cur);
        cur = null;
      }
      continue;
    }

    // Kept word — extend current range only if forward in both time AND original order
    if (cur === null) {
      cur = { start: w.start, end: w.end };
    } else if (
      w.start >= cur.end - 0.01 &&
      w.originalIndex === lastOrigIdx + 1
    ) {
      // Forward in time AND strictly next in original order → extend range
      cur.end = Math.max(cur.end, w.end);
    } else {
      // Backward in time or original order (moved/pasted word) → new range
      ranges.push(cur);
      cur = { start: w.start, end: w.end };
    }

    lastOrigIdx = w.originalIndex;
  }

  if (cur) ranges.push(cur);
  return ranges;
}

// --- Keep ranges for export ---

export function computeKeepRanges(
  segments: TranscriptionSegment[],
  editOps: EditOperation[],
  fileDuration?: number,
): TimeRange[] {
  const flat = buildFlatWordList(segments);
  const effective = applyEdits(flat, editOps);

  const kept = effective.filter((w) => !w.deleted);
  if (kept.length === 0) return [];

  // No edits → empty array signals backend to copy full file as-is
  if (editOps.length === 0) {
    return [];
  }

  // No deletions, moves, pastes, or delete_ranges → single range
  const hasAnyDeletion = effective.some((w) => w.deleted);
  const hasAnyMove = editOps.some((op) => op.type === 'move');
  const hasAnyPaste = editOps.some((op) => op.type === 'paste');
  const hasAnyRangeDeletion = editOps.some((op) => op.type === 'delete_range');
  if (!hasAnyDeletion && !hasAnyMove && !hasAnyPaste && !hasAnyRangeDeletion) {
    return [{ start: kept[0].start, end: kept[kept.length - 1].end }];
  }

  // Build ranges using originalIndex — same logic as playback schedule:
  // only merge strictly consecutive original words, break at deletions/moves
  //
  // Smart expansion: expand boundaries into silence gaps between kept/deleted words
  // to protect phonetic tails (consonant releases) while never crossing into deleted audio.
  const TAIL_PROTECTION = 0.04;   // 40ms minimum safety margin for phonetic tails
  const MAX_GAP_EXPANSION = 0.15; // 150ms maximum expansion into silence gaps
  const DELETED_BOUNDARY_SAFETY = 0.05; // 50ms buffer: 30ms for backend snapping + 20ms for WhisperX inaccuracy

  // Track the effective-list index of each range's first and last kept word
  interface RangeInfo {
    range: TimeRange;
    firstKeptIdx: number; // index in `effective` array
    lastKeptIdx: number;
  }

  const rangeInfos: RangeInfo[] = [];
  let cur: TimeRange | null = null;
  let curFirstIdx = -1;
  let curLastIdx = -1;
  let lastOrigIdx = -1;

  for (let ei = 0; ei < effective.length; ei++) {
    const w = effective[ei];
    if (w.deleted) {
      if (cur) {
        rangeInfos.push({ range: { ...cur }, firstKeptIdx: curFirstIdx, lastKeptIdx: curLastIdx });
        cur = null;
      }
      continue;
    }

    if (cur === null) {
      cur = { start: w.start, end: w.end };
      curFirstIdx = ei;
      curLastIdx = ei;
    } else if (
      w.start >= cur.end - 0.01 &&
      w.originalIndex === lastOrigIdx + 1
    ) {
      cur.end = Math.max(cur.end, w.end);
      curLastIdx = ei;
    } else {
      rangeInfos.push({ range: { ...cur }, firstKeptIdx: curFirstIdx, lastKeptIdx: curLastIdx });
      cur = { start: w.start, end: w.end };
      curFirstIdx = ei;
      curLastIdx = ei;
    }

    lastOrigIdx = w.originalIndex;
  }
  if (cur) rangeInfos.push({ range: { ...cur }, firstKeptIdx: curFirstIdx, lastKeptIdx: curLastIdx });

  // Smart expansion pass: expand range boundaries into silence gaps
  for (let ri = 0; ri < rangeInfos.length; ri++) {
    const info = rangeInfos[ri];

    // --- Expand range.end (look for deleted word AFTER last kept word) ---
    // Don't expand the very last range's end if the last word in effective list is kept
    const isLastRange = ri === rangeInfos.length - 1;
    const lastWordKeptGlobal = effective.length > 0 && !effective[effective.length - 1].deleted;
    if (!(isLastRange && lastWordKeptGlobal)) {
      // Find first deleted word after this range's last kept word
      let nextDeletedIdx = -1;
      for (let j = info.lastKeptIdx + 1; j < effective.length; j++) {
        if (effective[j].deleted) {
          nextDeletedIdx = j;
          break;
        }
        // Hit a non-deleted word that's not in this range — stop
        break;
      }
      if (nextDeletedIdx !== -1) {
        const deletedWord = effective[nextDeletedIdx];
        const lastKeptWord = effective[info.lastKeptIdx];
        const gap = deletedWord.start - lastKeptWord.end;
        if (gap > 0) {
          // Expand into the silence gap, capped at MAX_GAP_EXPANSION
          const expansion = Math.min(Math.max(TAIL_PROTECTION, gap), MAX_GAP_EXPANSION);
          // Never cross into deleted word audio — stop 50ms before to account for
          // backend snapping overshoot (30ms) + WhisperX timestamp inaccuracy (20ms)
          info.range.end = Math.min(lastKeptWord.end + expansion, deletedWord.start - DELETED_BOUNDARY_SAFETY);
          // But never shrink below the original word.end
          info.range.end = Math.max(info.range.end, lastKeptWord.end);
        }
        // gap <= 0: timestamps overlap, don't expand
      } else {
        // No deleted word found after — just add tail protection
        info.range.end += TAIL_PROTECTION;
      }
    }

    // --- Expand range.start (look for deleted word BEFORE first kept word) ---
    // Don't expand the very first range's start if the first word in effective list is kept
    const isFirstRange = ri === 0;
    const firstWordKeptGlobal = effective.length > 0 && !effective[0].deleted;
    if (!(isFirstRange && firstWordKeptGlobal)) {
      // Find last deleted word before this range's first kept word
      let prevDeletedIdx = -1;
      for (let j = info.firstKeptIdx - 1; j >= 0; j--) {
        if (effective[j].deleted) {
          prevDeletedIdx = j;
          break;
        }
        // Hit a non-deleted word that's not in this range — stop
        break;
      }
      if (prevDeletedIdx !== -1) {
        const deletedWord = effective[prevDeletedIdx];
        const firstKeptWord = effective[info.firstKeptIdx];
        const gap = firstKeptWord.start - deletedWord.end;
        if (gap > 0) {
          const expansion = Math.min(Math.max(TAIL_PROTECTION, gap), MAX_GAP_EXPANSION);
          // Never cross into deleted word audio — stop 50ms after to account for
          // backend snapping overshoot (30ms) + WhisperX timestamp inaccuracy (20ms)
          info.range.start = Math.max(firstKeptWord.start - expansion, deletedWord.end + DELETED_BOUNDARY_SAFETY);
          // But never expand past the original word.start
          info.range.start = Math.min(info.range.start, firstKeptWord.start);
        }
        // gap <= 0: timestamps overlap, don't expand
      } else {
        // No deleted word found before — just add tail protection
        info.range.start = Math.max(0, info.range.start - TAIL_PROTECTION);
      }
    }
  }

  // Apply file-boundary extensions for first/last ranges
  const firstWordKept = effective.length > 0 && !effective[0].deleted;
  const lastWordKept = effective.length > 0 && !effective[effective.length - 1].deleted;

  let wordRanges = rangeInfos.map((info, i) => ({
    start: i === 0 && firstWordKept ? 0 : Math.max(0, info.range.start),
    end: i === rangeInfos.length - 1 && fileDuration && lastWordKept ? fileDuration : info.range.end,
  }));

  // Subtract delete_range intervals (silence removal) from the word-based ranges
  const deleteRanges = editOps
    .filter((op): op is { type: 'delete_range'; start: number; end: number } => op.type === 'delete_range')
    .sort((a, b) => a.start - b.start);

  if (deleteRanges.length > 0) {
    wordRanges = subtractRanges(wordRanges, deleteRanges);
  }

  return wordRanges;
}

function subtractRanges(
  source: TimeRange[],
  toRemove: TimeRange[],
): TimeRange[] {
  let result: TimeRange[] = [...source];
  for (const rem of toRemove) {
    const next: TimeRange[] = [];
    for (const r of result) {
      if (rem.end <= r.start || rem.start >= r.end) {
        next.push(r); // no overlap
      } else {
        if (rem.start > r.start) next.push({ start: r.start, end: rem.start });
        if (rem.end < r.end) next.push({ start: rem.end, end: r.end });
      }
    }
    result = next;
  }
  return result.filter((r) => r.end - r.start > 0.05); // drop sub-50ms fragments
}

// --- Active word detection (binary search) ---

export function findActiveWordIndex(flatWords: EffectiveWord[], currentTime: number): number {
  if (flatWords.length === 0) return -1;

  // Linear scan is fine for typical transcript sizes (< 5000 words)
  // and avoids edge cases with overlapping/unsorted timings
  for (let i = 0; i < flatWords.length; i++) {
    const w = flatWords[i];
    if (!w.deleted && currentTime >= w.start && currentTime <= w.end) {
      return i;
    }
  }
  return -1;
}

// --- Word range selection helper ---

export function getWordIdsInRange(
  flatWords: EffectiveWord[],
  fromId: WordId,
  toId: WordId
): WordId[] {
  const fromIdx = flatWords.findIndex((w) => w.wordId === fromId);
  const toIdx = flatWords.findIndex((w) => w.wordId === toId);
  if (fromIdx === -1 || toIdx === -1) return [];

  const start = Math.min(fromIdx, toIdx);
  const end = Math.max(fromIdx, toIdx);
  return flatWords.slice(start, end + 1).map((w) => w.wordId);
}

// --- Collect text of selected words ---

export function getSelectedText(
  flatWords: EffectiveWord[],
  selectedIds: Set<WordId>
): string {
  return flatWords
    .filter((w) => selectedIds.has(w.wordId) && !w.deleted)
    .map((w) => w.word)
    .join(' ');
}

// --- AI utilities ---

/**
 * Find silence gaps between word timings across all segments.
 * Returns time ranges where no one is speaking for longer than thresholdSec.
 */
export function findSilenceGaps(
  segments: TranscriptionSegment[],
  thresholdSec: number,
): TimeRange[] {
  // Collect all word-level timings in chronological order
  const wordTimes: { start: number; end: number }[] = [];
  for (const seg of segments) {
    if (seg.words && seg.words.length > 0) {
      for (const w of seg.words) {
        wordTimes.push({ start: w.start, end: w.end });
      }
    } else {
      wordTimes.push({ start: seg.start, end: seg.end });
    }
  }
  wordTimes.sort((a, b) => a.start - b.start);

  const gaps: TimeRange[] = [];
  for (let i = 1; i < wordTimes.length; i++) {
    const gapStart = wordTimes[i - 1].end;
    const gapEnd = wordTimes[i].start;
    if (gapEnd - gapStart >= thresholdSec) {
      gaps.push({ start: gapStart, end: gapEnd });
    }
  }
  return gaps;
}

/**
 * Format transcript segments into a compact string for the Qwen LLM prompt.
 * Keeps [segIdx-wordIdx:speaker] word format matching the backend's expected IDs.
 */
export function buildTranscriptContext(
  segments: TranscriptionSegment[],
  speakerNames: Record<string, string>,
  maxChars = 12000,
): string {
  const lines: string[] = [];

  if (Object.keys(speakerNames).length > 0) {
    const map = Object.entries(speakerNames).map(([k, v]) => `${k}=${v}`).join(', ');
    lines.push(`[Speakers: ${map}]`);
  }

  let totalSec = 0;
  for (const seg of segments) {
    totalSec = Math.max(totalSec, seg.end);
    if (seg.words) for (const w of seg.words) totalSec = Math.max(totalSec, w.end);
  }
  const mins = Math.floor(totalSec / 60);
  const secs = Math.floor(totalSec % 60);
  lines.push(`[Duration: ${mins}m ${secs.toString().padStart(2, '0')}s]`);
  lines.push('');

  for (let si = 0; si < segments.length; si++) {
    const seg = segments[si];
    const spk = seg.speaker || 'SPEAKER';
    const spkLabel = speakerNames[spk] || spk;
    if (seg.words && seg.words.length > 0) {
      for (let wi = 0; wi < seg.words.length; wi++) {
        const w = seg.words[wi];
        const mm = Math.floor(w.start / 60).toString().padStart(2, '0');
        const ss = Math.floor(w.start % 60).toString().padStart(2, '0');
        lines.push(`${mm}:${ss} [${si}-${wi}:${spkLabel}] ${w.word.trim()}`);
      }
    } else {
      const mm = Math.floor(seg.start / 60).toString().padStart(2, '0');
      const ss = Math.floor(seg.start % 60).toString().padStart(2, '0');
      lines.push(`${mm}:${ss} [${si}:${spkLabel}] ${seg.text.trim()}`);
    }
  }

  let ctx = lines.join('\n');
  if (ctx.length > maxChars) ctx = ctx.slice(0, maxChars) + '\n[...truncated...]';
  return ctx;
}
