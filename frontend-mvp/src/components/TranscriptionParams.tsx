import { useEffect, useState } from 'react';
import { fetchAvailableModels, fetchSupportedLanguages, type TranscriptionParams } from '../api/graphql';

interface TranscriptionParamsProps {
  params: TranscriptionParams;
  onChange: (params: TranscriptionParams) => void;
  disabled?: boolean;
}

export default function TranscriptionParamsPanel({ params, onChange, disabled }: TranscriptionParamsProps) {
  const [models, setModels] = useState<string[]>([]);
  const [languages, setLanguages] = useState<string[]>([]);

  useEffect(() => {
    const load = (attempt = 0) => {
      fetchAvailableModels().then(setModels).catch(() => {
        if (attempt < 5) setTimeout(() => load(attempt + 1), 2000);
      });
      fetchSupportedLanguages().then(setLanguages).catch(() => {
        if (attempt < 5) setTimeout(() => load(attempt + 1), 2000);
      });
    };
    load();
  }, []);

  return (
    <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Model</label>
        <select
          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          value={params.model || 'BASE'}
          onChange={(e) => onChange({ ...params, model: e.target.value })}
        >
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-300 mb-1">Language</label>
        <select
          className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
          value={params.language || ''}
          onChange={(e) => onChange({ ...params, language: e.target.value || null })}
        >
          <option value="">Auto-detect</option>
          {languages.map((l) => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>
      </div>

      <div className="flex items-end gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={params.enableDiarization ?? true}
            onChange={(e) => onChange({ ...params, enableDiarization: e.target.checked })}
            className="rounded"
          />
          Speaker ID
        </label>
      </div>

      <div className="flex items-end gap-4">
        <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={params.returnWordTimings ?? false}
            onChange={(e) => onChange({ ...params, returnWordTimings: e.target.checked })}
            className="rounded"
          />
          Word Timings
        </label>
      </div>
    </div>
  );
}
