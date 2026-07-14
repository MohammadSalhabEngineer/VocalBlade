import { useCallback, useRef, useState } from 'react';

interface AudioUploadProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

const ALLOWED_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.mp4', '.mpga', '.mpeg', '.webm'];

export default function AudioUpload({ onFileSelected, disabled }: AudioUploadProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        alert(`Unsupported format: ${ext}\nAllowed: ${ALLOWED_EXTENSIONS.join(', ')}`);
        return;
      }
      if (file.size > 100 * 1024 * 1024) {
        alert('File size exceeds 100MB limit');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${
        dragOver ? 'border-blue-400 bg-blue-400/10' : 'border-slate-600 hover:border-slate-400'
      } ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_EXTENSIONS.join(',')}
        className="hidden"
        onChange={(e) => {
          if (e.target.files?.[0]) handleFile(e.target.files[0]);
        }}
      />
      <div className="text-4xl mb-3">🎙️</div>
      <p className="text-lg font-medium">Drop audio/video file here or click to browse</p>
      <p className="text-sm text-slate-400 mt-2">
        Supports: WAV, MP3, M4A, MP4, WebM (max 100MB)
      </p>
    </div>
  );
}
