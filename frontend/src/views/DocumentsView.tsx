import React, { useCallback, useRef, useState } from 'react';
import { useDocuments } from '../hooks/useDocuments';

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

function formatDate(iso: string): string {
  try {
    return new Intl.DateTimeFormat('en-IN', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

const STAGE_LABELS: Record<string, string> = {
  queued:    'Queued',
  parsing:   'Parsing PDF…',
  chunking:  'Chunking…',
  embedding: 'Embedding…',
  indexing:  'Indexing…',
  complete:  'Complete',
  failed:    'Failed',
};

export function DocumentsView() {
  const { documents, jobs, loading, error, uploadDocument, deleteDocument, refresh } =
    useDocuments();

  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0]; // one at a time
      if (file.type !== 'application/pdf' && !file.name.endsWith('.pdf')) {
        setUploadError('Only PDF files are supported.');
        return;
      }
      setUploadError(null);
      setUploading(true);
      const ok = await uploadDocument(file);
      setUploading(false);
      if (!ok) setUploadError('Upload failed. Check the error above.');
    },
    [uploadDocument],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles],
  );

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(true);
  };

  const handleDelete = async (docId: string, filename: string) => {
    if (!window.confirm(`Delete "${filename}"? This will remove all indexed vectors.`)) return;
    try {
      await deleteDocument(docId);
    } catch {
      // already handled in hook
    }
  };

  const activeJobs = Array.from(jobs.values());

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Upload Zone */}
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setDragging(false)}
        onClick={() => fileInputRef.current?.click()}
        id="doc-dropzone"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          style={{ display: 'none' }}
          id="doc-file-input"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {uploading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
            <span className="spinner" />
            <div className="dropzone-title">Uploading…</div>
          </div>
        ) : (
          <>
            <div className="dropzone-icon">📤</div>
            <div className="dropzone-title">Drop a PDF here or click to browse</div>
            <div className="dropzone-subtitle">PDF only · Max 50 MB · Async ingestion with progress</div>
          </>
        )}
      </div>

      {/* Upload error */}
      {(uploadError || error) && (
        <div className="alert alert-error">{uploadError || error}</div>
      )}

      {/* In-progress jobs */}
      {activeJobs.length > 0 && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">⏳ Ingestion Jobs</span>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {activeJobs.map((job) => (
              <div key={job.jobId} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-sm truncate" style={{ maxWidth: '60%' }}>{job.filename}</span>
                  <span className={`status-badge status-${job.status}`}>{job.status}</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div className="progress-bar-track" style={{ flex: 1 }}>
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                  <span className="text-xs text-muted" style={{ minWidth: 32 }}>
                    {job.progress}%
                  </span>
                </div>
                <span className="text-xs text-muted">
                  {STAGE_LABELS[job.stage] ?? job.stage}
                  {job.error && (
                    <span style={{ color: 'var(--c-error)', marginLeft: 8 }}>
                      — {job.error}
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Document list */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">📁 My Documents ({documents.length})</span>
          <button
            className="btn btn-ghost btn-sm"
            onClick={refresh}
            disabled={loading}
            id="doc-refresh-btn"
          >
            {loading ? <span className="spinner spinner-sm" /> : '↺ Refresh'}
          </button>
        </div>

        {documents.length === 0 && !loading ? (
          <div className="empty-state" style={{ padding: '40px 24px' }}>
            <div className="empty-state-icon">📭</div>
            <div className="empty-state-title">No documents yet</div>
            <div className="empty-state-desc">
              Upload a PDF above to add it to your private document corpus.
              Documents are isolated per-user.
            </div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="doc-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Size</th>
                  <th>Pages</th>
                  <th>Chunks</th>
                  <th>Status</th>
                  <th>Uploaded</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.document_id}>
                    <td className="truncate" style={{ maxWidth: 240 }}>
                      <span title={doc.filename}>{doc.filename}</span>
                    </td>
                    <td className="text-muted">{formatBytes(doc.file_size_bytes)}</td>
                    <td className="text-muted">{doc.page_count}</td>
                    <td className="text-muted">{doc.chunk_count}</td>
                    <td>
                      <span className={`status-badge status-${doc.status}`}>{doc.status}</span>
                    </td>
                    <td className="text-muted text-xs">{formatDate(doc.created_at)}</td>
                    <td>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(doc.document_id, doc.filename)}
                        id={`delete-doc-${doc.document_id}`}
                        title="Delete document"
                      >
                        🗑
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Info footer */}
      <div className="alert alert-info" style={{ fontSize: '0.78rem' }}>
        ℹ️ Your documents are isolated to your session.
        Only you can query them. Files are not shared with other users.
      </div>
    </div>
  );
}
