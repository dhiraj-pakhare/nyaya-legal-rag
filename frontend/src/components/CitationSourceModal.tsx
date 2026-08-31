import { useEffect } from 'react';
import type { CitationDTO } from '../api/types';

interface Props {
  citation: CitationDTO | null;
  onClose: () => void;
}

export function CitationSourceModal({ citation, onClose }: Props) {
  useEffect(() => {
    if (!citation) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [citation, onClose]);

  if (!citation) return null;

  const chunkId = citation.chunk_id || citation.source_id;

  const getPageRange = () => {
    if (citation.page_start != null && citation.page_end != null) {
      return citation.page_start === citation.page_end
        ? `Page ${citation.page_start}`
        : `Pages ${citation.page_start} – ${citation.page_end}`;
    }
    if (citation.page_start != null) {
      return `Page ${citation.page_start}`;
    }
    if ('page_number' in citation && citation.page_number != null) {
      return `Page ${citation.page_number}`;
    }
    return null;
  };

  const pageRange = getPageRange();

  return (
    <div
      className="citation-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="citation-modal-title"
    >
      <div
        className="citation-modal-container"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="citation-modal-header">
          <div className="citation-modal-header-left">
            <span className={`citation-chip ${citation.citation_type}`}>
              {citation.citation_text}
            </span>
            <span className="citation-modal-type-badge">
              {citation.citation_type === 'STATUTORY'
                ? 'Statutory Evidence'
                : citation.citation_type === 'DOCUMENT'
                ? 'Document Evidence'
                : 'Form Evidence'}
            </span>
            {citation.is_verified && (
              <span className="citation-verified-badge">
                ✓ Verified
              </span>
            )}
          </div>
          <button
            type="button"
            className="citation-modal-close-btn"
            onClick={onClose}
            aria-label="Close citation details"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="citation-modal-body">
          {/* Statutory Citation Fields */}
          {citation.citation_type === 'STATUTORY' && (
            <div className="citation-detail-grid">
              <div className="citation-detail-item">
                <span className="citation-detail-label">Citation</span>
                <span className="citation-detail-value font-mono">
                  {citation.citation_text}
                </span>
              </div>
              <div className="citation-detail-item">
                <span className="citation-detail-label">Act</span>
                <span className="citation-detail-value">
                  {citation.act} ({citation.act_short})
                </span>
              </div>
              <div className="citation-detail-item">
                <span className="citation-detail-label">Section</span>
                <span className="citation-detail-value font-mono">
                  {citation.section}
                </span>
              </div>
              {pageRange && (
                <div className="citation-detail-item">
                  <span className="citation-detail-label">Page Range</span>
                  <span className="citation-detail-value">
                    {pageRange}
                  </span>
                </div>
              )}
              {chunkId && (
                <div className="citation-detail-item">
                  <span className="citation-detail-label">Chunk ID</span>
                  <span className="citation-detail-value font-mono text-xs text-muted">
                    {chunkId}
                  </span>
                </div>
              )}
              <div className="citation-detail-item">
                <span className="citation-detail-label">Verified</span>
                <span className="citation-detail-value text-success font-bold">
                  {citation.is_verified ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          )}

          {/* Document Citation Fields */}
          {citation.citation_type === 'DOCUMENT' && (
            <div className="citation-detail-grid">
              <div className="citation-detail-item">
                <span className="citation-detail-label">Citation</span>
                <span className="citation-detail-value font-mono">
                  {citation.citation_text}
                </span>
              </div>
              <div className="citation-detail-item">
                <span className="citation-detail-label">Filename</span>
                <span className="citation-detail-value">
                  {citation.filename}
                </span>
              </div>
              <div className="citation-detail-item">
                <span className="citation-detail-label">Document ID</span>
                <span className="citation-detail-value font-mono text-xs">
                  {citation.document_id}
                </span>
              </div>
              {pageRange && (
                <div className="citation-detail-item">
                  <span className="citation-detail-label">Page</span>
                  <span className="citation-detail-value">
                    {pageRange}
                  </span>
                </div>
              )}
              <div className="citation-detail-item">
                <span className="citation-detail-label">Verified</span>
                <span className="citation-detail-value text-success font-bold">
                  {citation.is_verified ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          )}

          {/* Form Citation Fields */}
          {citation.citation_type === 'FORM' && (
            <div className="citation-detail-grid">
              <div className="citation-detail-item">
                <span className="citation-detail-label">Citation</span>
                <span className="citation-detail-value font-mono">
                  {citation.citation_text}
                </span>
              </div>
              <div className="citation-detail-item">
                <span className="citation-detail-label">Form</span>
                <span className="citation-detail-value">
                  Form {citation.form_number}: {citation.form_title}
                </span>
              </div>
              {citation.applicable_sections && citation.applicable_sections.length > 0 && (
                <div className="citation-detail-item">
                  <span className="citation-detail-label">Applicable Sections</span>
                  <span className="citation-detail-value font-mono">
                    {citation.applicable_sections.join(', ')}
                  </span>
                </div>
              )}
              {pageRange && (
                <div className="citation-detail-item">
                  <span className="citation-detail-label">Page Range</span>
                  <span className="citation-detail-value">
                    {pageRange}
                  </span>
                </div>
              )}
              <div className="citation-detail-item">
                <span className="citation-detail-label">Verified</span>
                <span className="citation-detail-value text-success font-bold">
                  {citation.is_verified ? 'Yes' : 'No'}
                </span>
              </div>
            </div>
          )}

          {/* Source / Evidence Text Section */}
          <div className="citation-source-text-section">
            <div className="citation-source-text-header">
              <span className="citation-source-text-title">
                {citation.citation_type === 'STATUTORY'
                  ? 'Statutory Evidence Text'
                  : 'Source Evidence Text'}
              </span>
            </div>

            <div className="citation-source-text-box">
              {citation.source_text ? (
                <pre className="citation-source-content font-serif">
                  {citation.source_text}
                </pre>
              ) : citation.citation_type === 'STATUTORY' && citation.section_title ? (
                <div className="citation-source-fallback">
                  <div className="citation-source-title-lead font-semibold">
                    {citation.section_title}
                  </div>
                  <div className="text-xs text-muted" style={{ marginTop: 8 }}>
                    Note: Complete statutory chunk text is indexed under chunk ID{' '}
                    <code className="font-mono">{chunkId}</code>.
                  </div>
                </div>
              ) : citation.citation_type === 'FORM' && citation.form_title ? (
                <div className="citation-source-fallback">
                  <div className="citation-source-title-lead font-semibold">
                    Form {citation.form_number}: {citation.form_title}
                  </div>
                </div>
              ) : (
                <div className="text-muted text-sm italic">
                  No extended source text returned for this citation.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="citation-modal-footer">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
