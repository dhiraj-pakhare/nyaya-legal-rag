import { useEffect, useMemo, useState } from 'react';
import { formsApi, APIError } from '../api/client';
import type { StatutoryFormListItemDTO } from '../api/types';

const BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000';

export function FormsView() {
  const [forms, setForms] = useState<StatutoryFormListItemDTO[]>([]);
  const [totalForms, setTotalForms] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    formsApi.list().then((res) => {
      if (cancelled) return;
      setForms(res.forms);
      setTotalForms(res.total_forms);
    }).catch((err) => {
      if (cancelled) return;
      setError(err instanceof APIError ? err.message : 'Failed to load forms.');
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });

    return () => { cancelled = true; };
  }, []);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return forms;
    return forms.filter(
      (f) =>
        f.title.toLowerCase().includes(q) ||
        String(f.form_number).includes(q) ||
        f.applicable_sections.some((s) => s.toLowerCase().includes(q)) ||
        f.slug.toLowerCase().includes(q),
    );
  }, [forms, searchQuery]);

  return (
    <div className="page-content" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Header strip */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: '0.78rem', color: 'var(--c-text-muted)', marginBottom: 2 }}>
            Bharatiya Nagarik Suraksha Sanhita, 2023 — The Second Schedule
          </div>
          <div style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--c-text)' }}>
            {totalForms} Statutory Forms
            {filtered.length !== forms.length && ` · ${filtered.length} matching`}
          </div>
        </div>

        {/* Search */}
        <div className="search-box">
          <span className="search-box-icon">🔍</span>
          <input
            id="forms-search"
            type="text"
            className="search-input"
            placeholder="Search by title, form number, or section…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      {/* Error */}
      {error && <div className="alert alert-error">{error}</div>}

      {/* Loading */}
      {loading && (
        <div className="empty-state">
          <span className="spinner" />
          <div className="empty-state-title">Loading statutory forms…</div>
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && filtered.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">📋</div>
          <div className="empty-state-title">No forms found</div>
          <div className="empty-state-desc">
            Try adjusting your search query.
          </div>
        </div>
      )}

      {/* Grid */}
      {!loading && filtered.length > 0 && (
        <div className="forms-grid">
          {filtered.map((form) => (
            <FormCard key={form.form_id} form={form} />
          ))}
        </div>
      )}

      {/* Provenance notice */}
      {!loading && !error && forms.length > 0 && (
        <div className="alert alert-info" style={{ fontSize: '0.78rem' }}>
          ℹ️ All forms are extracted from the official BNSS 2023 statutory text.
          Downloads are served directly from the backend.
        </div>
      )}
    </div>
  );
}

function FormCard({ form }: { form: StatutoryFormListItemDTO }) {
  function handleDownload() {
    // The backend serves the file; open in new tab
    const url = form.download_url.startsWith('http')
      ? form.download_url
      : `${BASE_URL}${form.download_url}`;
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  const needsReviewLabel = form.needs_review ? (
    <span
      className="status-badge status-QUEUED"
      style={{ fontSize: '0.65rem' }}
      title="This form requires manual review for accuracy"
    >
      Needs Review
    </span>
  ) : null;

  return (
    <div className="form-card" id={`form-${form.form_number}`}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <span className="form-card-number">Form {form.form_number}</span>
        {needsReviewLabel}
      </div>

      <div className="form-card-title">{form.title}</div>

      {form.applicable_sections.length > 0 && (
        <div className="form-card-sections">
          {form.applicable_sections.slice(0, 4).map((s) => (
            <span key={s} className="form-section-tag">{s}</span>
          ))}
          {form.applicable_sections.length > 4 && (
            <span className="form-section-tag">+{form.applicable_sections.length - 4}</span>
          )}
        </div>
      )}

      <div className="form-card-footer">
        <div className="form-card-meta">
          {form.page_count} page{form.page_count !== 1 ? 's' : ''}
          {form.byte_size
            ? ` · ${(form.byte_size / 1024).toFixed(0)} KB`
            : ''}
          {' · '}
          <span
            style={{ color: form.extraction_confidence >= 0.9 ? 'var(--c-success)' : 'var(--c-warn)' }}
          >
            {Math.round(form.extraction_confidence * 100)}% confidence
          </span>
        </div>
        <button
          className="btn btn-sm btn-ghost"
          onClick={handleDownload}
          id={`download-form-${form.form_number}`}
          title={`Download ${form.filename}`}
        >
          ⬇ PDF
        </button>
      </div>
    </div>
  );
}
