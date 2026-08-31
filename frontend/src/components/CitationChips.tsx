import type { CitationDTO } from '../api/types';

interface Props {
  citations: CitationDTO[];
  onSelectCitation?: (citation: CitationDTO) => void;
}

const TYPE_LABELS: Record<string, string> = {
  STATUTORY: '§',
  DOCUMENT:  '📄',
  FORM:      '📋',
};

export function CitationChips({ citations, onSelectCitation }: Props) {
  if (!citations.length) return null;
  return (
    <div className="citations-list">
      {citations.map((c, i) => (
        <button
          type="button"
          key={`${c.source_id}-${i}`}
          className={`citation-chip ${c.citation_type} ${onSelectCitation ? 'clickable' : ''}`}
          title={getCitationTitle(c)}
          onClick={() => onSelectCitation?.(c)}
          aria-label={`View evidence for citation ${c.citation_text}`}
        >
          {c.is_verified && <span className="citation-verified-dot" />}
          {TYPE_LABELS[c.citation_type] ?? ''}
          {' '}{c.citation_text}
        </button>
      ))}
    </div>
  );
}

function getCitationTitle(c: CitationDTO): string {
  if (c.citation_type === 'STATUTORY') {
    return `${c.act_short} § ${c.section} — ${c.section_title}`;
  }
  if (c.citation_type === 'DOCUMENT') {
    return `${c.filename}, page ${c.page_number}`;
  }
  if (c.citation_type === 'FORM') {
    return `Form ${c.form_number}: ${c.form_title}`;
  }
  return (c as { citation_text: string }).citation_text;
}
