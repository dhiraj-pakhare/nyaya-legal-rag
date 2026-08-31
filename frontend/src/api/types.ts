/**
 * TypeScript types mirroring backend DTO schemas exactly.
 * Do NOT add fields not present in the backend contract.
 */

// ─── Citations ───────────────────────────────────────────────────────────────

export type CitationType = 'STATUTORY' | 'DOCUMENT' | 'FORM';

export interface BaseCitationDTO {
  citation_text: string;
  citation_type: CitationType;
  is_verified: boolean;
  source_id: string;
  chunk_id?: string;
  source_text?: string;
  page_start: number | null;
  page_end: number | null;
}

export interface StatutoryCitationDTO extends BaseCitationDTO {
  citation_type: 'STATUTORY';
  act: string;
  act_short: string;
  section: string;
  section_title: string;
}

export interface DocumentCitationDTO extends BaseCitationDTO {
  citation_type: 'DOCUMENT';
  document_id: string;
  filename: string;
  page_number: number;
}

export interface FormCitationDTO extends BaseCitationDTO {
  citation_type: 'FORM';
  form_number: number;
  form_title: string;
  applicable_sections: string[];
}

export type CitationDTO = StatutoryCitationDTO | DocumentCitationDTO | FormCitationDTO;

// ─── Query ───────────────────────────────────────────────────────────────────

export interface QueryRequestDTO {
  query: string;
  document_ids?: string[] | null;
  enable_forms?: boolean;
}

export interface QueryResponseDTO {
  query: string;
  status: 'SUCCESS' | 'REFUSED' | 'VALIDATION_FAILED' | 'AMBIGUOUS' | 'NOT_FOUND' | 'ERROR';
  answer: string | null;
  is_refused: boolean;
  refusal_reason: string | null;
  citations: CitationDTO[];
  confidence_score: number;
  routed_corpus: 'STATUTORY' | 'USER_DOCUMENT' | 'COMBINED' | 'STATUTORY_FORM' | string;
  candidate_forms: Record<string, unknown>[] | null;
  telemetry: Record<string, unknown> | null;
}

// SSE stream events
export type SSEEventType = 'status' | 'token' | 'citation' | 'complete' | 'refusal' | 'error';
export interface SSEEvent {
  event: SSEEventType;
  data: Record<string, unknown>;
}

// ─── Documents ───────────────────────────────────────────────────────────────

export type IngestionStatus = 'QUEUED' | 'PROCESSING' | 'READY' | 'FAILED';
export type IngestionStage = 'queued' | 'parsing' | 'chunking' | 'embedding' | 'indexing' | 'complete' | 'failed';

export interface DocumentUploadResponseDTO {
  job_id: string;
  document_id: string;
  filename: string;
  status: IngestionStatus;
  progress: number;
  stage: IngestionStage;
  created_at: string;
  message: string;
}

export interface DocumentStatusDTO {
  job_id: string;
  document_id: string;
  status: IngestionStatus;
  progress: number;
  stage: IngestionStage;
  error: string | null;
  page_count: number | null;
  chunk_count: number | null;
  updated_at: string;
}

export interface DocumentListItemDTO {
  document_id: string;
  filename: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
  status: IngestionStatus;
}

export interface DocumentDetailDTO {
  document_id: string;
  filename: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
  status: IngestionStatus;
  sha256_hash: string;
}

// ─── Statutory Forms ─────────────────────────────────────────────────────────

export interface StatutoryFormListItemDTO {
  form_number: number;
  form_id: string;
  title: string;
  slug: string;
  filename: string;
  applicable_sections: string[];
  page_start: number;
  page_end: number;
  page_count: number;
  byte_size: number | null;
  sha256: string | null;
  extraction_confidence: number;
  needs_review: boolean;
  download_url: string;
  provenance: string;
}

export interface StatutoryFormListResponseDTO {
  total_forms: number;
  schedule: string;
  forms: StatutoryFormListItemDTO[];
}

// ─── Errors ──────────────────────────────────────────────────────────────────

export interface APIErrorDetail {
  code: string;
  message: string;
  status_code: number;
  details: Record<string, unknown> | null;
}

export interface APIErrorResponse {
  error: APIErrorDetail;
}
