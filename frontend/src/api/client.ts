/**
 * Centralized API client.
 *
 * Responsibilities:
 *  - Single base URL resolution from VITE_API_BASE_URL
 *  - Authorization header injection (Bearer token)
 *  - 4xx/5xx error normalisation into APIError
 *  - 429 Retry-After header extraction
 *  - SSE streaming helper
 *  - Typed wrappers: query, documents, forms
 */

import type {
  APIErrorResponse,
  DocumentListItemDTO,
  DocumentStatusDTO,
  DocumentUploadResponseDTO,
  QueryRequestDTO,
  QueryResponseDTO,
  StatutoryFormListResponseDTO,
} from './types';

// ─── Configuration ────────────────────────────────────────────────────────────

// In dev, Vite proxies /api → http://localhost:8000 (see vite.config.ts).
// In production, set VITE_API_BASE_URL to the backend origin.
const VITE_API_BASE = import.meta.env.VITE_API_BASE_URL as string | undefined;

const API_BASE = VITE_API_BASE ? `${VITE_API_BASE}/api/v1` : '/api/v1';

// ─── Auth Token Storage ────────────────────────────────────────────────────────

let _authToken: string | null = null;

export function setAuthToken(token: string | null): void {
  _authToken = token;
}

export function getAuthToken(): string | null {
  return _authToken;
}

// ─── Typed Error ───────────────────────────────────────────────────────────────

export class APIError extends Error {
  readonly status: number;
  readonly code: string;
  readonly retryAfter?: number;
  readonly details?: Record<string, unknown> | null;

  constructor(
    status: number,
    code: string,
    message: string,
    retryAfter?: number,
    details?: Record<string, unknown> | null,
  ) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.code = code;
    this.retryAfter = retryAfter;
    this.details = details;
  }
}

// ─── Core Fetch ───────────────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    ...(init.headers as Record<string, string>),
  };

  if (_authToken) {
    headers['Authorization'] = `Bearer ${_authToken}`;
  }

  // Dev fallback: pass a dev header so backend resolves identity
  if (!_authToken && import.meta.env.DEV) {
    headers['X-User-ID'] = 'frontend_dev_user';
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    let code = 'UNKNOWN_ERROR';
    let message = `HTTP ${response.status}`;
    let details: Record<string, unknown> | null = null;
    let retryAfter: number | undefined;

    // Extract Retry-After for 429
    if (response.status === 429) {
      const ra = response.headers.get('Retry-After');
      retryAfter = ra ? parseInt(ra, 10) : undefined;
      code = 'RATE_LIMIT_EXCEEDED';
    }

    try {
      const body = (await response.json()) as APIErrorResponse;
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
      details = body.error?.details ?? null;
    } catch {
      // Body not JSON – use defaults
    }

    throw new APIError(response.status, code, message, retryAfter, details);
  }

  // Handle empty bodies (e.g. 204)
  if (response.status === 204) {
    return undefined as unknown as T;
  }

  return response.json() as Promise<T>;
}

// ─── JSON Helpers ─────────────────────────────────────────────────────────────

function jsonInit(body: unknown, extra: RequestInit = {}): RequestInit {
  return {
    ...extra,
    method: extra.method ?? 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(extra.headers as Record<string, string>),
    },
    body: JSON.stringify(body),
  };
}

// ─── Query API ────────────────────────────────────────────────────────────────

export const queryApi = {
  execute(req: QueryRequestDTO): Promise<QueryResponseDTO> {
    return apiFetch<QueryResponseDTO>('/query', jsonInit(req));
  },

  /**
   * Open an SSE stream for a legal query.
   * The caller receives the raw fetch Response and is responsible for
   * reading `response.body` as a ReadableStream<Uint8Array>.
   */
  async streamRaw(req: QueryRequestDTO): Promise<Response> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    };

    if (_authToken) {
      headers['Authorization'] = `Bearer ${_authToken}`;
    } else if (import.meta.env.DEV) {
      headers['X-User-ID'] = 'frontend_dev_user';
    }

    const response = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(req),
    });

    if (!response.ok) {
      let code = 'STREAM_ERROR';
      let message = `HTTP ${response.status}`;
      let retryAfter: number | undefined;

      if (response.status === 429) {
        const ra = response.headers.get('Retry-After');
        retryAfter = ra ? parseInt(ra, 10) : undefined;
        code = 'RATE_LIMIT_EXCEEDED';
        message = 'Rate limit exceeded. Please wait before retrying.';
      }

      throw new APIError(response.status, code, message, retryAfter);
    }

    return response;
  },
};

// ─── Documents API ────────────────────────────────────────────────────────────

export const documentsApi = {
  upload(file: File): Promise<DocumentUploadResponseDTO> {
    const form = new FormData();
    form.append('file', file);

    const uploadHeaders: Record<string, string> = {};
    if (_authToken) {
      uploadHeaders['Authorization'] = `Bearer ${_authToken}`;
    } else if (import.meta.env.DEV) {
      uploadHeaders['X-User-ID'] = 'frontend_dev_user';
    }

    return fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      headers: uploadHeaders,
      body: form,
    }).then(async (res) => {
      if (!res.ok) {
        let code = 'UPLOAD_ERROR';
        let message = `Upload failed: HTTP ${res.status}`;
        let retryAfter: number | undefined;
        if (res.status === 429) {
          const ra = res.headers.get('Retry-After');
          retryAfter = ra ? parseInt(ra, 10) : undefined;
          code = 'RATE_LIMIT_EXCEEDED';
          message = 'Rate limit exceeded. Please wait before retrying.';
        }
        try {
          const body = (await res.json()) as APIErrorResponse;
          code = body.error?.code ?? code;
          message = body.error?.message ?? message;
        } catch { /* ignore */ }
        throw new APIError(res.status, code, message, retryAfter);
      }
      return res.json() as Promise<DocumentUploadResponseDTO>;
    });
  },

  list(): Promise<DocumentListItemDTO[]> {
    return apiFetch<DocumentListItemDTO[]>('/documents');
  },

  getStatus(documentId: string): Promise<DocumentStatusDTO> {
    return apiFetch<DocumentStatusDTO>(`/documents/${documentId}/status`);
  },

  delete(documentId: string): Promise<void> {
    return apiFetch<void>(`/documents/${documentId}`, { method: 'DELETE' });
  },
};

// ─── Forms API ────────────────────────────────────────────────────────────────

export const formsApi = {
  list(): Promise<StatutoryFormListResponseDTO> {
    return apiFetch<StatutoryFormListResponseDTO>('/forms');
  },

  downloadUrl(filename: string): string {
    if (_authToken) {
      return `${API_BASE}/forms/download/${filename}?token=${encodeURIComponent(_authToken)}`;
    }
    return `${API_BASE}/forms/download/${filename}`;
  },
};
