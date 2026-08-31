/**
 * useDocuments – React hook for document lifecycle management.
 * Wraps listing, uploading, status polling, and deletion.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { documentsApi, APIError } from '../api/client';
import type { DocumentListItemDTO, DocumentStatusDTO, IngestionStatus } from '../api/types';

export interface UploadJob {
  jobId: string;
  documentId: string;
  filename: string;
  status: IngestionStatus;
  progress: number;
  stage: string;
  error: string | null;
}

export function useDocuments() {
  const [documents, setDocuments] = useState<DocumentListItemDTO[]>([]);
  const [jobs, setJobs] = useState<Map<string, UploadJob>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Keep a set of document IDs currently being polled
  const pollingRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const docs = await documentsApi.list();
      setDocuments(docs);
    } catch (err) {
      setError(err instanceof APIError ? err.message : 'Failed to load documents.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
    // Cleanup polling on unmount
    return () => {
      pollingRef.current.forEach((t) => clearTimeout(t));
      pollingRef.current.clear();
    };
  }, [fetchDocuments]);

  const startPolling = useCallback(
    (documentId: string, jobId: string) => {
      const poll = async () => {
        try {
          const statusDTO: DocumentStatusDTO = await documentsApi.getStatus(documentId);
          setJobs((prev) => {
            const next = new Map(prev);
            next.set(jobId, {
              jobId: statusDTO.job_id,
              documentId: statusDTO.document_id,
              filename: prev.get(jobId)?.filename ?? documentId,
              status: statusDTO.status,
              progress: statusDTO.progress,
              stage: statusDTO.stage,
              error: statusDTO.error,
            });
            return next;
          });

          if (statusDTO.status === 'READY') {
            pollingRef.current.delete(jobId);
            // Refresh document list when indexing is complete
            fetchDocuments();
          } else if (statusDTO.status === 'FAILED') {
            pollingRef.current.delete(jobId);
          } else {
            // Continue polling every 2 s
            const timer = setTimeout(() => poll(), 2000);
            pollingRef.current.set(jobId, timer);
          }
        } catch {
          // If status check fails, stop polling
          pollingRef.current.delete(jobId);
        }
      };

      poll();
    },
    [fetchDocuments],
  );

  const uploadDocument = useCallback(
    async (file: File): Promise<boolean> => {
      try {
        const uploadRes = await documentsApi.upload(file);
        const job: UploadJob = {
          jobId: uploadRes.job_id,
          documentId: uploadRes.document_id,
          filename: uploadRes.filename,
          status: uploadRes.status,
          progress: uploadRes.progress,
          stage: uploadRes.stage,
          error: null,
        };
        setJobs((prev) => new Map(prev).set(uploadRes.job_id, job));
        startPolling(uploadRes.document_id, uploadRes.job_id);
        return true;
      } catch (err) {
        const msg =
          err instanceof APIError
            ? err.status === 429
              ? `Rate limit exceeded. Retry in ${err.retryAfter ?? '?'}s.`
              : err.message
            : 'Upload failed. Please try again.';
        setError(msg);
        return false;
      }
    },
    [startPolling],
  );

  const deleteDocument = useCallback(
    async (documentId: string): Promise<void> => {
      await documentsApi.delete(documentId);
      setDocuments((prev) => prev.filter((d) => d.document_id !== documentId));
    },
    [],
  );

  return {
    documents,
    jobs,
    loading,
    error,
    uploadDocument,
    deleteDocument,
    refresh: fetchDocuments,
  };
}
