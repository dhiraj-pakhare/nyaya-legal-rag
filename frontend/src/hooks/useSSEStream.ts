/**
 * useSSEStream – React hook for consuming a backend SSE stream.
 *
 * Protocol (per backend contract):
 *   event: status   – {message: string}
 *   event: citation – {citation: CitationDTO}
 *   event: token    – {token: string}
 *   event: complete – {answer?: string, citations?: CitationDTO[], ...}
 *   event: refusal  – {reason: string}
 *   event: error    – {message: string}
 */

import { useCallback, useRef, useState } from 'react';
import { queryApi, APIError } from '../api/client';
import type { CitationDTO, QueryRequestDTO } from '../api/types';

export type StreamStatus = 'idle' | 'connecting' | 'streaming' | 'complete' | 'error';

export interface StreamState {
  status: StreamStatus;
  tokens: string;
  citations: CitationDTO[];
  statusMessage: string;
  error: string | null;
  retryAfter: number | null;
}

const INITIAL: StreamState = {
  status: 'idle',
  tokens: '',
  citations: [],
  statusMessage: '',
  error: null,
  retryAfter: null,
};

export function useSSEStream() {
  const [state, setState] = useState<StreamState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL);
  }, []);

  const stream = useCallback(async (req: QueryRequestDTO) => {
    // Cancel any in-flight stream
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setState({
      status: 'connecting',
      tokens: '',
      citations: [],
      statusMessage: 'Connecting…',
      error: null,
      retryAfter: null,
    });

    try {
      const response = await queryApi.streamRaw(req);

      if (!response.body) {
        throw new Error('No response body from stream endpoint.');
      }

      setState((prev) => ({ ...prev, status: 'streaming', statusMessage: 'Streaming…' }));

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      outer: while (true) {
        if (controller.signal.aborted) break;

        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE messages are separated by double newline
        const parts = buffer.split('\n\n');
        buffer = parts.pop() ?? '';

        for (const part of parts) {
          if (controller.signal.aborted) break outer;
          if (!part.trim()) continue;

          // Parse "event: X\ndata: {...}"
          let eventType = 'message';
          let dataLine = '';

          for (const line of part.split('\n')) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              dataLine = line.slice(5).trim();
            }
          }

          if (!dataLine) continue;

          let payload: Record<string, unknown> = {};
          try {
            payload = JSON.parse(dataLine) as Record<string, unknown>;
          } catch {
            continue;
          }

          switch (eventType) {
            case 'status':
              setState((prev) => ({
                ...prev,
                statusMessage: (payload.message as string | undefined) ?? '',
              }));
              break;

            case 'token':
              setState((prev) => ({
                ...prev,
                tokens: prev.tokens + ((payload.token as string | undefined) ?? ''),
              }));
              break;

            case 'citation': {
              const incoming: CitationDTO[] = Array.isArray(payload.citations)
                ? (payload.citations as CitationDTO[])
                : payload.citation
                ? [payload.citation as CitationDTO]
                : [];
              setState((prev) => ({
                ...prev,
                citations: [
                  ...prev.citations,
                  ...incoming,
                ],
              }));
              break;
            }

            case 'complete':
              setState((prev) => ({
                ...prev,
                status: 'complete',
                statusMessage: 'Complete',
                // Backend may send final full answer in complete payload
                tokens:
                  typeof payload.answer === 'string'
                    ? payload.answer
                    : prev.tokens,
                citations: Array.isArray(payload.citations)
                  ? (payload.citations as CitationDTO[])
                  : prev.citations,
              }));
              break outer;

            case 'refusal':
              setState((prev) => ({
                ...prev,
                status: 'complete',
                statusMessage: 'Query refused',
                error: (payload.reason as string | undefined) ?? 'Query refused.',
              }));
              break outer;

            case 'error':
              setState((prev) => ({
                ...prev,
                status: 'error',
                error: (payload.message as string | undefined) ?? 'Stream error.',
              }));
              break outer;
          }
        }
      }

      // If we exit loop without hitting 'complete'/'error' event mark complete
      setState((prev) => {
        if (prev.status === 'streaming' || prev.status === 'connecting') {
          return { ...prev, status: 'complete', statusMessage: 'Complete' };
        }
        return prev;
      });
    } catch (err) {
      if ((err as DOMException).name === 'AbortError') return;

      if (err instanceof APIError) {
        setState((prev) => ({
          ...prev,
          status: 'error',
          error:
            err.status === 429
              ? `Rate limit reached. Retry in ${err.retryAfter ?? '?'} seconds.`
              : err.message,
          retryAfter: err.retryAfter ?? null,
        }));
      } else {
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: 'Failed to connect to server. Is the backend running?',
        }));
      }
    }
  }, []);

  return { state, stream, reset };
}
