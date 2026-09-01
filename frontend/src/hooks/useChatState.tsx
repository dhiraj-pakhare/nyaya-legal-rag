import {
  createContext,
  useContext,
  useState,
  useRef,
  useEffect,
  useCallback,
  type ReactNode,
} from 'react';
import { useSSEStream, type StreamState } from './useSSEStream';
import type { CitationDTO } from '../api/types';

export type { StreamState };

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  citations: CitationDTO[];
  corpus?: string;
  isStreaming?: boolean;
  error?: string | null;
  isRefused?: boolean;
}

export interface ChatStateContextValue {
  messages: ChatMessage[];
  sendMessage: (query: string, activeDocIds?: string[], enableForms?: boolean) => void;
  clearMessages: () => void;
  isStreaming: boolean;
  streamState: StreamState;
}

const ChatStateContext = createContext<ChatStateContextValue | null>(null);

let msgCounter = 0;
function uid() {
  return `msg-${++msgCounter}`;
}

export function ChatStateProvider({ children }: { children: ReactNode }) {
  const { state: streamState, stream, reset: resetStream } = useSSEStream();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const streamMsgId = useRef<string | null>(null);

  // Sync streaming state into the active streaming message
  useEffect(() => {
    const id = streamMsgId.current;
    if (!id) return;

    if (streamState.status === 'streaming' || streamState.status === 'connecting') {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? { ...m, text: streamState.tokens, citations: streamState.citations, isStreaming: true }
            : m
        )
      );
    } else if (streamState.status === 'complete') {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                text: streamState.tokens || m.text,
                citations: streamState.citations,
                isStreaming: false,
                error: streamState.error,
              }
            : m
        )
      );
      streamMsgId.current = null;
    } else if (streamState.status === 'error') {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                isStreaming: false,
                error: streamState.error ?? 'An error occurred.',
                text: m.text || '',
              }
            : m
        )
      );
      streamMsgId.current = null;
    }
  }, [streamState.status, streamState.tokens, streamState.citations, streamState.error]);

  const sendMessage = useCallback(
    (query: string, activeDocIds?: string[], enableForms?: boolean) => {
      const trimmed = query.trim();
      if (!trimmed) return;
      if (streamState.status === 'streaming' || streamState.status === 'connecting') return;

      // Reset previous stream
      resetStream();

      // Add user message
      const userMsg: ChatMessage = { id: uid(), role: 'user', text: trimmed, citations: [] };
      // Add placeholder assistant message
      const assistantMsgId = uid();
      streamMsgId.current = assistantMsgId;
      const assistantMsg: ChatMessage = {
        id: assistantMsgId,
        role: 'assistant',
        text: '',
        citations: [],
        isStreaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      stream({
        query: trimmed,
        document_ids: activeDocIds && activeDocIds.length ? activeDocIds : undefined,
        enable_forms: enableForms,
      });
    },
    [streamState.status, resetStream, stream]
  );

  const clearMessages = useCallback(() => {
    resetStream();
    streamMsgId.current = null;
    setMessages([]);
  }, [resetStream]);

  const isStreaming = streamState.status === 'streaming' || streamState.status === 'connecting';

  return (
    <ChatStateContext.Provider
      value={{
        messages,
        sendMessage,
        clearMessages,
        isStreaming,
        streamState,
      }}
    >
      {children}
    </ChatStateContext.Provider>
  );
}

export function useChatState(): ChatStateContextValue {
  const context = useContext(ChatStateContext);
  if (!context) {
    throw new Error('useChatState must be used within a ChatStateProvider');
  }
  return context;
}
