import React, { useEffect, useRef, useState } from 'react';
import { useSSEStream } from '../hooks/useSSEStream';
import { CitationChips } from '../components/CitationChips';
import { CitationSourceModal } from '../components/CitationSourceModal';
import type { CitationDTO } from '../api/types';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  citations: CitationDTO[];
  corpus?: string;
  isStreaming?: boolean;
  error?: string | null;
  isRefused?: boolean;
}

let msgCounter = 0;
function uid() { return `msg-${++msgCounter}`; }

const CORPUS_LABELS: Record<string, string> = {
  STATUTORY:      'Statutory Corpus (BNS/BNSS)',
  USER_DOCUMENT:  'Your Documents',
  COMBINED:       'Combined Sources',
  STATUTORY_FORM: 'Statutory Forms',
};

export function ChatView() {
  const { state: streamState, stream, reset } = useSSEStream();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [enableForms, setEnableForms] = useState(true);
  const [activeDocIds] = useState<string[]>([]);
  const [activeCitation, setActiveCitation] = useState<CitationDTO | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const streamMsgId = useRef<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll when messages or stream tokens change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamState.tokens]);

  // Sync streaming state into the streaming message
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
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamState.status, streamState.tokens, streamState.citations, streamState.error]);

  const handleSend = () => {
    const query = input.trim();
    if (!query) return;
    if (streamState.status === 'streaming' || streamState.status === 'connecting') return;

    // Clear previous stream
    reset();

    // Add user message
    const userMsg: Message = { id: uid(), role: 'user', text: query, citations: [] };
    // Add placeholder assistant message
    const assistantMsgId = uid();
    streamMsgId.current = assistantMsgId;
    const assistantMsg: Message = {
      id: assistantMsgId,
      role: 'assistant',
      text: '',
      citations: [],
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput('');

    stream({
      query,
      document_ids: activeDocIds.length ? activeDocIds : undefined,
      enable_forms: enableForms,
    });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    // Auto-resize textarea
    const el = e.target;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };

  const isDisabled = streamState.status === 'streaming' || streamState.status === 'connecting';

  return (
    <div className="chat-layout">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="empty-state" style={{ marginTop: 40 }}>
            <div className="empty-state-icon">⚖️</div>
            <div className="empty-state-title">Nyaya Legal Assistant</div>
            <div className="empty-state-desc">
              Ask any question about the Bharatiya Nyaya Sanhita (BNS) or Bharatiya Nagarik
              Suraksha Sanhita (BNSS). Citations are verified against the statutory corpus.
            </div>
            <div className="corpus-legend" style={{ marginTop: 20, justifyContent: 'center' }}>
              <div className="corpus-legend-item">
                <div className="corpus-legend-dot" style={{ background: 'var(--c-statutory)' }} />
                Statutory Corpus
              </div>
              <div className="corpus-legend-item">
                <div className="corpus-legend-dot" style={{ background: 'var(--c-document)' }} />
                Your Documents
              </div>
              <div className="corpus-legend-item">
                <div className="corpus-legend-dot" style={{ background: 'var(--c-form)' }} />
                Statutory Forms
              </div>
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`chat-message ${msg.role}`}>
            <div className={`message-avatar ${msg.role}`}>
              {msg.role === 'user' ? '👤' : '⚖️'}
            </div>
            <div className="message-body">
              {/* Corpus badge for assistant */}
              {msg.role === 'assistant' && msg.corpus && (
                <span className={`message-corpus-badge corpus-${msg.corpus}`}>
                  {CORPUS_LABELS[msg.corpus] ?? msg.corpus}
                </span>
              )}

              {/* Bubble */}
              <div className={`message-bubble${msg.isStreaming && !msg.text ? ' text-muted' : ''}`}>
                {msg.isStreaming && !msg.text ? (
                  <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span className="spinner spinner-sm" />
                    <span className="text-muted text-sm">{streamState.statusMessage || 'Thinking…'}</span>
                  </span>
                ) : msg.isRefused ? (
                  <div className="refusal-banner">
                    <span className="refusal-banner-icon">🚫</span>
                    <span>{msg.text || 'Query refused.'}</span>
                  </div>
                ) : (
                  <span className={msg.isStreaming ? 'typing-cursor' : ''}>
                    {msg.text || '\u00a0'}
                  </span>
                )}
              </div>

              {/* Citations */}
              {msg.citations.length > 0 && !msg.isStreaming && (
                <CitationChips
                  citations={msg.citations}
                  onSelectCitation={setActiveCitation}
                />
              )}

              {/* Error */}
              {msg.error && !msg.isStreaming && (
                <div className="alert alert-error" style={{ fontSize: '0.8rem' }}>
                  ⚠️ {msg.error}
                </div>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Citation Evidence Modal */}
      <CitationSourceModal
        citation={activeCitation}
        onClose={() => setActiveCitation(null)}
      />

      {/* Input Area */}
      <div className="chat-input-area">
        {/* Rate limit banner */}
        {streamState.status === 'error' && streamState.retryAfter && (
          <div className="alert alert-warn" style={{ marginBottom: 10 }}>
            ⏳ Rate limit reached. Please wait {streamState.retryAfter}s before retrying.
          </div>
        )}

        <div className="chat-input-row">
          <div className="chat-input-wrapper">
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-input"
              placeholder="Ask a legal question… (Enter to send, Shift+Enter for newline)"
              value={input}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              disabled={isDisabled}
              rows={1}
            />
          </div>
          <button
            id="chat-send-btn"
            className="btn btn-primary"
            onClick={handleSend}
            disabled={isDisabled || !input.trim()}
            style={{ height: 48, padding: '0 20px' }}
          >
            {isDisabled ? (
              <><span className="spinner spinner-sm" /> Streaming…</>
            ) : (
              'Send ↑'
            )}
          </button>
        </div>

        <div className="chat-options-row">
          <label className="chat-option-label" htmlFor="enable-forms">
            <input
              id="enable-forms"
              type="checkbox"
              checked={enableForms}
              onChange={(e) => setEnableForms(e.target.checked)}
            />
            Include statutory forms
          </label>
          <span className="text-xs text-muted">
            Streaming · Citations verified server-side
          </span>
        </div>
      </div>
    </div>
  );
}
