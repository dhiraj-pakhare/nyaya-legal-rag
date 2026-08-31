import React from 'react';

interface Props {
  onDevLogin: () => void;
  onLogin: (token: string, userId: string) => void;
}

export function AuthScreen({ onDevLogin, onLogin }: Props) {
  const [token, setToken] = React.useState('');
  const [error, setError] = React.useState('');

  function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    const t = token.trim();
    if (!t) { setError('Please enter a token.'); return; }
    // Extract userId from token if structured (e.g. "token_alice" → "alice")
    let userId = t;
    if (t.startsWith('token_')) {
      userId = t.slice(6);
    } else if (t.includes(':')) {
      userId = t.split(':')[0];
    }
    onLogin(t, userId);
  }

  return (
    <div className="auth-screen">
      <div className="auth-card">
        <div className="auth-logo">⚖️</div>
        <h1 className="auth-title">Nyaya Legal RAG</h1>
        <p className="auth-subtitle">
          AI-powered legal assistant grounded in Bharatiya Nyaya Sanhita
        </p>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label className="form-label" htmlFor="auth-token">
              Bearer Token
            </label>
            <input
              id="auth-token"
              type="text"
              className="form-input"
              placeholder="token_yourname"
              value={token}
              onChange={(e) => { setToken(e.target.value); setError(''); }}
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          {error && (
            <div className="alert alert-error" style={{ marginBottom: 12, fontSize: '0.8rem' }}>
              {error}
            </div>
          )}
          <button type="submit" className="btn btn-primary w-full" style={{ width: '100%', justifyContent: 'center' }}>
            Sign in
          </button>
        </form>

        <div className="divider" />

        <button
          type="button"
          className="btn btn-ghost w-full"
          style={{ width: '100%', justifyContent: 'center' }}
          onClick={onDevLogin}
        >
          🛠 Continue as Demo User
        </button>

        <p className="text-xs text-muted" style={{ marginTop: 14, textAlign: 'center', lineHeight: 1.5 }}>
          Demo login uses a pre-configured development identity.
          No data is persisted in this session.
        </p>
      </div>
    </div>
  );
}
