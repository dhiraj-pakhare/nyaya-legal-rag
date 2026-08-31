import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthScreen } from './components/AuthScreen';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './views/ChatView';
import { DocumentsView } from './views/DocumentsView';
import { FormsView } from './views/FormsView';
import { useAuth } from './hooks/useAuth';

function PageHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="page-header">
      <h1>{title}</h1>
      {subtitle && <span className="text-xs text-muted">{subtitle}</span>}
    </div>
  );
}

function App() {
  const { auth, login, logout, devLogin } = useAuth();

  if (!auth.isAuthenticated) {
    return (
      <AuthScreen
        onLogin={login}
        onDevLogin={devLogin}
      />
    );
  }

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar userId={auth.userId} onLogout={logout} />
        <div className="main-panel">
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />

            <Route
              path="/chat"
              element={
                <>
                  <PageHeader
                    title="⚖️ Legal Assistant"
                    subtitle="Grounded in BNS 2023 & BNSS 2023 · Citations verified server-side"
                  />
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                    <ChatView />
                  </div>
                </>
              }
            />

            <Route
              path="/documents"
              element={
                <>
                  <PageHeader
                    title="📄 My Documents"
                    subtitle="Upload PDFs for private document RAG · Per-user isolation"
                  />
                  <div style={{ flex: 1, overflowY: 'auto' }}>
                    <DocumentsView />
                  </div>
                </>
              }
            />

            <Route
              path="/forms"
              element={
                <>
                  <PageHeader
                    title="📋 Statutory Forms"
                    subtitle="BNSS 2023 — The Second Schedule · 58 forms"
                  />
                  <div style={{ flex: 1, overflowY: 'auto' }}>
                    <FormsView />
                  </div>
                </>
              }
            />

            <Route path="*" element={<Navigate to="/chat" replace />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}

export default App;
