import { useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthScreen } from './components/AuthScreen';
import { Sidebar } from './components/Sidebar';
import { ChatView } from './views/ChatView';
import { DocumentsView } from './views/DocumentsView';
import { FormsView } from './views/FormsView';
import { useAuth } from './hooks/useAuth';

function PageHeader({
  title,
  subtitle,
  onMenuClick,
}: {
  title: string;
  subtitle?: string;
  onMenuClick?: () => void;
}) {
  return (
    <header className="page-header">
      <button
        type="button"
        className="mobile-menu-btn"
        onClick={onMenuClick}
        aria-label="Open navigation menu"
        id="mobile-nav-toggle"
      >
        <span className="mobile-menu-icon" aria-hidden="true">☰</span>
      </button>
      <div className="page-header-titles">
        <h1 className="page-header-title">{title}</h1>
        {subtitle && <span className="page-header-subtitle">{subtitle}</span>}
      </div>
    </header>
  );
}

function App() {
  const { auth, login, logout, devLogin } = useAuth();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  if (!auth.isAuthenticated) {
    return (
      <AuthScreen
        onLogin={login}
        onDevLogin={devLogin}
      />
    );
  }

  const handleCloseSidebar = () => setSidebarOpen(false);
  const handleOpenSidebar = () => setSidebarOpen(true);

  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar
          userId={auth.userId}
          onLogout={logout}
          isOpen={sidebarOpen}
          onClose={handleCloseSidebar}
        />
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
                    onMenuClick={handleOpenSidebar}
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
                    onMenuClick={handleOpenSidebar}
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
                    onMenuClick={handleOpenSidebar}
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
