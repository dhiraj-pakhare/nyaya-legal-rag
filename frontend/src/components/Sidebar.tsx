import { NavLink } from 'react-router-dom';

interface Props {
  userId: string | null;
  onLogout: () => void;
  isOpen?: boolean;
  onClose?: () => void;
}

const NAV_ITEMS = [
  { to: '/chat',      icon: '⚖️', label: 'Legal Assistant' },
  { to: '/documents', icon: '📄', label: 'My Documents' },
  { to: '/forms',     icon: '📋', label: 'Statutory Forms' },
];

export function Sidebar({ userId, onLogout, isOpen = false, onClose }: Props) {
  return (
    <>
      {/* Mobile backdrop */}
      <div
        className={`sidebar-backdrop ${isOpen ? 'open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`sidebar ${isOpen ? 'open' : ''}`} aria-label="Main Navigation">
        {/* Brand */}
        <div className="sidebar-brand">
          <div className="sidebar-brand-logo">
            <div className="sidebar-brand-icon">⚖️</div>
            <div>
              <div className="sidebar-brand-name">Nyaya</div>
              <div className="sidebar-brand-subtitle">Legal RAG</div>
            </div>
          </div>
          {onClose && (
            <button
              type="button"
              className="sidebar-close-btn"
              onClick={onClose}
              aria-label="Close navigation menu"
              id="sidebar-close-btn"
            >
              ✕
            </button>
          )}
        </div>

        {/* Nav */}
        <nav className="sidebar-nav">
          {NAV_ITEMS.map(({ to, icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
              onClick={onClose}
            >
              <span className="nav-item-icon">{icon}</span>
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="sidebar-footer">
          {userId && (
            <div
              style={{
                fontSize: '0.75rem',
                color: 'var(--c-text-muted)',
                marginBottom: 8,
                padding: '0 4px',
              }}
            >
              Signed in as{' '}
              <span style={{ color: 'var(--c-text-dim)', fontWeight: 600 }}>
                {userId}
              </span>
            </div>
          )}
          <button
            className="btn btn-ghost btn-sm"
            style={{ width: '100%' }}
            onClick={() => {
              onClose?.();
              onLogout();
            }}
          >
            Sign out
          </button>
        </div>
      </aside>
    </>
  );
}
