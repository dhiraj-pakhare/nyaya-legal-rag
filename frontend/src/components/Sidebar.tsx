import { NavLink } from 'react-router-dom';

interface Props {
  userId: string | null;
  onLogout: () => void;
}

const NAV_ITEMS = [
  { to: '/chat',      icon: '⚖️', label: 'Legal Assistant' },
  { to: '/documents', icon: '📄', label: 'My Documents' },
  { to: '/forms',     icon: '📋', label: 'Statutory Forms' },
];

export function Sidebar({ userId, onLogout }: Props) {
  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-logo">
          <div className="sidebar-brand-icon">⚖️</div>
          <div>
            <div className="sidebar-brand-name">Nyaya</div>
            <div className="sidebar-brand-subtitle">Legal RAG</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ to, icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
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
        <button className="btn btn-ghost btn-sm" style={{ width: '100%' }} onClick={onLogout}>
          Sign out
        </button>
      </div>
    </aside>
  );
}
