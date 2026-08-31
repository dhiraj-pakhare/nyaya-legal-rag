import { useState, useCallback } from 'react';
import { setAuthToken } from '../api/client';

export interface AuthState {
  token: string | null;
  userId: string | null;
  isAuthenticated: boolean;
}

/**
 * Very simple auth state manager.
 * In dev mode the backend accepts any Bearer token string.
 * The token is kept in memory only (no localStorage) for security.
 */
export function useAuth() {
  const [auth, setAuth] = useState<AuthState>({
    token: null,
    userId: null,
    isAuthenticated: false,
  });

  const login = useCallback((token: string, userId: string) => {
    setAuthToken(token);
    setAuth({ token, userId, isAuthenticated: true });
  }, []);

  const logout = useCallback(() => {
    setAuthToken(null);
    setAuth({ token: null, userId: null, isAuthenticated: false });
  }, []);

  const devLogin = useCallback(() => {
    const token = 'token_demo_user';
    const userId = 'demo_user';
    setAuthToken(token);
    setAuth({ token, userId, isAuthenticated: true });
  }, []);

  return { auth, login, logout, devLogin };
}
