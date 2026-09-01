import { useState, useCallback } from 'react';
import { setAuthToken, getAuthToken } from '../api/client';

export interface AuthState {
  token: string | null;
  userId: string | null;
  isAuthenticated: boolean;
}

function extractUserId(token: string): string {
  if (token.startsWith('token_')) {
    return token.slice(6);
  } else if (token.includes(':')) {
    return token.split(':')[0];
  }
  return token;
}

/**
 * Auth state manager.
 * Persists the session token in sessionStorage (via client.ts) so that page
 * refreshes and in-session navigations retain authenticated state.
 */
export function useAuth() {
  const [auth, setAuth] = useState<AuthState>(() => {
    const initialToken = getAuthToken();
    if (initialToken) {
      return {
        token: initialToken,
        userId: extractUserId(initialToken),
        isAuthenticated: true,
      };
    }
    return {
      token: null,
      userId: null,
      isAuthenticated: false,
    };
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
