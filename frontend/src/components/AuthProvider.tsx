import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

interface User {
  id: string
  username: string
  email: string
  full_name: string
  is_active: boolean
  is_superuser: boolean
}

interface AuthContextType {
  user: User | null
  token: string | null
  login: (username: string, password: string) => Promise<boolean>
  logout: () => void
  isAuthenticated: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

interface AuthProviderProps {
  children: ReactNode
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('grace_token')
  )

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const payload = {
        user_id: username,
        roles: ['user']
      }

      const response = await fetch('/api/v1/auth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (response.ok) {
        const data = await response.json()
        setToken(data.access_token)
        localStorage.setItem('grace_token', data.access_token)
        // Fetch user info
        const userResponse = await fetch('/api/v1/auth/me', {
          headers: {
            'Authorization': `Bearer ${data.access_token}`,
          },
        })
        
        if (userResponse.ok) {
          const userData = await userResponse.json()
          setUser(userData)
          return true
        }
      }
      
      return false
    } catch (error) {
      console.error('Login error:', error)
      return false
    }
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    localStorage.removeItem('grace_token')
  }

  // Check token validity on mount
  useEffect(() => {
    if (token) {
      fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      })
        .then(response => {
          if (response.ok) {
            return response.json()
          } else {
            throw new Error('Token invalid')
          }
        })
        .then(userData => setUser(userData))
        .catch(() => logout())
    }
  }, [token])

  const value: AuthContextType = {
    user,
    token,
    login,
    logout,
    isAuthenticated: !!user
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}