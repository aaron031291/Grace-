import React, { useState } from 'react'
import { useAuth } from './AuthProvider'
import { PanelManager } from './PanelManager'

export const OrbInterface: React.FC = () => {
  const { isAuthenticated } = useAuth()

  if (!isAuthenticated) {
    return <LoginForm />
  }

  return (
    <div className="h-screen bg-gray-900 text-white overflow-hidden">
      <Header />
      <PanelManager />
    </div>
  )
}

const LoginForm: React.FC = () => {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { login } = useAuth()

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    const success = await login(username, password)
    if (!success) {
      setError('Invalid credentials')
    }
    
    setLoading(false)
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-900">
      <div className="max-w-md w-full space-y-8 p-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-white">
            Grace AI System
          </h2>
          <p className="mt-2 text-center text-sm text-gray-400">
            Sign in to access your governance interface
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleLogin}>
          <div className="space-y-4">
            <div>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="relative block w-full px-3 py-2 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Username or email"
              />
            </div>
            <div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="relative block w-full px-3 py-2 border border-gray-700 placeholder-gray-500 text-white bg-gray-800 rounded-md focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Password"
              />
            </div>
          </div>

          {error && (
            <div className="text-red-400 text-sm text-center">{error}</div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
            >
              {loading ? 'Signing in...' : 'Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

const Header: React.FC = () => {
  const { user, logout } = useAuth()

  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-bold text-white">Grace Orb Interface</h1>
        <div className="connection-status connected">
          Connected
        </div>
      </div>
      <div className="flex items-center space-x-4">
        <span className="text-sm text-gray-300">
          Welcome, {user?.full_name || user?.username}
        </span>
        <button
          onClick={logout}
          className="px-3 py-1 text-sm bg-gray-700 text-white rounded hover:bg-gray-600"
        >
          Logout
        </button>
      </div>
    </header>
  )
}