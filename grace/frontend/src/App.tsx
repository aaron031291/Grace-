import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { OrbInterface } from './components/OrbInterface'
import { AuthProvider } from './components/AuthProvider'

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-gray-900">
        <Routes>
          <Route path="/" element={<OrbInterface />} />
          <Route path="/session/:sessionId" element={<OrbInterface />} />
        </Routes>
      </div>
    </AuthProvider>
  )
}

export default App