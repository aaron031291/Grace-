import { Routes, Route } from 'react-router-dom'
import { OrbInterface } from './components/OrbInterface'
import { AuthProvider } from './components/AuthProvider'
import { ConnectionTest } from './components/ConnectionTest'

function App() {
  return (
    <AuthProvider>
      <div className="min-h-screen bg-gray-900">
        <Routes>
          <Route path="/" element={<OrbInterface />} />
          <Route path="/session/:sessionId" element={<OrbInterface />} />
          <Route path="/test" element={<ConnectionTest />} />
        </Routes>
      </div>
    </AuthProvider>
  )
}

export default App