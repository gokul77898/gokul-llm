import { useState } from 'react'
import ChatGPT from './components/ChatGPT'
import MonitoringDashboard from './components/MonitoringDashboard'
import AdminDashboard from './components/AdminDashboard'
import MOETester from './components/MOETester'

function App() {
  const [view, setView] = useState('moe') // Default to MoE tester

  return (
    <div className="min-h-screen">
      {/* View Switcher */}
      <div className="fixed top-4 right-4 z-50 flex gap-2">
        <button
          onClick={() => setView('moe')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'moe'
              ? 'bg-orange-600 text-white shadow-lg'
              : 'bg-white text-gray-700 hover:bg-gray-100 shadow'
          }`}
        >
          🤖 MoE Test
        </button>
        <button
          onClick={() => setView('chat')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'chat'
              ? 'bg-blue-600 text-white shadow-lg'
              : 'bg-white text-gray-700 hover:bg-gray-100 shadow'
          }`}
        >
          💬 Chat
        </button>
        <button
          onClick={() => setView('dashboard')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'dashboard'
              ? 'bg-purple-600 text-white shadow-lg'
              : 'bg-white text-gray-700 hover:bg-gray-100 shadow'
          }`}
        >
          📊 Monitor
        </button>
        <button
          onClick={() => setView('admin')}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            view === 'admin'
              ? 'bg-green-600 text-white shadow-lg'
              : 'bg-white text-gray-700 hover:bg-gray-100 shadow'
          }`}
        >
          ⚙️ Admin
        </button>
      </div>

      {/* Render Selected View */}
      {view === 'moe' && <MOETester />}
      {view === 'chat' && <ChatGPT />}
      {view === 'dashboard' && <MonitoringDashboard />}
      {view === 'admin' && <AdminDashboard />}
    </div>
  )
}

export default App
