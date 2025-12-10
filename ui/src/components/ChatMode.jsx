import { useState } from 'react'

const ChatMode = ({ onSubmit, loading }) => {
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [model, setModel] = useState('auto')
  const [topK, setTopK] = useState(5)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (query.trim()) {
      // Add user message
      const userMessage = { role: 'user', content: query, timestamp: new Date() }
      setMessages(prev => [...prev, userMessage])
      
      // Submit and wait for response
      const queryData = { query, model, top_k: topK }
      setQuery('')
      
      try {
        const result = await onSubmit(queryData)
        
        // Add assistant message
        const assistantMessage = {
          role: 'assistant',
          content: result.answer || 'No response generated',
          confidence: result.confidence,
          docs: result.retrieved_docs,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, assistantMessage])
      } catch (error) {
        const errorMessage = {
          role: 'error',
          content: error.message || 'Failed to get response',
          timestamp: new Date()
        }
        setMessages(prev => [...prev, errorMessage])
      }
    }
  }

  const clearHistory = () => {
    setMessages([])
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden flex flex-col h-[600px]">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-4 flex items-center justify-between">
        <h3 className="text-xl font-bold text-white">Chat Mode</h3>
        <button
          onClick={clearHistory}
          className="px-3 py-1 bg-white/20 hover:bg-white/30 text-white text-sm rounded transition-colors"
          disabled={messages.length === 0}
        >
          Clear History
        </button>
      </div>

      {/* Messages Area */}
      <div className="flex-1 p-6 overflow-y-auto space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-12">
            <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p className="mt-4 text-sm">Start a conversation</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[80%] rounded-lg px-4 py-3 ${
                msg.role === 'user' 
                  ? 'bg-blue-500 text-white' 
                  : msg.role === 'error'
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-gray-100 text-gray-900'
              }`}>
                <p className="text-sm">{msg.content}</p>
                {msg.confidence !== undefined && (
                  <p className="text-xs mt-2 opacity-75">
                    Confidence: {(msg.confidence * 100).toFixed(1)}% • Docs: {msg.docs}
                  </p>
                )}
                <p className="text-xs mt-1 opacity-50">
                  {msg.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4 bg-gray-50">
        <form onSubmit={handleSubmit} className="space-y-3">
          <div className="flex space-x-2">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
              disabled={loading}
            >
              <option value="auto">Auto (MoE)</option>
              <option value="inlegalbert">InLegalBERT</option>
              <option value="incaselawbert">InCaseLawBERT</option>
              <option value="indicbert">IndicBERT</option>
            </select>
            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm"
              min="1"
              max="20"
              disabled={loading}
            />
          </div>
          <div className="flex space-x-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? '...' : 'Send'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default ChatMode
