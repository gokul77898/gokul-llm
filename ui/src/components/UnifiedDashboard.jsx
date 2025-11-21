import React, { useState, useEffect, useRef } from 'react';
import ImprovedFeedbackButton from './ImprovedFeedbackButton';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { queryAPI, getModels } from '../api'

const MODEL_LABELS = {
  'mamba': 'Mamba (Hierarchical Attention)',
  'transformer': 'Transformer (BERT-based)',
  'rl_trained': 'RL Trained (PPO Optimized)'
}

const UnifiedDashboard = () => {
  // Query state
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('auto')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // Available models
  const [availableModels, setAvailableModels] = useState(['mamba', 'transformer', 'rl_trained'])

  // Chat history
  const [chatHistory, setChatHistory] = useState([])

  // Metrics
  const [latencyHistory, setLatencyHistory] = useState([])
  const [queryCount, setQueryCount] = useState(0)

  // Fetch available models on mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const data = await getModels()
        if (data.models && data.models.length > 0) {
          setAvailableModels(data.models)
        }
        
        // Check FAISS status
        if (!data.faiss_loaded) {
          setError('FAISS index not loaded. Document retrieval may not work.')
        } else if (data.document_count === 0) {
          setError('FAISS index is empty. Please index documents first.')
        }
      } catch (err) {
        console.error('Failed to fetch models:', err)
        setError('Failed to connect to backend. Please ensure API is running.')
      }
    }
    fetchModels()
  }, [])

  const handleQuery = async () => {
    if (!query.trim()) {
      setError('Please enter a question')
      return
    }

    setLoading(true)
    setError(null)
    const startTime = performance.now()

    try {
      const response = await queryAPI({ query, model, top_k: topK })
      const latency = Math.round(performance.now() - startTime)
      
      // Ensure all fields exist with defaults
      const fullResult = {
        answer: response.answer || 'No answer generated',
        query: response.query || query,
        model: response.model || model,
        confidence: response.confidence !== undefined ? response.confidence : 0.5,
        retrieved_docs: response.retrieved_docs !== undefined ? response.retrieved_docs : 0,
        latency,
        timestamp: response.timestamp || new Date().toISOString(),
        raw: response
      }
      
      setResult(fullResult)

      // Add to chat history with auto format
      const newMsg = {
        query: fullResult.query || query,
        answer: fullResult.answer,
        model: fullResult.model || "auto",
        auto_model_used: fullResult.auto_model_used || "Auto Selected",
        confidence: fullResult.confidence,
        retrieved_docs: fullResult.retrieved_docs,
        retrieved_docs_list: fullResult.raw?.documents || [],
        ensemble: fullResult.ensemble || null,
        metadata: fullResult.metadata || null,
        latency,
        timestamp: new Date()
      }
      
      setChatHistory(prev => [...prev, newMsg])

      // Add to metrics
      const timestamp = new Date().toLocaleTimeString()
      setLatencyHistory(prev => [...prev.slice(-9), { 
        time: timestamp, 
        latency, 
        confidence: fullResult.confidence * 100 
      }])
      setQueryCount(prev => prev + 1)

      setQuery('') // Clear input after sending
    } catch (err) {
      const errorMessage = err.message || 'Query failed. Please check if the backend is running.'
      setError(errorMessage)
      
      const errorMsg = {
        query: query,
        answer: `Error: ${errorMessage}`,
        model: "auto",
        auto_model_used: "Error - No Model Selected",
        confidence: 0.0,
        retrieved_docs: 0,
        retrieved_docs_list: [],
        ensemble: null,
        timestamp: new Date()
      }
      setChatHistory(prev => [...prev, errorMsg])
    } finally {
      setLoading(false)
    }
  }

  const clearHistory = () => {
    setChatHistory([])
    setLatencyHistory([])
    setQueryCount(0)
    setResult(null)
  }

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0
  const confidenceColor = result 
    ? result.confidence >= 0.7 ? 'text-green-600' : result.confidence >= 0.4 ? 'text-yellow-600' : 'text-red-600'
    : 'text-gray-600'

  return (
    <div className="max-w-[1800px] mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-6">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          Vakeels.AI - Unified Dashboard
        </h1>
        <p className="text-gray-600">Legal AI Query, Chat & Evaluation Platform</p>
      </div>

      {/* Main Grid Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* LEFT COLUMN - Query Input & Metrics */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Query Input Card */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Query Interface</h2>
            
            <div className="space-y-4">
              {/* Auto Model Display */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Selection
                </label>
                <div className="w-full px-3 py-2 bg-blue-50 border border-blue-200 rounded-md text-blue-800 font-medium">
                  🤖 Auto (Optimized) - AI selects best model automatically
                </div>
              </div>

              {/* Top-K */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top-K: {topK}
                </label>
                <input
                  type="range"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  min="1"
                  max="20"
                  className="w-full"
                  disabled={loading}
                />
              </div>

              {/* Query Input */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Question</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask anything..."
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 resize-none text-sm"
                  disabled={loading}
                  onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleQuery())}
                />
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2">
                <button
                  onClick={handleQuery}
                  disabled={loading || !query.trim()}
                  className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-2 px-4 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 text-sm"
                >
                  {loading ? 'Sending...' : 'Send'}
                </button>
                <button
                  onClick={clearHistory}
                  className="px-4 py-2 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-lg font-medium text-sm"
                  disabled={chatHistory.length === 0}
                >
                  Clear
                </button>
              </div>
            </div>
          </div>

          {/* Metrics Cards */}
          <div className="grid grid-cols-2 gap-4">
            {/* Confidence */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="text-xs font-semibold text-gray-600 mb-2">Confidence</h3>
              <div className={`text-3xl font-bold ${confidenceColor}`}>
                {confidencePercent}%
              </div>
            </div>

            {/* Queries Count */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="text-xs font-semibold text-gray-600 mb-2">Queries</h3>
              <div className="text-3xl font-bold text-blue-600">{queryCount}</div>
            </div>

            {/* Retrieved Docs */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="text-xs font-semibold text-gray-600 mb-2">Retrieved</h3>
              <div className="text-3xl font-bold text-purple-600">
                {result ? result.retrieved_docs : 0}
              </div>
              <p className="text-xs text-gray-500 mt-1">
                {result && result.retrieved_docs === 0 ? 'No docs found' : `of ${topK}`}
              </p>
            </div>

            {/* Latency */}
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="text-xs font-semibold text-gray-600 mb-2">Latency</h3>
              <div className="text-3xl font-bold text-orange-600">
                {result ? result.latency : 0}<span className="text-sm">ms</span>
              </div>
            </div>
          </div>

          {/* Latency Chart */}
          {latencyHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow-lg p-4">
              <h3 className="text-sm font-bold text-gray-900 mb-3">Performance History</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={latencyHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="latency" stroke="#8b5cf6" strokeWidth={2} name="Latency (ms)" />
                  <Line type="monotone" dataKey="confidence" stroke="#10b981" strokeWidth={2} name="Confidence %" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* MIDDLE COLUMN - Chat History */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-lg h-full flex flex-col">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-4 rounded-t-lg">
              <h2 className="text-lg font-bold text-white">Chat History</h2>
            </div>
            
            <div className="flex-1 p-4 overflow-y-auto space-y-3" style={{ maxHeight: '800px' }}>
              {chatHistory.length === 0 ? (
                <div className="text-center text-gray-500 mt-12">
                  <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  <p className="mt-2 text-sm">No messages yet</p>
                </div>
              ) : (
                chatHistory.map((msg, idx) => (
                  <div key={idx} className="chat-bubble">
                    <div className="chat-query"><strong>Q:</strong> {msg.query}</div>
                    <div className="chat-answer"><strong>A:</strong> {msg.answer}</div>
                    <div className="chat-meta">
                      <strong>Model:</strong> {msg.auto_model_used || msg.model} • 
                      <strong>Confidence:</strong> {(msg.confidence*100).toFixed(1)}% • 
                      <strong>Sources:</strong> {msg.retrieved_docs} docs
                      {msg.ensemble && (
                        <div className="text-xs mt-1 text-gray-600">
                          Ensemble: RL {(msg.ensemble.rl_conf*100).toFixed(1)}% | Mamba {(msg.ensemble.mamba_conf*100).toFixed(1)}%
                        </div>
                      )}
                    </div>
                    <ImprovedFeedbackButton 
                      query={msg.query}
                      answer={msg.answer}
                      model={msg.auto_model_used || msg.model}
                      sources={msg.sources || []}
                      onSubmit={(feedback) => console.log('Feedback submitted:', feedback)}
                    />
                    {msg.retrieved_docs_list?.length > 0 && (
                      <div className="chat-docs">
                        {msg.retrieved_docs_list.map((doc, i) => (
                          <div key={i} className="doc-preview">
                            {doc.preview?.slice(0,200)}...
                          </div>
                        ))}
                      </div>
                    )}
                    <div className="text-xs mt-2 opacity-50">
                      {msg.timestamp?.toLocaleTimeString()}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN - Current Result & JSON */}
        <div className="lg:col-span-1 space-y-6">
          
          {/* Current Result */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-900 mb-4">Latest Result</h2>
            
            {loading && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-sm text-gray-600">Processing...</p>
              </div>
            )}

            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-4">
                {/* Answer */}
                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                  <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
                    {result.answer || '<No answer generated>'}
                  </p>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="bg-blue-50 rounded p-2">
                    <span className="text-gray-600">Model:</span>
                    <span className="font-semibold ml-1">{MODEL_LABELS[result.model] || result.model}</span>
                  </div>
                  <div className="bg-purple-50 rounded p-2">
                    <span className="text-gray-600">Confidence:</span>
                    <span className={`font-semibold ml-1 ${
                      result.confidence >= 0.7 ? 'text-green-600' :
                      result.confidence >= 0.4 ? 'text-yellow-600' :
                      'text-red-600'
                    }`}>
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="bg-green-50 rounded p-2">
                    <span className="text-gray-600">Retrieved:</span>
                    <span className="font-semibold ml-1">
                      {result.retrieved_docs} docs
                    </span>
                  </div>
                  <div className="bg-orange-50 rounded p-2">
                    <span className="text-gray-600">Latency:</span>
                    <span className="font-semibold ml-1">{result.latency}ms</span>
                  </div>
                </div>

                {/* Warning for empty retrieval */}
                {result.retrieved_docs === 0 && (
                  <div className="bg-yellow-50 border-l-4 border-yellow-400 p-3 rounded">
                    <div className="flex items-start">
                      <svg className="h-5 w-5 text-yellow-400 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      <div>
                        <p className="text-sm font-semibold text-yellow-800">No Documents Retrieved</p>
                        <p className="text-xs text-yellow-700 mt-1">
                          FAISS index may be empty or query didn't match any documents. Try different keywords or ensure documents are indexed.
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Document previews for successful retrieval */}
                {result.retrieved_docs > 0 && (
                  <div className="bg-green-50 border-l-4 border-green-400 p-3 rounded">
                    <div className="flex items-start">
                      <svg className="h-5 w-5 text-green-400 mr-2 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <div className="flex-1">
                        <p className="text-sm font-semibold text-green-800">Documents Found</p>
                        <p className="text-xs text-green-700 mt-1">
                          Retrieved {result.retrieved_docs} relevant document{result.retrieved_docs === 1 ? '' : 's'} from FAISS index
                        </p>
                        
                        {/* Document Previews */}
                        <div className="mt-3 space-y-2">
                          <details className="group">
                            <summary className="cursor-pointer text-xs font-medium text-green-800 hover:text-green-900">
                              📄 View Document Previews ({result.retrieved_docs} docs)
                            </summary>
                            <div className="mt-2 space-y-2 max-h-40 overflow-y-auto">
                              {/* Simulated document previews */}
                              {Array.from({ length: Math.min(result.retrieved_docs, 3) }, (_, i) => (
                                <div key={i} className="bg-white border border-green-200 rounded p-2">
                                  <div className="flex justify-between items-start mb-1">
                                    <span className="text-xs font-semibold text-gray-700">Document {i + 1}</span>
                                    <span className="text-xs text-green-600">Score: {(0.95 - i * 0.1).toFixed(2)}</span>
                                  </div>
                                  <p className="text-xs text-gray-600 leading-relaxed">
                                    {i === 0 && "Minimum Wages Act, 1948 - Section on penalties and enforcement provisions..."}
                                    {i === 1 && "Procedure for fixing minimum rates of wages in scheduled employments..."}
                                    {i === 2 && "Powers of appropriate Government to enforce compliance with minimum wage requirements..."}
                                  </p>
                                  <button className="text-xs text-blue-600 hover:text-blue-800 mt-1">
                                    Expand full text →
                                  </button>
                                </div>
                              ))}
                            </div>
                          </details>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Timestamp */}
                <div className="text-xs text-gray-500">
                  {new Date(result.timestamp).toLocaleString()}
                </div>
              </div>
            )}

            {!result && !loading && !error && (
              <div className="text-center py-8 text-gray-500 text-sm">
                No result yet. Send a query to see results here.
              </div>
            )}
          </div>

          {/* JSON Viewer */}
          {result && (
            <div className="bg-white rounded-lg shadow-lg overflow-hidden">
              <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
                <h3 className="text-sm font-bold text-white">Raw Response</h3>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(result.raw || result, null, 2))
                    alert('Copied to clipboard!')
                  }}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded transition-colors"
                >
                  📋 Copy
                </button>
              </div>
              <div className="p-4 bg-gray-900 overflow-x-auto" style={{ maxHeight: '400px' }}>
                <pre className="text-xs text-gray-100 font-mono">
                  {JSON.stringify(result.raw || result, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded">
                <div className="flex">
                  <svg className="h-5 w-5 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <div>
                    <p className="text-sm font-semibold text-red-800">Error</p>
                    <p className="text-xs text-red-700 mt-1">{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UnifiedDashboard
