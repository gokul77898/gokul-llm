import { useState } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { queryAPI } from '../api'

const EvaluationDashboard = () => {
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('rl_trained')
  const [topK, setTopK] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [latencyHistory, setLatencyHistory] = useState([])
  const [error, setError] = useState(null)

  const handleRunQuery = async () => {
    if (!query.trim()) return

    setLoading(true)
    setError(null)
    const startTime = performance.now()

    try {
      const response = await queryAPI({ query, model, top_k: topK })
      const latency = Math.round(performance.now() - startTime)
      
      setResult({ ...response, latency })
      
      // Add to latency history
      const timestamp = new Date().toLocaleTimeString()
      setLatencyHistory(prev => [...prev.slice(-9), { time: timestamp, latency }])
    } catch (err) {
      setError(err.message || 'Query failed')
    } finally {
      setLoading(false)
    }
  }

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0
  const confidenceColor = result 
    ? result.confidence >= 0.7 ? 'text-green-600' : result.confidence >= 0.4 ? 'text-yellow-600' : 'text-red-600'
    : 'text-gray-600'

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
          Vakeels.AI Evaluation Dashboard 🚀
        </h1>
        <p className="text-gray-600">Interactive evaluation for all trained models</p>
      </div>

      {/* Query Section */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Query & Model Selection</h2>
        
        <div className="space-y-4">
          {/* Query Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Question
            </label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Type a legal question here..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={loading}
              onKeyPress={(e) => e.key === 'Enter' && handleRunQuery()}
            />
          </div>

          {/* Model and Top-K Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select Model
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 bg-white"
                disabled={loading}
              >
                <option value="auto">Auto (MoE Router)</option>
                <option value="inlegalbert">InLegalBERT (NER, Classification)</option>
                <option value="incaselawbert">InCaseLawBERT (Case Analysis)</option>
                <option value="indicbert">IndicBERT (Semantic Analysis)</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                {selectedModel === 'auto' && '🤖 Automatically routes to the best expert based on your query'}
                {selectedModel === 'inlegalbert' && '🔹 Best for named entity recognition and legal classification'}
                {selectedModel === 'incaselawbert' && '🔹 Specialized for case law analysis and legal reasoning'}
                {selectedModel === 'indicbert' && '🔹 Optimized for semantic analysis and question answering'}
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Top-K Documents
              </label>
              <input
                type="number"
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value))}
                min="1"
                max="20"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                disabled={loading}
              />
            </div>
          </div>

          {/* Run Button */}
          <button
            onClick={handleRunQuery}
            disabled={loading || !query.trim()}
            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
          >
            {loading ? 'Processing...' : 'Run Query'}
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4 mb-6">
          <p className="text-red-700 font-medium">{error}</p>
        </div>
      )}

      {/* Results Section */}
      {result && (
        <>
          {/* Answer Card */}
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-bold text-gray-900 mb-3">Answer</h3>
            <div className="bg-gray-50 rounded-lg p-4 border border-gray-200 mb-4">
              <p className="text-gray-800 leading-relaxed">
                {result.answer || '<No answer generated>'}
              </p>
            </div>
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>Confidence: <span className={`font-bold ${confidenceColor}`}>{confidencePercent}%</span></span>
              <span>Retrieved Docs: <span className="font-bold">{result.retrieved_docs}</span></span>
              <span>Latency: <span className="font-bold">{result.latency}ms</span></span>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            {/* Confidence Gauge */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Confidence Score</h3>
              <div className="flex items-center justify-center">
                <div className="relative w-40 h-40">
                  <svg className="transform -rotate-90 w-40 h-40">
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="transparent"
                      className="text-gray-200"
                    />
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="currentColor"
                      strokeWidth="12"
                      fill="transparent"
                      strokeDasharray={`${2 * Math.PI * 70}`}
                      strokeDashoffset={`${2 * Math.PI * 70 * (1 - result.confidence)}`}
                      className={result.confidence >= 0.7 ? 'text-green-500' : result.confidence >= 0.4 ? 'text-yellow-500' : 'text-red-500'}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className={`text-3xl font-bold ${confidenceColor}`}>
                      {confidencePercent}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Retrieved Docs Count */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Documents Retrieved</h3>
              <div className="flex items-center justify-center h-40">
                <div className="text-center">
                  <div className="text-6xl font-bold text-blue-600">{result.retrieved_docs}</div>
                  <div className="text-sm text-gray-600 mt-2">out of {topK} requested</div>
                </div>
              </div>
            </div>

            {/* Latency */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Query Latency</h3>
              <div className="flex items-center justify-center h-40">
                <div className="text-center">
                  <div className="text-6xl font-bold text-purple-600">{result.latency}</div>
                  <div className="text-sm text-gray-600 mt-2">milliseconds</div>
                </div>
              </div>
            </div>
          </div>

          {/* Latency Chart */}
          {latencyHistory.length > 0 && (
            <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">Latency Over Time</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={latencyHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="latency" stroke="#8b5cf6" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Model Info */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Query Details</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Model:</span>
                <span className="font-semibold">{result.model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Query:</span>
                <span className="font-semibold">{result.query}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Timestamp:</span>
                <span className="font-semibold">{new Date(result.timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </>
      )}

      {/* No Results Placeholder */}
      {!result && !loading && !error && (
        <div className="bg-white rounded-lg shadow-lg p-12 text-center">
          <svg className="mx-auto h-16 w-16 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900">No evaluation results yet</h3>
          <p className="mt-2 text-sm text-gray-500">Run a query to see metrics and performance data</p>
        </div>
      )}
    </div>
  )
}

export default EvaluationDashboard
