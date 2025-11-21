import { useState } from 'react'

const QueryForm = ({ onSubmit, loading }) => {
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('mamba')
  const [topK, setTopK] = useState(5)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      onSubmit({ query, model, top_k: topK })
    }
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Query Configuration</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Query Input */}
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Your Question
          </label>
          <textarea
            id="query"
            rows={4}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
            placeholder="Enter your legal question here... e.g., What is contract law?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
        </div>

        {/* Model Selection */}
        <div>
          <label htmlFor="model" className="block text-sm font-medium text-gray-700 mb-2">
            Model Selection
          </label>
          <select
            id="model"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all bg-white"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={loading}
          >
            <option value="mamba">Mamba (Hierarchical Attention)</option>
            <option value="transformer">Transformer (BERT-based)</option>
            <option value="rl_trained">RL Optimized (PPO fine-tuned)</option>
          </select>
          <p className="mt-2 text-xs text-gray-500">
            {model === 'mamba' && '🔹 Best for long documents with hierarchical structure'}
            {model === 'transformer' && '🔹 Best for classification and NER tasks'}
            {model === 'rl_trained' && '🔹 RLHF-optimized for human-aligned responses. Uses policy checkpoint from checkpoints/rlhf/ppo/ppo_final.pt'}
          </p>
        </div>

        {/* Top-K Input */}
        <div>
          <label htmlFor="topk" className="block text-sm font-medium text-gray-700 mb-2">
            Top-K Documents
          </label>
          <input
            id="topk"
            type="number"
            min="1"
            max="20"
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value))}
            disabled={loading}
          />
          <p className="mt-2 text-xs text-gray-500">Number of documents to retrieve (1-20)</p>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            'Run Query'
          )}
        </button>
      </form>
    </div>
  )
}

export default QueryForm
