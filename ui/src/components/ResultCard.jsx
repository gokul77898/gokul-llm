const ResultCard = ({ result }) => {
  const confidenceColor = result.confidence >= 0.7 
    ? 'text-green-600' 
    : result.confidence >= 0.4 
    ? 'text-yellow-600' 
    : 'text-red-600'

  const confidenceLabel = result.confidence >= 0.7 
    ? 'High' 
    : result.confidence >= 0.4 
    ? 'Medium' 
    : 'Low'

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-4">
        <h3 className="text-xl font-bold text-white">Query Results</h3>
      </div>

      <div className="p-6 space-y-6">
        {/* Answer Section */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">Answer</h4>
            <span className={`text-xs font-medium ${confidenceColor}`}>
              {confidenceLabel} Confidence
            </span>
          </div>
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            {result.answer ? (
              <p className="text-gray-800 leading-relaxed">{result.answer}</p>
            ) : (
              <p className="text-gray-500 italic">No answer generated</p>
            )}
          </div>
        </div>

        {/* Metadata Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-blue-50 rounded-lg p-4 border border-blue-100">
            <p className="text-xs font-medium text-blue-600 mb-1">Confidence Score</p>
            <p className="text-2xl font-bold text-blue-900">
              {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4 border border-purple-100">
            <p className="text-xs font-medium text-purple-600 mb-1">Documents Retrieved</p>
            <p className="text-2xl font-bold text-purple-900">{result.retrieved_docs}</p>
          </div>
        </div>

        {/* Model Info */}
        <div className="flex items-center space-x-2 text-sm text-gray-600 bg-gray-50 rounded-lg p-3">
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
          </svg>
          <span>Model: <span className="font-semibold">{result.model}</span></span>
          <span className="text-gray-400">•</span>
          <span>Query: <span className="font-semibold">{result.query?.substring(0, 30)}...</span></span>
        </div>

        {/* Timestamp */}
        {result.timestamp && (
          <div className="text-xs text-gray-500 border-t pt-3">
            Generated at: {new Date(result.timestamp).toLocaleString()}
          </div>
        )}
      </div>
    </div>
  )
}

export default ResultCard
