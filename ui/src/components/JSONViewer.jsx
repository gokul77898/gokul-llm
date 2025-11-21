import { useState } from 'react'

const JSONViewer = ({ data }) => {
  const [isOpen, setIsOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="bg-white rounded-lg shadow-lg overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 bg-gray-50 hover:bg-gray-100 transition-colors flex items-center justify-between"
      >
        <div className="flex items-center space-x-2">
          <svg 
            className={`h-5 w-5 text-gray-600 transition-transform ${isOpen ? 'rotate-90' : ''}`} 
            fill="currentColor" 
            viewBox="0 0 20 20"
          >
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
          <h4 className="text-sm font-semibold text-gray-700">Raw JSON Response</h4>
        </div>
        <span className="text-xs text-gray-500">
          {isOpen ? 'Click to collapse' : 'Click to expand'}
        </span>
      </button>

      {/* JSON Content */}
      {isOpen && (
        <div className="p-6 border-t border-gray-200">
          <div className="relative">
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 px-3 py-1 bg-gray-800 hover:bg-gray-700 text-white text-xs rounded transition-colors"
            >
              {copied ? '✓ Copied!' : 'Copy'}
            </button>
            <pre className="bg-gray-900 text-gray-100 rounded-lg p-4 overflow-x-auto text-sm">
              <code>{JSON.stringify(data, null, 2)}</code>
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}

export default JSONViewer
