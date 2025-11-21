import React, { useState, useEffect } from 'react';
import { Activity, Database, Cpu, AlertCircle, CheckCircle, XCircle } from 'lucide-react';

const AdminDashboard = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [chromaStats, setChromaStats] = useState(null);
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [queryInput, setQueryInput] = useState('');
  const [queryResults, setQueryResults] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch system health
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/system/health');
        const data = await response.json();
        setSystemHealth(data);
      } catch (error) {
        console.error('Failed to fetch health:', error);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  // Fetch ChromaDB stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/chroma/stats');
        const data = await response.json();
        setChromaStats(data);
      } catch (error) {
        console.error('Failed to fetch chroma stats:', error);
      }
    };

    fetchStats();
  }, []);

  // Fetch training status
  useEffect(() => {
    const fetchTraining = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/training/status');
        const data = await response.json();
        setTrainingStatus(data);
      } catch (error) {
        console.error('Failed to fetch training status:', error);
      }
    };

    fetchTraining();
  }, []);

  const testQuery = async () => {
    if (!queryInput.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryInput, top_k: 5 })
      });
      const data = await response.json();
      setQueryResults(data);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const StatusBadge = ({ status }) => {
    const isOk = status === 'ok' || status === 'ready' || status === 'initialized';
    return (
      <span className={`px-2 py-1 rounded text-xs font-medium ${
        isOk ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
      }`}>
        {isOk ? <CheckCircle className="inline w-3 h-3 mr-1" /> : <XCircle className="inline w-3 h-3 mr-1" />}
        {status}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Admin Dashboard</h1>

        {/* System Health */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">ChromaDB</h3>
              <Database className="w-6 h-6 text-blue-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.chroma} />}
            <p className="text-sm text-gray-600 mt-2">Vector Database</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">Model Selector</h3>
              <Cpu className="w-6 h-6 text-purple-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.model_selector} />}
            <p className="text-sm text-gray-600 mt-2">Auto Model Selection</p>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold">Pipeline</h3>
              <Activity className="w-6 h-6 text-green-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.pipeline} />}
            <p className="text-sm text-gray-600 mt-2">AutoPipeline Ready</p>
          </div>
        </div>

        {/* ChromaDB Stats */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">ChromaDB Statistics</h2>
          {chromaStats ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600">Collection</p>
                <p className="text-2xl font-bold">{chromaStats.collection}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Documents</p>
                <p className="text-2xl font-bold">{chromaStats.document_count}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Dimension</p>
                <p className="text-2xl font-bold">{chromaStats.embedding_dimension}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Status</p>
                <StatusBadge status={chromaStats.status} />
              </div>
            </div>
          ) : (
            <p>Loading...</p>
          )}
          <p className="text-xs text-gray-500 mt-4">Storage: {chromaStats?.storage_path}</p>
        </div>

        {/* Query Test Tool */}
        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Query Test Tool</h2>
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder="Enter test query..."
              className="flex-1 px-4 py-2 border rounded-lg"
              onKeyPress={(e) => e.key === 'Enter' && testQuery()}
            />
            <button
              onClick={testQuery}
              disabled={loading || !queryInput.trim()}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Testing...' : 'Test Query'}
            </button>
          </div>

          {queryResults && (
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Results ({queryResults.retrieved_docs} docs retrieved)</h3>
              <div className="bg-gray-50 p-4 rounded">
                <p><strong>Model Used:</strong> {queryResults.auto_model_used}</p>
                <p><strong>Confidence:</strong> {(queryResults.confidence * 100).toFixed(1)}%</p>
                <p><strong>Latency:</strong> {queryResults.latency.toFixed(0)}ms</p>
                {queryResults.sources && queryResults.sources.length > 0 && (
                  <div className="mt-3">
                    <p className="font-semibold">Top Sources:</p>
                    {queryResults.sources.slice(0, 3).map((source, i) => (
                      <div key={i} className="mt-2 p-2 bg-white rounded text-sm">
                        <p className="text-xs text-gray-600">Score: {source.score?.toFixed(3) || 'N/A'}</p>
                        <p className="truncate">{source.content?.slice(0, 150) || source.text?.slice(0, 150)}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Training Console */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-bold mb-4">Training Console</h2>
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex items-start">
              <AlertCircle className="w-5 h-5 text-yellow-600 mr-2 mt-0.5" />
              <div>
                <p className="font-semibold text-yellow-800">Training Disabled - Setup Mode</p>
                <p className="text-sm text-yellow-700">All training modules are in skeleton mode only.</p>
              </div>
            </div>
          </div>

          {trainingStatus && (
            <div className="space-y-3">
              <div className="border rounded p-3">
                <p className="font-semibold">SFT Training</p>
                <p className="text-sm text-gray-600">Status: {trainingStatus.sft?.status || 'N/A'}</p>
                <p className="text-xs text-gray-500">{trainingStatus.sft?.message}</p>
              </div>
              <div className="border rounded p-3">
                <p className="font-semibold">RL Training</p>
                <p className="text-sm text-gray-600">Status: {trainingStatus.rl?.status || 'N/A'}</p>
                <p className="text-xs text-gray-500">{trainingStatus.rl?.message}</p>
              </div>
              <div className="border rounded p-3">
                <p className="font-semibold">RLHF Training</p>
                <p className="text-sm text-gray-600">Status: {trainingStatus.rlhf?.status || 'N/A'}</p>
                <p className="text-xs text-gray-500">{trainingStatus.rlhf?.message}</p>
              </div>
            </div>
          )}

          <div className="mt-4 flex gap-2">
            <button disabled className="px-4 py-2 bg-gray-300 text-gray-600 rounded cursor-not-allowed">
              Start SFT
            </button>
            <button disabled className="px-4 py-2 bg-gray-300 text-gray-600 rounded cursor-not-allowed">
              Start RL
            </button>
            <button disabled className="px-4 py-2 bg-gray-300 text-gray-600 rounded cursor-not-allowed">
              Start RLHF
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-2">Training buttons are disabled in setup mode</p>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
