import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, Database, Cpu, TrendingUp } from 'lucide-react';

const MonitoringDashboard = () => {
  const [systemHealth, setSystemHealth] = useState(null);
  const [chromaStats, setChromaStats] = useState(null);
  const [modelSelectionLog, setModelSelectionLog] = useState([]);

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
    const interval = setInterval(fetchStats, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  // Fetch model selection log
  useEffect(() => {
    const fetchLog = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/model_selector/log?limit=20');
        const data = await response.json();
        setModelSelectionLog(data.selections || []);
      } catch (error) {
        console.error('Failed to fetch model selection log:', error);
      }
    };

    fetchLog();
    const interval = setInterval(fetchLog, 15000); // Refresh every 15s
    return () => clearInterval(interval);
  }, []);

  const StatusBadge = ({ status }) => {
    const isOk = status === 'ok' || status === 'ready' || status === 'initialized';
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
        isOk ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
      }`}>
        {status}
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            System Monitoring Dashboard
          </h1>
          <p className="text-gray-600">Real-time system health and performance metrics</p>
          <p className="text-sm text-gray-500 mt-2">
            💬 Use <strong>Chat</strong> interface to send queries and get answers
          </p>
        </div>

        {/* System Health Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-600">ChromaDB</h3>
              <Database className="w-6 h-6 text-blue-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.chroma} />}
            <p className="text-2xl font-bold text-gray-900 mt-3">
              {chromaStats?.document_count || 0}
            </p>
            <p className="text-xs text-gray-500">Documents Indexed</p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-600">Model Selector</h3>
              <Cpu className="w-6 h-6 text-purple-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.model_selector} />}
            <p className="text-2xl font-bold text-gray-900 mt-3">
              {modelSelectionLog.length}
            </p>
            <p className="text-xs text-gray-500">Recent Selections</p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-600">Pipeline</h3>
              <Activity className="w-6 h-6 text-green-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.pipeline} />}
            <p className="text-2xl font-bold text-gray-900 mt-3">
              {systemHealth?.data_loaded ? 'Loaded' : 'Empty'}
            </p>
            <p className="text-xs text-gray-500">Data Status</p>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-gray-600">Training</h3>
              <TrendingUp className="w-6 h-6 text-orange-600" />
            </div>
            {systemHealth && <StatusBadge status={systemHealth.training} />}
            <p className="text-2xl font-bold text-gray-900 mt-3">
              Setup Mode
            </p>
            <p className="text-xs text-gray-500">Training Disabled</p>
          </div>
        </div>

        {/* ChromaDB Details */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold mb-4 text-gray-900">ChromaDB Statistics</h2>
            {chromaStats ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Collection</span>
                  <span className="font-semibold text-gray-900">{chromaStats.collection}</span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Documents</span>
                  <span className="font-semibold text-gray-900">{chromaStats.document_count}</span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Embedding Dimension</span>
                  <span className="font-semibold text-gray-900">{chromaStats.embedding_dimension}</span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Status</span>
                  <StatusBadge status={chromaStats.status} />
                </div>
                <div className="pt-2">
                  <p className="text-xs text-gray-500">Storage: {chromaStats.storage_path}</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">Loading...</p>
            )}
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold mb-4 text-gray-900">System Health</h2>
            {systemHealth ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">ChromaDB</span>
                  <StatusBadge status={systemHealth.chroma} />
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Retriever</span>
                  <StatusBadge status={systemHealth.retriever} />
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Model Selector</span>
                  <StatusBadge status={systemHealth.model_selector} />
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Pipeline</span>
                  <StatusBadge status={systemHealth.pipeline} />
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-gray-200">
                  <span className="text-sm text-gray-600">Training</span>
                  <StatusBadge status={systemHealth.training} />
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Data Loaded</span>
                  <span className={`font-semibold ${systemHealth.data_loaded ? 'text-green-600' : 'text-orange-600'}`}>
                    {systemHealth.data_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-gray-500">Loading...</p>
            )}
          </div>
        </div>

        {/* Model Selection Log */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4 text-gray-900">Recent Model Selections</h2>
          {modelSelectionLog.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-xs font-semibold text-gray-600">Query</th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-gray-600">Model Selected</th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-gray-600">Reason</th>
                    <th className="text-left py-3 px-4 text-xs font-semibold text-gray-600">Complexity</th>
                  </tr>
                </thead>
                <tbody>
                  {modelSelectionLog.slice(0, 10).map((log, idx) => (
                    <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 text-sm text-gray-700 max-w-xs truncate">
                        {log.query}
                      </td>
                      <td className="py-3 px-4">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          log.model === 'rl_trained' 
                            ? 'bg-blue-100 text-blue-800' 
                            : 'bg-purple-100 text-purple-800'
                        }`}>
                          {log.model}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600 max-w-md truncate">
                        {log.reason}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">
                        {log.analysis?.complexity?.toLowerCase() || 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p className="text-sm">No model selections yet</p>
              <p className="text-xs mt-1">Start chatting to see model selection activity</p>
            </div>
          )}
        </div>

        {/* Info Banner */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start">
            <svg className="w-5 h-5 text-blue-600 mr-3 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="text-sm font-semibold text-blue-800">Monitoring Dashboard</p>
              <p className="text-xs text-blue-700 mt-1">
                This dashboard shows real-time system metrics and health. 
                To send queries and chat with the AI, click the <strong>💬 Chat</strong> button at the top-right.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MonitoringDashboard;
