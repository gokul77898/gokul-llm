import React, { useState } from "react";
import { runMoE } from "../api";

export default function MOETester() {
  const [text, setText] = useState("");
  const [task, setTask] = useState("qa");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    if (!text.trim()) {
      setResult({ error: "Please enter some text to analyze" });
      return;
    }

    setLoading(true);
    const startTime = Date.now();
    
    try {
      const res = await runMoE(text, task);
      const endTime = Date.now();
      const latency = endTime - startTime;
      
      setResult({
        ...res,
        latency_ms: latency
      });
    } catch (err) {
      setResult({ error: err.toString() });
    }
    setLoading(false);
  }

  const clearResults = () => {
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white shadow rounded-lg p-6">
          <h2 className="text-2xl font-bold mb-6 text-gray-800">🤖 MoE Testing Panel</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Legal Text or Question:
              </label>
              <textarea
                className="w-full border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={4}
                placeholder="Enter your legal question or text for analysis..."
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Task Type:
              </label>
              <select
                className="border border-gray-300 rounded-md p-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={task}
                onChange={(e) => setTask(e.target.value)}
              >
                <option value="qa">Question Answering</option>
                <option value="ner">Named Entity Recognition</option>
                <option value="classification">Classification</option>
                <option value="similarity">Similarity Analysis</option>
                <option value="case-classification">Case Classification</option>
                <option value="legal-reasoning">Legal Reasoning</option>
              </select>
            </div>

            <div className="flex gap-3">
              <button
                className={`px-6 py-2 rounded-md font-medium transition-colors ${
                  loading
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
                }`}
                onClick={handleRun}
                disabled={loading}
              >
                {loading ? "🔄 Processing..." : "🚀 Run MoE Analysis"}
              </button>

              {result && (
                <button
                  className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-md font-medium transition-colors"
                  onClick={clearResults}
                >
                  Clear Results
                </button>
              )}
            </div>
          </div>

          {result && (
            <div className="mt-6 bg-gray-50 rounded-lg p-4 border">
              {result.error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-4">
                  <h3 className="font-bold text-red-800 mb-2">❌ Error:</h3>
                  <p className="text-red-700">{result.error}</p>
                </div>
              )}

              {result.output && (
                <div className="space-y-4">
                  <div className="bg-green-50 border border-green-200 rounded-md p-4">
                    <h3 className="font-bold text-green-800 mb-2">📝 Model Output:</h3>
                    <div className="bg-white p-3 rounded border">
                      <pre className="whitespace-pre-wrap text-sm text-gray-800">
                        {result.output}
                      </pre>
                    </div>
                  </div>

                  <div className="grid md:grid-cols-2 gap-4">
                    {result.expert && (
                      <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                        <h3 className="font-bold text-blue-800 mb-2">🎯 Selected Expert:</h3>
                        <p className="text-blue-700 font-mono">{result.expert}</p>
                      </div>
                    )}

                    {result.latency_ms && (
                      <div className="bg-purple-50 border border-purple-200 rounded-md p-4">
                        <h3 className="font-bold text-purple-800 mb-2">⚡ Latency:</h3>
                        <p className="text-purple-700 font-mono">{result.latency_ms} ms</p>
                      </div>
                    )}
                  </div>

                  {result.routing_reason && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
                      <h3 className="font-bold text-yellow-800 mb-2">🧠 Routing Decision:</h3>
                      <p className="text-yellow-700 text-sm">{result.routing_reason}</p>
                    </div>
                  )}

                  {result.tokens && (
                    <div className="bg-indigo-50 border border-indigo-200 rounded-md p-4">
                      <h3 className="font-bold text-indigo-800 mb-2">📊 Token Count:</h3>
                      <p className="text-indigo-700">{result.tokens} tokens generated</p>
                    </div>
                  )}

                  {(result.metadata || result.router_metadata) && (
                    <div className="bg-gray-100 border border-gray-300 rounded-md p-4">
                      <h3 className="font-bold text-gray-800 mb-2">🔍 Routing Metadata:</h3>
                      <pre className="text-xs text-gray-600 overflow-auto">
                        {JSON.stringify(result.metadata || result.router_metadata, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
