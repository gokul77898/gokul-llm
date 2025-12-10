import React from "react";

export default function ResultCard({ result }) {
  if (!result) return null;

  // Normalize incoming backend response
  const { type, text, model, confidence, label, entities } = result.answer || result;

  return (
    <div className="p-4 border rounded-lg shadow bg-white my-3">
      <h3 className="text-lg font-bold mb-2">Model: {model}</h3>

      {/* GENERATION RESULT */}
      {type === "generation" && (
        <p className="whitespace-pre-wrap text-gray-800">
          {text}
        </p>
      )}

      {/* CLASSIFICATION */}
      {type === "classification" && (
        <div>
          <p className="text-gray-800">
            <strong>Predicted Label:</strong> {label}
          </p>
          <p className="text-gray-600">
            <strong>Confidence:</strong> {(confidence * 100).toFixed(1)}%
          </p>
        </div>
      )}

      {/* NER */}
      {type === "ner" && (
        <div>
          <strong>Named Entities:</strong>
          <ul className="mt-2">
            {entities && entities.map((e, i) => (
              <li key={i} className="text-gray-800">
                <span className="font-mono bg-gray-200 p-1 rounded">{e.token}</span>
                {" → "}
                <span className="font-bold">{e.label}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* FALLBACK - Show raw answer if no specific type */}
      {(!type || type === "unknown") && (
        <div>
          <p className="text-gray-800 whitespace-pre-wrap">
            {result.answer || text || "No response available"}
          </p>
          {result.confidence && (
            <p className="text-gray-600 mt-2">
              <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
            </p>
          )}
        </div>
      )}

      {/* Metadata */}
      {result.metadata && (
        <div className="mt-3 text-sm text-gray-500 border-t pt-2">
          <p><strong>Expert:</strong> {result.metadata.expert_used}</p>
          {result.metadata.task_type && (
            <p><strong>Task:</strong> {result.metadata.task_type}</p>
          )}
          {result.metadata.routing_score && (
            <p><strong>Routing Score:</strong> {result.metadata.routing_score}</p>
          )}
        </div>
      )}
    </div>
  );
}
