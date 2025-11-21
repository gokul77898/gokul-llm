import React, { useState } from 'react';
import { X, Check, AlertCircle } from 'lucide-react';

const ImprovedFeedbackButton = ({ query, answer, model, sources, onSubmit }) => {
  const [showModal, setShowModal] = useState(false);
  const [correctedAnswer, setCorrectedAnswer] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = async () => {
    const feedbackData = {
      query,
      answer,
      model,
      auto_model_used: model,
      sources: sources || [],
      user_corrected_answer: correctedAnswer.trim(),
      flagged_incorrect: true,
      timestamp: new Date().toISOString()
    };

    try {
      const response = await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
      });

      if (response.ok) {
        setSubmitted(true);
        setTimeout(() => {
          setShowModal(false);
          setSubmitted(false);
          setCorrectedAnswer('');
        }, 2000);
        
        if (onSubmit) onSubmit(feedbackData);
      }
    } catch (error) {
      console.error('Feedback submission failed:', error);
    }
  };

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="feedback-button"
        title="Mark this answer as incorrect"
      >
        <X size={14} />
        <span>Mark Incorrect</span>
      </button>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            {!submitted ? (
              <>
                <h3>
                  <AlertCircle size={20} />
                  Report Incorrect Answer
                </h3>
                <p className="modal-info">
                  Your feedback helps improve the model. The system won't auto-deploy 
                  changes - all corrections go to a training buffer for safe review.
                </p>
                
                <div className="form-group">
                  <label>Original Question:</label>
                  <div className="readonly-field">{query}</div>
                </div>
                
                <div className="form-group">
                  <label>System's Answer:</label>
                  <div className="readonly-field">{answer.substring(0, 200)}...</div>
                </div>
                
                <div className="form-group">
                  <label>Provide Correct Answer (optional):</label>
                  <textarea
                    value={correctedAnswer}
                    onChange={(e) => setCorrectedAnswer(e.target.value)}
                    placeholder="Enter the correct answer here, or leave blank to just flag as incorrect..."
                    rows={4}
                  />
                </div>
                
                <div className="modal-actions">
                  <button onClick={() => setShowModal(false)} className="btn-cancel">
                    Cancel
                  </button>
                  <button onClick={handleSubmit} className="btn-submit">
                    Submit Feedback
                  </button>
                </div>
              </>
            ) : (
              <div className="success-message">
                <Check size={48} color="#10b981" />
                <h3>Feedback Submitted!</h3>
                <p>Thank you for helping improve the model.</p>
              </div>
            )}
          </div>
        </div>
      )}

      <style jsx>{`
        .feedback-button {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          background: #fee2e2;
          color: #dc2626;
          border: 1px solid #fca5a5;
          border-radius: 6px;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .feedback-button:hover {
          background: #fca5a5;
          color: #991b1b;
        }
        
        .modal-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.5);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }
        
        .modal-content {
          background: white;
          padding: 24px;
          border-radius: 12px;
          max-width: 600px;
          width: 90%;
          max-height: 80vh;
          overflow-y: auto;
        }
        
        .modal-content h3 {
          display: flex;
          align-items: center;
          gap: 8px;
          margin: 0 0 16px 0;
          color: #dc2626;
        }
        
        .modal-info {
          font-size: 14px;
          color: #6b7280;
          margin-bottom: 20px;
          line-height: 1.5;
        }
        
        .form-group {
          margin-bottom: 16px;
        }
        
        .form-group label {
          display: block;
          font-weight: 600;
          margin-bottom: 6px;
          font-size: 14px;
        }
        
        .readonly-field {
          background: #f3f4f6;
          padding: 10px;
          border-radius: 6px;
          font-size: 13px;
          color: #374151;
          max-height: 100px;
          overflow-y: auto;
        }
        
        textarea {
          width: 100%;
          padding: 10px;
          border: 1px solid #d1d5db;
          border-radius: 6px;
          font-size: 14px;
          font-family: inherit;
          resize: vertical;
        }
        
        textarea:focus {
          outline: none;
          border-color: #3b82f6;
        }
        
        .modal-actions {
          display: flex;
          gap: 12px;
          justify-content: flex-end;
          margin-top: 20px;
        }
        
        .btn-cancel, .btn-submit {
          padding: 8px 16px;
          border-radius: 6px;
          font-size: 14px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .btn-cancel {
          background: #f3f4f6;
          border: 1px solid #d1d5db;
          color: #374151;
        }
        
        .btn-cancel:hover {
          background: #e5e7eb;
        }
        
        .btn-submit {
          background: #dc2626;
          border: none;
          color: white;
        }
        
        .btn-submit:hover {
          background: #b91c1c;
        }
        
        .success-message {
          text-align: center;
          padding: 20px;
        }
        
        .success-message h3 {
          color: #10b981;
          margin: 16px 0 8px 0;
        }
        
        .success-message p {
          color: #6b7280;
        }
      `}</style>
    </>
  );
};

export default ImprovedFeedbackButton;
