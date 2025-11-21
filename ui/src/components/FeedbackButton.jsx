import React, { useState } from 'react';

const FeedbackButton = ({ query, answer, modelUsed, onFeedbackSubmit }) => {
  const [showFeedbackForm, setShowFeedbackForm] = useState(false);
  const [correctAnswer, setCorrectAnswer] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleIncorrect = async () => {
    setSubmitting(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          wrong_answer: answer,
          correct_answer: correctAnswer || null,
          model_used: modelUsed
        })
      });
      
      if (response.ok) {
        alert('Feedback submitted! Model will improve.');
        setShowFeedbackForm(false);
        setCorrectAnswer('');
        if (onFeedbackSubmit) onFeedbackSubmit();
      }
    } catch (error) {
      alert('Failed to submit feedback');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="feedback-section mt-2">
      {!showFeedbackForm ? (
        <button
          onClick={() => setShowFeedbackForm(true)}
          className="px-3 py-1 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded"
        >
          ❌ Incorrect Answer
        </button>
      ) : (
        <div className="p-3 bg-gray-50 rounded mt-2">
          <p className="text-sm font-medium mb-2">Help improve the model:</p>
          <textarea
            value={correctAnswer}
            onChange={(e) => setCorrectAnswer(e.target.value)}
            placeholder="Enter correct answer (optional)"
            className="w-full px-2 py-1 border rounded text-sm"
            rows="3"
          />
          <div className="flex gap-2 mt-2">
            <button
              onClick={handleIncorrect}
              disabled={submitting}
              className="px-3 py-1 bg-blue-600 text-white rounded text-sm"
            >
              {submitting ? 'Submitting...' : 'Submit Feedback'}
            </button>
            <button
              onClick={() => setShowFeedbackForm(false)}
              className="px-3 py-1 bg-gray-300 rounded text-sm"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackButton;
