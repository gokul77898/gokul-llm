# Phase 5: Evaluation & Metrics Dashboard

## Overview

Interactive evaluation dashboard for testing and comparing all MARK models with real-time metrics visualization.

## Features

### 🎯 Core Functionality

1. **Model Selection**
   - Mamba (Hierarchical Attention)
   - Transformer (BERT-based)
   - RL Trained (PPO Optimized)

2. **Query Interface**
   - Single-line text input
   - Model dropdown selector
   - Top-K documents configuration (1-20)
   - Enter key support for quick queries

3. **Real-time Metrics**
   - **Confidence Score**: Circular gauge (0-100%)
   - **Retrieved Documents**: Count display
   - **Query Latency**: Milliseconds tracking
   - **Latency History**: Line chart showing performance over time

4. **Results Display**
   - Generated answer with formatting
   - Metadata footer (confidence, docs, latency)
   - Query details section
   - Timestamp tracking

### 📊 Visualizations

#### Confidence Gauge
- Circular progress indicator
- Color-coded:
  - Green: ≥70% (High confidence)
  - Yellow: 40-70% (Medium confidence)
  - Red: <40% (Low confidence)

#### Latency Chart
- Line graph showing response times
- X-axis: Timestamp
- Y-axis: Latency in milliseconds
- Tracks last 10 queries
- Built with Recharts library

#### Metrics Cards
- Large number displays
- Clean, modern design
- Real-time updates

## Tech Stack

- **React 18**: UI framework
- **Recharts 2.10**: Chart visualizations
- **Tailwind CSS**: Styling
- **Axios**: API calls

## Usage

### Access Dashboard

```
http://localhost:3000
```

Click **"Evaluation"** tab to open dashboard

### Running Queries

1. Enter question in text field
2. Select model from dropdown
3. Adjust top-K if needed (default: 5)
4. Click **"Run Query"** or press Enter
5. View results and metrics

### Example Queries

```
"What is contract law?"
"Explain breach of contract"
"What are the elements of negligence?"
"Define intellectual property rights"
```

## Files Created/Modified

### New Files
- `ui/src/components/EvaluationDashboard.jsx` - Main dashboard component

### Modified Files
- `ui/package.json` - Added recharts dependency
- `ui/src/App.jsx` - Added evaluation mode toggle

## Components Breakdown

### EvaluationDashboard.jsx

```javascript
State Management:
- query: User input
- model: Selected model
- topK: Number of documents
- result: Query response
- latencyHistory: Performance tracking
- loading/error: UI states

Key Functions:
- handleRunQuery(): Execute query and measure latency
- Performance tracking with performance.now()
- Auto-update latency history (max 10 entries)
```

## Metrics Explained

### Confidence Score
- **Range**: 0-100%
- **Source**: Backend's confidence calculation
- **Meaning**: How confident the model is in its answer
- **Threshold**: >70% is good, <40% may need review

### Retrieved Documents
- **Shows**: Actual docs found vs requested
- **Example**: "5 out of 5 requested" = Full retrieval
- **Note**: May be 0 if document store is empty

### Latency
- **Measurement**: Round-trip time (ms)
- **Includes**: Network + processing + generation
- **Typical**: 1000-3000ms for full pipeline
- **Factors**: Model complexity, document count, network speed

## API Integration

```javascript
// Query format
{
  "query": "What is contract law?",
  "model": "rl_trained",
  "top_k": 5
}

// Response format
{
  "answer": "Generated response...",
  "query": "What is contract law?",
  "model": "rl_trained",
  "retrieved_docs": 5,
  "confidence": 0.85,
  "timestamp": "2025-11-16T..."
}
```

## Performance Tracking

### Latency History
- Stores last 10 queries
- Updates automatically
- Visualized as line chart
- Timestamp format: HH:MM:SS

### Use Cases
1. **Model Comparison**: Run same query across models
2. **Performance Testing**: Track latency trends
3. **Quality Assessment**: Monitor confidence scores
4. **Debug Tool**: Identify slow queries

## Navigation

### Three Modes

1. **Query Mode**: Single Q&A with full JSON
2. **Chat Mode**: Conversation with history
3. **Evaluation**: Metrics dashboard ← **NEW**

Toggle between modes using top navigation buttons.

## Installation

```bash
# Install dependencies (includes recharts)
cd ui && npm install

# Start dev server
npm run dev

# Access at http://localhost:3000
```

## Screenshots

### Dashboard Layout
```
┌─────────────────────────────────────┐
│  MARK Evaluation Dashboard 🚀       │
├─────────────────────────────────────┤
│  Query Input: [___________]         │
│  Model: [RL Trained ▼]  TopK: [5]  │
│  [Run Query Button]                 │
├─────────────────────────────────────┤
│  Answer Card                        │
│  "Generated response..."            │
│  Confidence: 85% | Docs: 5 | 1250ms│
├─────────────────────────────────────┤
│  ┌────────┐ ┌────────┐ ┌────────┐ │
│  │ 85%    │ │   5    │ │ 1250   │ │
│  │Confiden│ │  Docs  │ │Latency │ │
│  └────────┘ └────────┘ └────────┘ │
├─────────────────────────────────────┤
│  Latency Over Time                  │
│  [Line Chart]                       │
└─────────────────────────────────────┘
```

## Future Enhancements

- [ ] Model comparison mode (side-by-side)
- [ ] Export metrics to CSV
- [ ] Advanced filtering options
- [ ] Historical query log
- [ ] A/B testing support
- [ ] Custom metric thresholds
- [ ] Real-time streaming visualizations

## Troubleshooting

**Charts not showing?**
- Verify recharts installed: `npm list recharts`
- Check browser console for errors
- Clear cache and reload

**Latency seems high?**
- Check backend is running
- Verify network connection
- Consider reducing top_k

**Confidence always low?**
- Document store may be empty
- Try different models
- Check retrieval results

## Status

✅ **Complete and Deployed**
- All metrics working
- Charts rendering correctly
- Real-time updates functional
- Backend integration stable

---

**Ready for evaluation and testing!** 🚀
