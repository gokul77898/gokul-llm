# Vakeels.AI UI - Frontend Interface

Beautiful React + Vite frontend for testing the Vakeels.AI legal intelligence platform.

## Quick Start

### 1. Install Dependencies
```bash
cd ui
npm install
```

### 2. Start Backend
Make sure your FastAPI backend is running on port 8000:
```bash
cd ..
python -m src.api.main --host 127.0.0.1 --port 8000
```

### 3. Start Frontend
```bash
npm run dev
```

The UI will be available at: **http://localhost:3000**

## Features

✅ Clean, modern interface with Tailwind CSS  
✅ Real-time query testing  
✅ Model selection (Mamba, Transformer, RL Trained)  
✅ Configurable top-k retrieval  
✅ Confidence scoring visualization  
✅ Collapsible JSON viewer  
✅ Loading states & error handling  
✅ Responsive design  

## Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Styling
- **Axios** - API calls

## Project Structure

```
ui/
├── src/
│   ├── App.jsx              # Main app component
│   ├── main.jsx             # Entry point
│   ├── api.js               # API client
│   ├── components/
│   │   ├── QueryForm.jsx    # Query input form
│   │   ├── ResultCard.jsx   # Results display
│   │   └── JSONViewer.jsx   # JSON response viewer
│   └── index.css            # Global styles
├── index.html               # HTML template
├── package.json             # Dependencies
├── vite.config.js           # Vite configuration
├── tailwind.config.js       # Tailwind config
└── postcss.config.js        # PostCSS config
```

## API Connection

The UI connects to your FastAPI backend at `http://localhost:8000/query`

Endpoint used:
```
POST /query
{
  "query": "What is contract law?",
  "model": "mamba",
  "top_k": 5
}
```

## Build for Production

```bash
npm run build
```

Built files will be in `dist/` directory.

## Troubleshooting

**Backend connection failed?**
- Ensure FastAPI is running on port 8000
- Check CORS is enabled in backend
- Verify firewall settings

**Port 3000 already in use?**
- Change port in `vite.config.js` under `server.port`

**Styles not loading?**
- Run `npm install` again
- Clear browser cache
- Check Tailwind config

## Screenshots

The UI includes:
- Modern gradient header
- Clean query form with model selection
- Beautiful result cards with confidence scores
- Collapsible JSON viewer
- Responsive layout

Enjoy testing your Vakeels.AI system! 🚀
