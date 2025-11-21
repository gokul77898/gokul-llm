#!/bin/bash
# Start MARK System (Backend + Frontend)

echo "=================================="
echo "  MARK AI System Startup"
echo "=================================="

# Kill existing processes
echo "Stopping existing processes..."
pkill -f "src.api.main" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend on port 8000..."
python3.10 -m src.api.main --host 127.0.0.1 --port 8000 > /tmp/mark_api.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check backend health
echo "Checking backend health..."
HEALTH=$(curl -s http://127.0.0.1:8000/health | python3.10 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
if [ "$HEALTH" = "healthy" ]; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend failed to start"
    exit 1
fi

# Start frontend
echo "Starting frontend on port 3000..."
cd ui && npm run dev > /tmp/mark_ui.log 2>&1 &
UI_PID=$!
echo "Frontend PID: $UI_PID"

sleep 3

echo ""
echo "=================================="
echo "  ✅ MARK System Started"
echo "=================================="
echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "API Docs: http://127.0.0.1:8000/docs"
echo ""
echo "To stop:"
echo "  kill $BACKEND_PID $UI_PID"
echo "  or run: pkill -f 'src.api.main|vite'"
echo "=================================="
