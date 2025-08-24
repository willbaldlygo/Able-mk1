#!/bin/bash

# Able Enhanced Startup Script
set -e

echo "🚀 Starting Able PDF Research Assistant..."

# Check environment
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your ANTHROPIC_API_KEY"
    exit 1
fi

if ! grep -q "ANTHROPIC_API_KEY=.*[^[:space:]]" .env; then
    echo "❌ Error: ANTHROPIC_API_KEY not set in .env file!"
    echo "Please add your Anthropic API key to .env file"
    echo "Get your API key from: https://console.anthropic.com/"
    exit 1
fi

# Cleanup function
cleanup() {
    echo "🛑 Shutting down servers..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$OLLAMA_PID" ]; then
        echo "  - Stopping Ollama..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Create directories
mkdir -p data/{vectordb,sources}

echo "🤖 Starting Ollama local AI service..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found! Please install with: brew install ollama"
    exit 1
fi

# Start Ollama service
echo "  - Launching Ollama server..."
ollama serve > /dev/null 2>&1 &
OLLAMA_PID=$!

sleep 3

# Verify Ollama is running and check for gpt-oss:20b model
if ! ollama list | grep -q "gpt-oss:20b"; then
    echo "⚠️  gpt-oss:20b model not found. Available models:"
    ollama list
    echo ""
    echo "💡 Your gpt-oss:20b model will be available once downloaded/loaded"
else
    echo "  ✅ gpt-oss:20b model available"
fi

echo "📦 Setting up dependencies..."

# Backend setup
echo "  - Python backend setup..."
cd backend

if [ ! -d "venv" ] || ! ./venv/bin/python --version &>/dev/null; then
    echo "    Creating/recreating virtual environment..."
    rm -rf venv
    python3 -m venv venv
fi

echo "    Installing Python dependencies..."
./venv/bin/pip install -r requirements.txt

echo "🐍 Starting FastAPI backend (port 8000)..."
./venv/bin/python main.py &
BACKEND_PID=$!

sleep 3

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

cd ../frontend

echo "⚛️  Setting up React frontend..."
echo "    Installing Node dependencies..."
npm install > /dev/null 2>&1

echo "🌐 Starting React frontend (port 3000)..."
npm start &
FRONTEND_PID=$!

cd ..

echo "⏳ Waiting for servers to initialize..."
sleep 5

echo ""
echo "✅ Able is now running!"
echo ""
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "🎯 Enhanced Features:"
echo "  • Microsoft GraphRAG integration"
echo "  • Dual AI providers (Anthropic + Ollama)"
echo "  • Intelligent search routing"
echo "  • Voice input capabilities" 
echo "  • Advanced knowledge synthesis"
echo "  • Local gpt-oss:20b model ready"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID