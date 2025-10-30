#!/bin/bash

# Talent Factory Startup Script
# This script starts the Talent Factory service and serves the UI

echo "🚀 Starting Talent Factory..."

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed (optional, for future enhancements)
if ! command -v node &> /dev/null
then
    echo "⚠️  Node.js is not installed. Some features may not work."
fi

# Navigate to the script directory
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "backend/venv" ]; then
    echo "Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source backend/venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
cd backend
pip install -r requirements.txt
cd ..

# Check for GPU availability
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected. Training will use CPU (slower)."
fi

# Start the Talent Factory backend service
echo "🔧 Starting Talent Factory backend service on port 8084..."
cd backend
python3 main.py &
BACKEND_PID=$!

# Wait a moment for the service to start
sleep 5

# Check if the service started successfully
if curl -s http://localhost:8084/health > /dev/null; then
    echo "✅ Talent Factory backend service is running on http://localhost:8084"
else
    echo "❌ Failed to start Talent Factory backend service"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

cd "$SCRIPT_DIR" # Back to talent-factory directory

# Start UI server (Next.js)
echo "🌐 Starting Next.js UI server on port 3004..."
cd ui

# Check if Next.js is available
if command -v npm &> /dev/null && [ -f "package.json" ]; then
    echo "Starting Next.js development server..."
    npm run dev -- --port 3004 &
    UI_PID=$!
else
    echo "Next.js not available, falling back to Python HTTP server..."
    python3 -m http.server 3004 &
    UI_PID=$!
fi

sleep 5 # Give UI server a moment to start

echo "✅ UI server is running on http://localhost:3004"

# Open the browser
echo "Opening Talent Factory in browser..."
if command -v open &> /dev/null; then
    open http://localhost:3004
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3004
elif command -v start &> /dev/null; then
    start http://localhost:3004
else
    echo "Please open http://localhost:3004 in your browser"
fi

echo ""
echo "🎉 Talent Factory is ready!"
echo ""
echo "🤖 Backend Service: http://localhost:8084"
echo "🌐 UI Server: http://localhost:3004"
echo "📚 MCP Catalogue: http://localhost:8084/mcp/talents"
echo ""
echo "📋 Quick Start:"
echo "   1. Open http://localhost:3004 in your browser"
echo "   2. Check your hardware compatibility on the dashboard"
echo "   3. Create a new talent using the wizard"
echo "   4. Upload your training dataset"
echo "   5. Start fine-tuning your model"
echo "   6. Publish your talent to the catalogue"
echo ""
echo "🔒 Security Features:"
echo "   • Local-first: All data stays on your machine"
echo "   • PII Detection: Automatically detects and masks sensitive data"
echo "   • Audit Logging: All actions are logged for compliance"
echo "   • Network Security: Only accessible from local network"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Talent Factory..."
    kill $BACKEND_PID 2>/dev/null
    kill $UI_PID 2>/dev/null
    echo "✅ Talent Factory stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep the script running to keep background processes alive
wait $BACKEND_PID $UI_PID
