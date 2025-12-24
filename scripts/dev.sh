#!/bin/bash

# Development environment startup script

set -e

echo "=========================================="
echo "  Development Environment Startup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Start Milvus with Docker
echo -e "\n${YELLOW}Starting Milvus...${NC}"
docker run -d --name milvus-standalone \
    -p 19530:19530 \
    -p 9091:9091 \
    -v milvus_data:/var/lib/milvus \
    milvusdb/milvus:v2.4.0 2>/dev/null || echo "Milvus already running or starting..."

# Wait for Milvus
echo -e "${YELLOW}Waiting for Milvus to be ready...${NC}"
sleep 5

# Start Backend
echo -e "\n${YELLOW}Starting Backend...${NC}"
cd "$PROJECT_ROOT/backend"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
pip install -r requirements.txt --quiet

# Copy env file if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file from .env.example"
fi

# Create data directories
mkdir -p data/knowledge_base

# Start backend in background
python run.py &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"

# Start Frontend
echo -e "\n${YELLOW}Starting Frontend...${NC}"
cd "$PROJECT_ROOT/frontend"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"

echo -e "\n${GREEN}Development environment is ready!${NC}"
echo -e "\nAccess:"
echo -e "  - Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "  - Backend: ${GREEN}http://localhost:8000${NC}"
echo -e "  - API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "\nPress Ctrl+C to stop all services"

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
