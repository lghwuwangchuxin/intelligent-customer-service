#!/bin/bash
# Run a specific microservice standalone
#
# Usage:
#   ./run_service.sh <service_name>
#
# Examples:
#   ./run_service.sh mcp_service
#   ./run_service.sh rag_service
#   ./run_service.sh api_gateway
#
# Available services:
#   - mcp_service (port 8001)
#   - rag_service (port 8002)
#   - evaluation_service (port 8003)
#   - monitoring_service (port 8004)
#   - single_agent_service (port 8005)
#   - multi_agent_service (port 8006)
#   - llm_manager_service (port 8007)
#   - api_gateway (port 8000)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Service name from argument
SERVICE_NAME=${1:-""}

if [ -z "$SERVICE_NAME" ]; then
    echo "Usage: $0 <service_name>"
    echo ""
    echo "Available services:"
    echo "  - mcp_service (port 8001)"
    echo "  - rag_service (port 8002)"
    echo "  - evaluation_service (port 8003)"
    echo "  - monitoring_service (port 8004)"
    echo "  - single_agent_service (port 8005)"
    echo "  - multi_agent_service (port 8006)"
    echo "  - llm_manager_service (port 8007)"
    echo "  - api_gateway (port 8000)"
    exit 1
fi

SERVICE_DIR="$SCRIPT_DIR/$SERVICE_NAME"

if [ ! -d "$SERVICE_DIR" ]; then
    echo "Error: Service directory not found: $SERVICE_DIR"
    exit 1
fi

# Check if run.py exists
if [ ! -f "$SERVICE_DIR/run.py" ]; then
    echo "Error: run.py not found in $SERVICE_DIR"
    exit 1
fi

# Create .env from .env.example if it doesn't exist
if [ -f "$SERVICE_DIR/.env.example" ] && [ ! -f "$SERVICE_DIR/.env" ]; then
    echo "Creating .env from .env.example..."
    cp "$SERVICE_DIR/.env.example" "$SERVICE_DIR/.env"
fi

# Install dependencies if requirements.txt exists
if [ -f "$SERVICE_DIR/requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -q -r "$SERVICE_DIR/requirements.txt"
fi

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting $SERVICE_NAME..."
echo "Working directory: $SERVICE_DIR"
echo ""

# Run the service
cd "$SERVICE_DIR"
python run.py
