#!/bin/bash
# Start all A2A agents locally
# Usage: ./start_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Starting A2A Multi-Agent System${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if a2a-sdk is installed
if ! python -c "import a2a" 2>/dev/null; then
    echo -e "${YELLOW}Installing A2A SDK...${NC}"
    pip install -r requirements.txt
fi

# Load environment variables
if [ -f ".env.agents" ]; then
    echo -e "${GREEN}Loading configuration from .env.agents${NC}"
    export $(cat .env.agents | grep -v '^#' | xargs)
fi

# Define agents and their ports
declare -A AGENTS=(
    ["travel_assistant"]=9101
    ["charging_manager"]=9002
    ["billing_advisor"]=9003
    ["emergency_support"]=9004
    ["data_analyst"]=9005
    ["maintenance_expert"]=9006
    ["energy_advisor"]=9007
    ["scheduling_advisor"]=9008
)

# Array to store PIDs
PIDS=()

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Stopping all agents...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
    done
    echo -e "${GREEN}All agents stopped.${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start each agent
for agent in "${!AGENTS[@]}"; do
    port=${AGENTS[$agent]}
    echo -e "${GREEN}Starting $agent on port $port...${NC}"

    AGENT_PORT=$port AGENT_HOST=0.0.0.0 python -m $agent &
    PIDS+=($!)

    # Wait a bit between starts
    sleep 1
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  All agents started!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Agents running:"
for agent in "${!AGENTS[@]}"; do
    port=${AGENTS[$agent]}
    echo -e "  - $agent: http://localhost:$port"
done
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all agents${NC}"
echo ""

# Wait for all background processes
wait
