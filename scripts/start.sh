#!/bin/bash

# Intelligent Customer Service - Startup Script

set -e

echo "=========================================="
echo "  Intelligent Customer Service Startup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in project root
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command_exists docker; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}Prerequisites OK${NC}"

# Check Ollama
echo -e "\n${YELLOW}Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo -e "${GREEN}Ollama is running${NC}"
else
    echo -e "${YELLOW}Warning: Ollama is not running. Please start Ollama first.${NC}"
    echo -e "Run: ${GREEN}ollama serve${NC}"
    echo -e "And pull a model: ${GREEN}ollama pull qwen2.5:7b${NC}"
fi

# Start services
echo -e "\n${YELLOW}Starting services...${NC}"

if docker compose version >/dev/null 2>&1; then
    docker compose up -d
else
    docker-compose up -d
fi

echo -e "\n${GREEN}Services started successfully!${NC}"
echo -e "\nAccess the application:"
echo -e "  - Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "  - Backend API: ${GREEN}http://localhost:8000${NC}"
echo -e "  - API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "\nTo view logs:"
echo -e "  ${YELLOW}docker compose logs -f${NC}"
echo -e "\nTo stop services:"
echo -e "  ${YELLOW}docker compose down${NC}"
