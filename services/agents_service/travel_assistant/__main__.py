"""
Travel Assistant Agent - Main Entry Point.
Runs as an A2A server on port 9001.
"""

import atexit
import logging
import os
import signal
import sys

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentSkill, AgentCapabilities

from .executor import TravelAssistantExecutor

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nacos_registry import register_agent, deregister_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Agent configuration
AGENT_NAME = "travel_assistant"
AGENT_PORT = int(os.getenv("AGENT_PORT", "9001"))
AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")

# Global service ID for cleanup
_service_id = None


def create_agent_card() -> AgentCard:
    """Create the agent card for Travel Assistant."""
    skill = AgentSkill(
        id="search_nearby_stations",
        name="搜索附近充电站",
        description="根据位置搜索附近的充电站，支持按距离、价格、可用性筛选",
        tags=["充电站", "搜索", "导航"],
        examples=[
            "附近有什么充电站？",
            "找一个最近的快充站",
            "帮我找个有空位的充电桩",
            "周围5公里内有充电站吗？",
        ],
    )

    return AgentCard(
        name="出行助手",
        description="专业的充电站导航助手，帮您快速找到合适的充电站",
        url=f"http://{AGENT_HOST}:{AGENT_PORT}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    # Create executor
    executor = TravelAssistantExecutor()

    # Create task store
    task_store = InMemoryTaskStore()

    # Create request handler
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    # Create agent card
    agent_card = create_agent_card()

    # Create application
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    logger.info(f"Travel Assistant Agent created on port {AGENT_PORT}")
    return app


def _cleanup():
    """Cleanup function for graceful shutdown."""
    global _service_id
    if _service_id:
        logger.info(f"Deregistering agent from Nacos: {_service_id}")
        deregister_agent(_service_id)
        _service_id = None


def _signal_handler(signum, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    _cleanup()
    sys.exit(0)


def main():
    """Run the Travel Assistant agent server."""
    global _service_id

    logger.info(f"Starting Travel Assistant Agent on {AGENT_HOST}:{AGENT_PORT}")

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Register cleanup on exit
    atexit.register(_cleanup)

    # Register with Nacos
    _service_id = register_agent(
        agent_name=AGENT_NAME,
        port=AGENT_PORT,
        version="1.0.0",
        tags=["navigation", "charging-station", "search"],
    )
    if _service_id:
        logger.info(f"Registered with Nacos: {_service_id}")
    else:
        logger.warning("Nacos registration skipped or failed")

    # Create and run app
    app = create_app()

    try:
        uvicorn.run(
            app.build(),
            host=AGENT_HOST,
            port=AGENT_PORT,
            log_level="info",
        )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
