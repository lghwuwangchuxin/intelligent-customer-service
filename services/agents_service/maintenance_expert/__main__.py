"""
Maintenance Expert Agent - Main Entry Point.
Runs as an A2A server on port 9006.
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

from .executor import MaintenanceExpertExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nacos_registry import register_agent, deregister_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AGENT_NAME = "maintenance_expert"
AGENT_PORT = int(os.getenv("AGENT_PORT", "9006"))
AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")

_service_id = None


def create_agent_card() -> AgentCard:
    """Create the agent card for Maintenance Expert."""
    skill = AgentSkill(
        id="check_device_health",
        name="检查设备健康",
        description="监测设备健康状态，预测维护需求",
        tags=["设备", "健康", "维护"],
        examples=[
            "检查充电桩的健康状态",
            "哪些设备需要维护？",
            "设备运行情况如何？",
            "有没有故障风险？",
        ],
    )

    return AgentCard(
        name="运维专家",
        description="设备健康监测和预测性维护专家",
        url=f"http://{AGENT_HOST}:{AGENT_PORT}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    executor = MaintenanceExpertExecutor()
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )
    agent_card = create_agent_card()

    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )

    logger.info(f"Maintenance Expert Agent created on port {AGENT_PORT}")
    return app


def _cleanup():
    global _service_id
    if _service_id:
        logger.info(f"Deregistering agent from Nacos: {_service_id}")
        deregister_agent(_service_id)
        _service_id = None


def _signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    _cleanup()
    sys.exit(0)


def main():
    """Run the Maintenance Expert agent server."""
    global _service_id

    logger.info(f"Starting Maintenance Expert Agent on {AGENT_HOST}:{AGENT_PORT}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup)

    _service_id = register_agent(
        agent_name=AGENT_NAME,
        port=AGENT_PORT,
        version="1.0.0",
        tags=["maintenance", "device", "health"],
    )
    if _service_id:
        logger.info(f"Registered with Nacos: {_service_id}")

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
