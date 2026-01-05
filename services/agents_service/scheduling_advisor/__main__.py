"""
Scheduling Advisor Agent - Main Entry Point.
Runs as an A2A server on port 9008.
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

from .executor import SchedulingAdvisorExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.nacos_registry import register_agent, deregister_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

AGENT_NAME = "scheduling_advisor"
AGENT_PORT = int(os.getenv("AGENT_PORT", "9008"))
AGENT_HOST = os.getenv("AGENT_HOST", "0.0.0.0")

_service_id = None


def create_agent_card() -> AgentCard:
    """Create the agent card for Scheduling Advisor."""
    skill = AgentSkill(
        id="forecast_load",
        name="负荷预测",
        description="预测用电负荷，优化调度计划",
        tags=["负荷", "预测", "调度"],
        examples=[
            "预测明天的用电负荷",
            "如何平衡峰谷用电？",
            "今天的负荷预测是什么？",
            "什么时候是用电高峰？",
        ],
    )

    return AgentCard(
        name="调度顾问",
        description="负荷预测和调度优化专家",
        url=f"http://{AGENT_HOST}:{AGENT_PORT}/",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )


def create_app() -> A2AStarletteApplication:
    """Create the A2A Starlette application."""
    executor = SchedulingAdvisorExecutor()
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

    logger.info(f"Scheduling Advisor Agent created on port {AGENT_PORT}")
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
    """Run the Scheduling Advisor agent server."""
    global _service_id

    logger.info(f"Starting Scheduling Advisor Agent on {AGENT_HOST}:{AGENT_PORT}")

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup)

    _service_id = register_agent(
        agent_name=AGENT_NAME,
        port=AGENT_PORT,
        version="1.0.0",
        tags=["scheduling", "load", "forecast"],
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
