#!/usr/bin/env python3
"""
Local Agent Runner - Start a single agent locally.

Usage:
    python run_agent.py <agent_name> [--port PORT]

Examples:
    python run_agent.py travel_assistant
    python run_agent.py billing_advisor --port 9003
    python run_agent.py all  # Start all agents
"""

import argparse
import importlib
import logging
import os
import sys
import subprocess
import signal
from pathlib import Path

# Add the agents directory to Python path
AGENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENTS_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Agent configurations
AGENTS = {
    "travel_assistant": {"port": 9101, "module": "travel_assistant"},
    "charging_manager": {"port": 9002, "module": "charging_manager"},
    "billing_advisor": {"port": 9003, "module": "billing_advisor"},
    "emergency_support": {"port": 9004, "module": "emergency_support"},
    "data_analyst": {"port": 9005, "module": "data_analyst"},
    "maintenance_expert": {"port": 9006, "module": "maintenance_expert"},
    "energy_advisor": {"port": 9007, "module": "energy_advisor"},
    "scheduling_advisor": {"port": 9008, "module": "scheduling_advisor"},
}


def run_single_agent(agent_name: str, port: int = None):
    """Run a single agent."""
    if agent_name not in AGENTS:
        logger.error(f"Unknown agent: {agent_name}")
        logger.info(f"Available agents: {', '.join(AGENTS.keys())}")
        return

    config = AGENTS[agent_name]
    agent_port = port or config["port"]

    # Set environment variables
    os.environ["AGENT_PORT"] = str(agent_port)
    os.environ["AGENT_HOST"] = "0.0.0.0"

    logger.info(f"Starting {agent_name} on port {agent_port}...")

    try:
        # Import and run the agent module
        module = importlib.import_module(f"{config['module']}.__main__")
        module.main()
    except KeyboardInterrupt:
        logger.info(f"Stopping {agent_name}...")
    except Exception as e:
        logger.error(f"Failed to start {agent_name}: {e}")
        raise


def run_all_agents():
    """Run all agents in separate processes."""
    processes = []

    def signal_handler(sig, frame):
        logger.info("Stopping all agents...")
        for p in processes:
            p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    for agent_name, config in AGENTS.items():
        port = config["port"]
        logger.info(f"Starting {agent_name} on port {port}...")

        env = os.environ.copy()
        env["AGENT_PORT"] = str(port)
        env["AGENT_HOST"] = "0.0.0.0"

        p = subprocess.Popen(
            [sys.executable, "-m", config["module"]],
            cwd=str(AGENTS_DIR),
            env=env,
        )
        processes.append(p)

    logger.info(f"All {len(processes)} agents started!")
    logger.info("Press Ctrl+C to stop all agents.")

    # Wait for all processes
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


def main():
    parser = argparse.ArgumentParser(description="Run A2A agents locally")
    parser.add_argument(
        "agent",
        choices=list(AGENTS.keys()) + ["all"],
        help="Agent to run (or 'all' for all agents)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override default port",
    )

    args = parser.parse_args()

    if args.agent == "all":
        run_all_agents()
    else:
        run_single_agent(args.agent, args.port)


if __name__ == "__main__":
    main()
