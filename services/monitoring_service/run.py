#!/usr/bin/env python3
"""
Monitoring Service - Standalone Runner

Usage:
    # Run with default config
    python run.py

    # Run with Langfuse enabled
    LANGFUSE_ENABLED=true LANGFUSE_PUBLIC_KEY=xxx python run.py

    # Run without Nacos
    NACOS_ENABLED=false python run.py
"""

import asyncio
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Load .env file if exists
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    from dotenv import load_dotenv
    load_dotenv(env_file)


def main():
    """Main entry point."""
    from services.common.config import get_service_config
    from services.common.logging import setup_logging, get_logger

    SERVICE_NAME = "monitoring-service"

    # Load configuration
    config = get_service_config(SERVICE_NAME)

    # Setup logging
    setup_logging(
        service_name=SERVICE_NAME,
        level=config.log_level,
        log_format=config.log_format,
        log_to_console=config.log_to_console,
        log_to_file=config.log_to_file,
        log_dir=config.log_dir,
    )

    logger = get_logger(SERVICE_NAME)
    logger.info(f"Starting {SERVICE_NAME}...")
    logger.info(f"Configuration: host={config.host}, port={config.http_port}")
    logger.info(f"Langfuse enabled: {config.langfuse_enabled}")

    # Initialize service
    from services.monitoring_service.service import MonitoringService
    from services.monitoring_service.server import serve

    service = MonitoringService.from_config(config)

    logger.info(f"{SERVICE_NAME} starting on http://{config.host}:{config.http_port}")

    # Start HTTP server
    asyncio.run(serve(config, service))


if __name__ == "__main__":
    main()
