#!/usr/bin/env python3
"""
API Gateway - Standalone Runner

Usage:
    # Run with default config
    python run.py

    # Run with custom port
    HTTP_PORT=8000 python run.py

    # Run without Nacos (use direct service URLs)
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
    import uvicorn
    from services.common.config import get_service_config
    from services.common.logging import setup_logging, get_logger

    SERVICE_NAME = "api-gateway"

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
    logger.info(f"Nacos enabled: {config.nacos_enabled}")

    # Import app
    from services.api_gateway.app import create_app

    app = create_app(config)

    logger.info(f"{SERVICE_NAME} starting on http://{config.host}:{config.http_port}")

    # Start HTTP server
    uvicorn.run(
        app,
        host=config.host,
        port=config.http_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
