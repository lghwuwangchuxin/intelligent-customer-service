#!/usr/bin/env python3
"""
MCP Service - Standalone Runner

Usage:
    # Run with default config
    python run.py

    # Run with custom port
    HTTP_PORT=8001 python run.py

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

    SERVICE_NAME = "mcp-service"

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
    logger.info(f"Configuration: host={config.host}, port={config.http_port}, nacos_enabled={config.nacos_enabled}")

    # Initialize service
    from services.mcp_service.service import MCPService
    from services.mcp_service.server import serve

    service = MCPService()
    service.initialize_default_tools(
        enable_web_search=os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true",
        enable_code_execution=os.getenv("ENABLE_CODE_EXECUTION", "true").lower() == "true",
        enable_knowledge=os.getenv("ENABLE_KNOWLEDGE", "true").lower() == "true",
    )

    logger.info(f"Registered {len(service.list_tools())} tools")
    logger.info(f"{SERVICE_NAME} starting on http://{config.host}:{config.http_port}")

    # Start HTTP server
    asyncio.run(serve(config, service))


if __name__ == "__main__":
    main()
