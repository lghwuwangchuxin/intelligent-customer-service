"""RAG Service entry point."""

import asyncio

from services.common.config import get_service_config
from services.common.logging import setup_logging, get_logger

SERVICE_NAME = "rag-service"


async def main():
    """Main entry point."""
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

    # Initialize service
    from .service import RAGService
    from .server import serve

    service = RAGService.from_config(config)
    await service.initialize()

    logger.info(f"{SERVICE_NAME} is running on port {config.http_port}")

    # Start HTTP server (includes Nacos registration)
    await serve(config, service)


if __name__ == "__main__":
    asyncio.run(main())
