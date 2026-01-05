"""API Gateway entry point."""

import uvicorn

from services.common.config import get_service_config


def main():
    """Run the API Gateway."""
    config = get_service_config("api-gateway")

    uvicorn.run(
        "services.api_gateway.app:app",
        host=config.host,
        port=config.http_port,
        reload=False,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
