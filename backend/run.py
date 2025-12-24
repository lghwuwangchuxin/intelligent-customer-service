#!/usr/bin/env python3
"""
Startup script for the Intelligent Customer Service backend.
"""
import uvicorn
from config.settings import settings


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )


if __name__ == "__main__":
    main()
