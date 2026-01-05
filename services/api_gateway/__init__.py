"""API Gateway - Unified entry point for microservices."""

from .app import create_app

__all__ = ["create_app"]
