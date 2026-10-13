"""API package: minimal stdlib HTTP server over live lab state."""

from .server import create_handler, serve

__all__ = ["create_handler", "serve"]
