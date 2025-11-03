"""
Web interface for memory system.

Provides FastAPI app and visualization interface.
"""
from .server import app, memory

__all__ = ["app", "memory"]
