"""
Tiled integration for multihead HRD data.

This module provides Tiled adapters and utilities for serving high-resolution
diffraction data through the Tiled data access framework.

Key components:
- HRDRawAdapter: Tiled adapter for HRDRawBase objects
- register_hrd_data: Functions for discovering and registering HRD data
"""

from .tiled import HRDRawAdapter, read_hrd_raw

__all__ = ["HRDRawAdapter", "read_hrd_raw"]