"""
src/utils/config_loader.py

Single entry point for reading config.yaml.

Usage:
    from src.utils.config_loader import cfg

    num_perm   = cfg("minhash", "num_perm", default=128)
    top_k      = cfg("retrieval", "top_k", default=5)
    chunks_path = cfg("paths", "chunks", default="data/processed/chunks.jsonl")
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


# Locate config.yaml relative to this file (two levels up = project root)
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache config.yaml.  Called once per process; subsequent calls
    return the cached dict.

    Raises:
        FileNotFoundError : if config.yaml is absent.
        yaml.YAMLError    : if config.yaml is malformed.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Ensure you are running from the project root."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def cfg(section: str, key: str, default: Any = None) -> Any:
    """
    Convenience accessor with a safe default.

    Args:
        section : Top-level key in config.yaml (e.g. "minhash").
        key     : Key within that section (e.g. "num_perm").
        default : Value to return if the key or section is absent.

    Returns:
        The config value, or default if not found.
    """
    try:
        return get_config().get(section, {}).get(key, default)
    except Exception:
        return default