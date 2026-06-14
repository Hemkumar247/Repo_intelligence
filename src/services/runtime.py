"""Lazy runtime accessors for heavyweight services."""
from __future__ import annotations

from functools import lru_cache

from config.settings import get_settings


@lru_cache()
def get_pipeline():
    from src.pipeline import RepoIntelligencePipeline

    return RepoIntelligencePipeline()


@lru_cache()
def get_lifecycle_manager():
    from src.services.lifecycle import RepoLifecycleManager

    settings = get_settings()
    return RepoLifecycleManager(settings.state_dir)

