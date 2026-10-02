"""Limbic — Self-organising memory system for AI agents.

Opaque, adaptive, local-first. Modelled on the brain's limbic system.

Architecture:
  hippocampus  → encoding (remember → store + embed + interference + surprise)
  amygdala     → salience (adaptive trust scoring)
  thalamus     → routing + gating (prefetch → search + relevance floor)
  maturation   → engram development (silent → active via sigmoid)
  reconsolidation → organic correction via priming (no content modification)

The agent sees two tools: remember(content) and recall(query).
The operator sees one file: limbic.db.
Everything inside is self-organising.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Re-export for convenience
from .embeddings import LimbicEmbedder, OnnxEmbedder
from .nli import LimbicNLI
from .storage import LimbicStorage
from .trust import TrustEngine
from .provider import LimbicMemoryProvider
from .tools import REMEMBER_SCHEMA, RECALL_SCHEMA, handle_tool_call


def register(ctx):
    """Called by the memory plugin discovery system.

    Per the memory-provider-plugin contract, we only need to register
    the MemoryProvider instance.  Tool schemas (remember/recall) are
    injected automatically by :meth:`MemoryManager.get_all_tool_schemas`,
    and lifecycle hooks (on_pre_compress, on_memory_write, etc.) are
    dispatched by the MemoryManager — no general-plugin hook registration
    is required.
    """
    provider = LimbicMemoryProvider()
    ctx.register_memory_provider(provider)
    logger.info("limbic memory provider registered")