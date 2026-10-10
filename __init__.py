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
import sys

logger = logging.getLogger(__name__)

# Re-export for convenience. TOLERANT of non-package import: pytest's package
# walk imports this repo-root __init__.py top-level whenever the checkout dir
# name isn't a valid package name (GitHub tarballs ship as ``owner-repo-sha``),
# and relative imports crash on that import. In that walk the actual ``limbic``
# package is aliased by conftest, so the re-exports skip harmlessly here.
_NONPACKAGE_IMPORT = __name__ != "limbic"  # pytest's walk imports by dirname
_PROVIDER_CLASS = None  # set when package-imported (eager re-exports below)
if not _NONPACKAGE_IMPORT:
    from .embeddings import LimbicEmbedder, OnnxEmbedder
    from .nli import LimbicNLI
    from .storage import LimbicStorage
    from .trust import TrustEngine
    from .provider import LimbicMemoryProvider
    _PROVIDER_CLASS = LimbicMemoryProvider
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
    cls = globals().get("_PROVIDER_CLASS")
    if cls is None:
        # Standalone file-load (hermes capability probe): the module name isn't
        # "limbic", so eager relative re-exports were skipped. Import provider.py
        # by path; sibling files resolve via the repo/package context.
        import importlib.util, os
        _pv = os.path.join(os.path.dirname(__file__), "provider.py")
        _spec = importlib.util.spec_from_file_location(
            "limbic.provider", _pv, submodule_search_locations=[os.path.dirname(_pv)])
        _prov = importlib.util.module_from_spec(_spec)
        sys.modules.setdefault("limbic.provider", _prov)
        _spec.loader.exec_module(_prov)
        cls = _prov.LimbicMemoryProvider
        globals()["_PROVIDER_CLASS"] = cls
    provider = cls()
    ctx.register_memory_provider(provider)
    logger.info("limbic memory provider registered")