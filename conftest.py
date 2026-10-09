"""Root conftest — makes the suite runnable from ANY checkout/tarball directory name.

Why this exists (repo-name independence): the repo root carries an
``__init__.py``, and pytest's package walk (prepend import mode) imports that
root ``__init__.py`` by directory name whenever ``tests`` gets collected from
the repo root. Two failure classes drove this:

1. Checkout/tarball dir name is NOT a valid package name (GitHub tarballs ship
   as ``owner-repo-sha``): pytest imports ``__init__.py`` as a non-package
   module, its relative re-export imports crash, and every test dies at setup
   ("attempted relative import with no known parent package"). The re-exports
   in ``__init__.py`` are tolerant of exactly that import (see there); THIS
   conftest is what makes everything else work.
2. Whatever the directory is called, the test-suite imports must resolve to
   THIS repo as the ``limbic`` package and, outside a Hermes install, the
   hermes-core ``agent.*`` interface must be importable (shim, same as
   ``tests/conftest.py`` provides).

Order matters: pytest loads the ROOT conftest before collecting anything, so
the aliasing + shim installed here are in place before any test module import.
``tests/conftest.py`` keeps its own de-duplicated versions so test runs that
enter via ``tests/`` directly (``pytest tests/test_x.py``) still work.
"""

import importlib.util
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRNAME = os.path.basename(REPO_ROOT)


def _alias_as_package():
    """Register REPO_ROOT as module ``limbic`` (and as ROOT_DIRNAME when valid)."""
    if "limbic" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "limbic", os.path.join(REPO_ROOT, "__init__.py"),
            submodule_search_locations=[REPO_ROOT])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["limbic"] = mod
        spec.loader.exec_module(mod)
    if ROOT_DIRNAME.isidentifier() and ROOT_DIRNAME not in sys.modules:
        sys.modules[ROOT_DIRNAME] = sys.modules["limbic"]


def _install_hermes_compat_shim():
    for entry in list(sys.path):
        if os.path.isdir(os.path.join(entry, "agent")):
            return
    mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        def initialize(self, *a, **k): raise NotImplementedError
        def prefetch(self, *a, **k): raise NotImplementedError
        def remember(self, *a, **k): raise NotImplementedError
        def recall(self, *a, **k): raise NotImplementedError
        def get_config_schema(self): return {}
        def save_config(self, cfg): pass
        def system_prompt_block(self): return ""
        def on_pre_compress(self, *a, **k): return ""
        def on_memory_write(self, *a, **k): return None
        def status(self, *a, **k): return {}
        def stats(self, *a, **k): return {}

    class RecallStatus:
        OK = "ok"
        NO_RESULT = "no_result"
        def __init__(self, status="", results=None, count=0):
            self.status = status
            self.results = results if results is not None else []
            self.count = count
        def __eq__(self, other):
            return isinstance(other, RecallStatus) and self.status == other.status

    mp.MemoryProvider = MemoryProvider
    mp.RecallStatus = RecallStatus
    agent_mod = types.ModuleType("agent")
    agent_mod.memory_provider = mp
    sys.modules["agent"] = agent_mod
    sys.modules["agent.memory_provider"] = mp


_alias_as_package()
_install_hermes_compat_shim()
sys.path.insert(0, REPO_ROOT)
