"""Standalone test harness.

Two jobs:
1. Make `import limbic` resolve to THIS repo regardless of the checkout
   directory's name (source tree, CI runner, arbitrary clone dir).
2. Stand in the hermes-core `agent.*` interface when running OUTSIDE a Hermes
   install (CI) — the plugin's provider imports the MemoryProvider ABC from
   core; the shim supplies the minimal import surface (no behavioural code).
   Inside a Hermes install the real core is importable and the shim no-ops.
"""
import importlib.util
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_limbic_package():
    """Alias the repo directory as the `limbic` package by module name, so a
    checkout named e.g. `limbic_verify` still imports as `limbic`."""
    if "limbic" in sys.modules:
        return
    exists = os.path.exists(os.path.join(REPO_ROOT, "__init__.py"))
    if not exists:
        return
    # The real package when the dir name already matches (source tree case):
    try:
        importlib.import_module("limbic")
        return
    except ImportError:
        pass
    spec = importlib.util.spec_from_file_location(
        "limbic", os.path.join(REPO_ROOT, "__init__.py"),
        submodule_search_locations=[REPO_ROOT])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["limbic"] = mod
    spec.loader.exec_module(mod)


def _install_hermes_compat_shim():
    for hook_path in (
        os.path.join(REPO_ROOT, "..", "hermes-agent"),
        os.path.dirname(sys.executable),
    ):
        if os.path.isdir(hook_path):
            break
    if _real_agent_available():
        return
    agent_mod = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # interface stand-in — ABC semantics live in core
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
    agent_mod.memory_provider = mp
    sys.modules["agent"] = agent_mod
    sys.modules["agent.memory_provider"] = mp


def _real_agent_available():
    for entry in list(sys.path):
        if os.path.isdir(os.path.join(entry, "agent")):
            return True
    return False


# ORDER: package alias + agent shim BEFORE any test import — pytest resolves
# tests/ rootless (no tests/__init__.py), so THIS conftest body runs first.
_install_hermes_compat_shim()
_install_limbic_package()
sys.path.insert(0, REPO_ROOT)
