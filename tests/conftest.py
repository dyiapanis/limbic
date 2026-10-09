"""Standalone test harness.

Two jobs:
1. Make `import limbic` resolve to THIS repo, not any installed/live copy.
2. Stand in the hermes-core `agent.*` interface when running OUTSIDE a Hermes
   install (CI) — the plugin's provider imports the MemoryProvider ABC from
   core; the shim supplies the minimal import surface (no behavioural code).
   Inside a Hermes install the real core is importable and the shim no-ops.
"""
import os
import sys
import types

REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _agent_importable() -> bool:
    for entry in list(sys.path):
        try:
            if os.path.isdir(os.path.join(entry, "agent")):
                return True
        except OSError:
            continue
    return False


def _install_hermes_compat_shim():
    if _agent_importable():
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


# ORDER MATTERS: shim BEFORE the parent-path insert — pytest imports the
# `limbic` package (rootdir has __init__.py) during conftest collection,
# before any test module import can trigger it again.
_install_hermes_compat_shim()
sys.path.insert(0, REPO_PARENT)
