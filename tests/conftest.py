"""Make `import limbic` resolve to THIS repo, not any installed/live copy."""
import os
import sys

# The repo root IS the `limbic` package — put its PARENT on sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# CI shim — Hermes core (`agent.*`) is not importable on a standalone checkout.
# The real ABC is used whenever Limbic runs inside Hermes; here we stand in the
# minimal interface surface the suite exercises (no behavioural code).
def _install_hermes_compat_shim():
    if _has_agent():
        return
    import types
    agent_mod = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # ABC stand-in with the plugin's contract
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

    class RecallStatus:  # enum-shaped stand-in (tests construct/compare)
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


def _has_agent():
    try:
        import agent  # noqa: F401
        return True
    except ImportError:
        return False


_install_hermes_compat_shim()
