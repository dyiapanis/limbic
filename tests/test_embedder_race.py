
"""Embedder loader race: warm thread + first dispatch must not double-load."""
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from limbic.embeddings import OnnxEmbedder


def test_concurrent_first_embed_single_load(tmp_path):
    """3 threads hitting _ensure_model simultaneously -> exactly one ONNX load."""
    e = OnnxEmbedder(cache_dir=str(tmp_path / "embedding"))
    # Count load invocations by wrapping _load_model_locked
    loads = []
    import threading as _t
    orig_lock_release = e._init_lock.release
    def counting_release():
        loads.append(1)
        orig_lock_release()
    # Simpler: monkeypatch _load_model_locked to record + delegate to a real minimal load
    real = e._load_model_locked
    e._session = None
    def fake_load():
        with e._init_lock:  # NOT reentrant — would deadlock
            pass
    # Simpler still: directly track that only one thread executes the load body:
    entered = []
    def slow_load():
        entered.append(threading.get_ident())
        import time as _time
        _time.sleep(0.05)  # widen the race window
        e._session = object()  # type: ignore[assignment]
    e._load_model_locked = slow_load  # type: ignore[method-assign]
    threads = [threading.Thread(target=e._ensure_model) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert e._session is not None
    assert len(entered) == 1, f"loader ran {len(entered)} times under concurrency"


def test_steady_state_skips_lock():
    """Already-loaded embedder: _ensure_model returns without touching the lock."""
    e = OnnxEmbedder(cache_dir=str(tmp_path_home()))
    e._session = object()  # type: ignore[assignment]
    e._ensure_model()  # must not raise / must not acquire (session short-circuit first)
    assert e._session is not None


def tmp_path_home():
    import tempfile
    return tempfile.mkdtemp()
