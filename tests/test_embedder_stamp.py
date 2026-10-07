"""Regression tests — embedder-stamp update mechanism (int8 switch release).

Covers:
1. Fresh empty DB adopts the running embedder's stamp on first open + check.
2. Same-stamp subsequent opens are never stale.
3. Stamp mismatch (plugin changed model without reindex) → stale_embedder=True.
4. reindex(embedder_stamp=...) writes the new stamp and clears the stale flag.
5. Legacy DB (facts exist, no stamp — pre-stamp corpus) adopts dims marker,
   does NOT mark stale (same dims keeps retrieval valid), and reindex+stamp
   cleanly upgrades it.
6. The stamp identity derivation (id@revision:artifact) — artifact must come
   from the embeddings pin table's single onnx engine file.
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel_path: str):
    """Import a submodule without triggering limbic/__init__ (host deps)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / rel_path, submodule_search_locations=[str(REPO)]
    )
    assert spec and spec.loader
    import types
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # store_sqlite imports sqlite_vec at module level — stub if missing
        raise
    return mod


def _make_storage(tmp_path, monkeypatch=None):
    import sqlite3
    # sqlite-vec must be importable; the plugin venv provides it at runtime.
    # Tests use the real module if present, else skip (CI without sqlite_vec).
    try:
        import sqlite_vec  # noqa: F401
    except ImportError:
        import pytest
        pytest.skip("sqlite_vec not installed")

    import types
    ssm = _load("limbic_store_sqlite", "store_sqlite.py")
    from limbic_store_sqlite import SQLiteStorage
    db_path = str(tmp_path / "limbic.db")
    return SQLiteStorage(path=db_path, dims=1024, collection_name="limbic_test")


STAMP_A = "Snowflake/snowflake-arctic-embed-l-v2.0@ac6544c8:model_int8.onnx"
STAMP_B = "Snowflake/snowflake-arctic-embed-l-v2.0@ac6544c8:model.onnx"


def _store_fact(storage, content="user prefers terse answers"):
    storage.store(
        content=content, user_id="u1",
        embedding=[1.0] + [0.0] * 1023,
        metadata={"trust_score": 0.5, "retrieval_count": 0},
    )


def test_fresh_db_adopts_stamp(tmp_path):
    s = _make_storage(tmp_path)
    try:
        assert s.check_embedder_stamp(STAMP_A, 1024) is False
        assert s.embedder_stamp == STAMP_A
        assert s.stale_embedder is False
    finally:
        s.close()


def test_same_stamp_not_stale(tmp_path):
    s = _make_storage(tmp_path)
    try:
        s.check_embedder_stamp(STAMP_A, 1024)
        assert s.check_embedder_stamp(STAMP_A, 1024) is False
        assert s.stale_embedder is False
    finally:
        s.close()


def test_stamp_mismatch_marks_stale(tmp_path):
    s = _make_storage(tmp_path)
    try:
        s.check_embedder_stamp(STAMP_A, 1024)
        # plugin update switches engine artifact (int8 → fp32) without reindex
        assert s.check_embedder_stamp(STAMP_B, 1024) is True
        assert s.stale_embedder is True
    finally:
        s.close()


def test_reindex_stamp_clears_stale(tmp_path):
    s = _make_storage(tmp_path)
    try:
        s.check_embedder_stamp(STAMP_A, 1024)
        s.check_embedder_stamp(STAMP_B, 1024)
        assert s.stale_embedder is True

        class FakeEmbedder:
            dims = 1024
            def embed(self, content, is_query=False):
                return [0.5] + [0.0] * 1023

        result = s.reindex(FakeEmbedder(), 1024, embedder_stamp=STAMP_B)
        assert result["status"] == "complete"
        assert result["embedder_stamp"] == STAMP_B
        assert s.embedder_stamp == STAMP_B

        # next open-check with the new stamp is clean
        s2 = _make_storage(tmp_path)
        try:
            assert s2.check_embedder_stamp(STAMP_B, 1024) is False
            assert s2.stale_embedder is False
        finally:
            s2.close()
    finally:
        s.close()


def test_legacy_db_adopts_dims_marker_not_stale(tmp_path):
    """Pre-stamp corpus (existing facts, no stamp): not stale (dims match),
    dims marker recorded so future dimension changes still detect."""
    s = _make_storage(tmp_path)
    try:
        _store_fact(s)  # fact exists, no stamp yet
        assert s.embedder_stamp is None
        assert s.check_embedder_stamp(STAMP_A, 1024) is False
        # stamp NOT adopted on a populated legacy DB (would lie about vectors)
        assert s.embedder_stamp is None
        # reindex+stamp upgrades it
        class FakeEmbedder:
            dims = 1024
            def embed(self, content, is_query=False):
                return [0.5] + [0.0] * 1023
        result = s.reindex(FakeEmbedder(), 1024, embedder_stamp=STAMP_A)
        assert result["status"] == "complete"
        assert s.embedder_stamp == STAMP_A
    finally:
        s.close()


def test_stamp_identity_from_pins():
    """The provider/reindex stamp derivation matches the pins table exactly."""
    mp = _load("limbic_model_pins", "model_pins.py")
    pin = mp.model_pin(mp.EMBEDDING_MODEL_ID)
    engines = [k for k in pin["files"] if k.startswith("onnx/") and k.endswith(".onnx")]
    assert engines == ["onnx/model_int8.onnx"], (
        f"embeddings pin must carry exactly the int8 engine: {engines}"
    )
    artifact = engines[0].split("/")[-1]
    assert artifact == "model_int8.onnx"


def test_embedder_stamp_helper_matches_format():
    """model_pins.embedder_stamp() — single source for provider + CLI + reindex."""
    mp = _load("limbic_model_pins", "model_pins.py")
    stamp = mp.embedder_stamp()
    assert stamp == (
        f"{mp.EMBEDDING_MODEL_ID}@{mp.EMBEDDING_REVISION}:model_int8.onnx"
    )


def test_legacy_dims_mismatch_flags_stale(tmp_path):
    """Legacy corpus: first same-dims open records the dims marker (not stale);
    a LATER open at different dims MUST flag stale — old vectors are
    physically incompatible with a different-dims vec table."""
    s = _make_storage(tmp_path)
    try:
        _store_fact(s)
        assert s.embedder_stamp is None
        # 0.6.0 first open on the legacy DB — same dims 1024, any stamp:
        # not stale, marker recorded
        assert s.check_embedder_stamp(STAMP_A, 1024) is False
        assert s.stale_embedder is False
        # later plugin update that changes dims (no reindex yet): flags
        assert s.check_embedder_stamp("other/model@abc:model.onnx", 384) is True
        assert s.stale_embedder is True
    finally:
        s.close()