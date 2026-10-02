"""Storage-layer tests for Limbic SQLiteStorage.

Model-free by design: dims=8, fake embeddings, temp dirs.

Run:
    cd <hermes-plugins-parent> && \
    python -m pytest limbic/tests/test_store_sqlite.py -v
"""
import math
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.store_sqlite import SQLiteStorage  # noqa: E402

DIMS = 8


def _vec(i: int) -> list[float]:
    """Deterministic fake embedding, L2-normalised."""
    v = [float(i + j % 7 + 1) for j in range(DIMS)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    yield s
    s.close()


def _mkfact(store, i):
    return store.store(
        f"fact number {i} about topic {i % 3}",
        user_id="alice",
        embedding=_vec(i),
        metadata={"trust_score": 0.5, "pinned": False},
    )


def test_store_get_roundtrip(store):
    fid = _mkfact(store, 1)
    assert fact_row(store, fid)


def fact_row(store, fid):
    fact = store.get(fid)
    if fact is None:
        return False
    return fact["content"] == "fact number 1 about topic 1" and fact["user_id"] == "alice"


def test_ids_are_sequential_autoincrement(store):
    a, b, c = _mkfact(store, 1), _mkfact(store, 2), _mkfact(store, 3)
    assert b == a + 1 and c == b + 1


def test_update_rejects_unknown_column(store):
    _mkfact(store, 1)
    with pytest.raises(ValueError):
        store.update(1, **{"content; DROP TABLE facts;--": "x"})


def test_update_rejects_fact_id_rewrite(store):
    _mkfact(store, 1)
    with pytest.raises((ValueError, TypeError)):
        store.update(1, fact_id=999)


def test_update_whitelisted_columns_work(store):
    fid = _mkfact(store, 1)
    store.update(fid, trust_score=0.9, retrieval_count=7)
    f = store.get(fid)
    assert f["trust_score"] == 0.9
    assert f["retrieval_count"] == 7


def test_delete_removes_fact_vec_fts_atomically(store):
    fids = [_mkfact(store, i) for i in range(3)]
    target = fids[1]
    before = store.count()
    store.delete_fact(target)
    assert store.count() == before - 1
    assert store.get(target) is None
    n = store._db.execute("SELECT COUNT(*) FROM vec_index WHERE rowid = ?", (target,)).fetchone()[0]
    assert n == 0
    hits = store.search_fts("number", {"user_id": "alice"}, limit=10)
    assert all(h["fact_id"] != target for h in hits)


def test_atrophy_sweeps_high_trust_if_marked(store):
    """No fact is exempt from removal by trust alone.

    There is no pinned class, so a marked fact is swept regardless of how
    high its trust is — the marking decision (is_atrophy_candidate) is the
    only gate.
    """
    for i in range(4):
        _mkfact(store, i)
    store.update(1, trust_score=0.9)
    for fid in (2, 3, 4):
        store.mark_atrophy(fid)
    store.mark_atrophy(1)
    before = (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()
    deleted = store.sweep_atrophy(before, limit=10)
    assert deleted == 4
    assert store.get(1) is None


def test_reindex_preserves_vectors_and_cosine_ddl(store):
    for i in range(5):
        _mkfact(store, i)

    def fake_embedder(text):
        return _vec(abs(hash(text)) % 97)

    store.reindex(fake_embedder, DIMS)
    after_vec = store._db.execute("SELECT COUNT(*) FROM vec_index").fetchone()[0]
    assert after_vec == 5
    ddl = store._db.execute(
        "SELECT sql FROM sqlite_master WHERE name='vec_index'"
    ).fetchone()[0]
    assert "distance_metric=cosine" in ddl


def test_databaseerror_surfaced_on_corrupt_db(store, tmp_path):
    for i in range(3):
        _mkfact(store, i)
    store._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store.close()

    path = str(tmp_path / "limbic.db")
    # Corrupt from offset 100 (page 1, past the 100-byte header) rather than a
    # mid-file page: a fixed mid-file offset lands on whatever object the
    # current schema happens to place there, so it silently stops corrupting
    # anything when the schema shrinks. Page 1 always holds sqlite_master.
    with open(path, "r+b") as f:
        f.seek(100, os.SEEK_SET)
        f.write(b"\xff" * 4096)

    s2 = SQLiteStorage(path, dims=DIMS)
    assert s2.healthy is False
    assert s2._degraded is True
    s2.close()


def test_backup_produces_consistent_copy(store, tmp_path):
    for i in range(3):
        _mkfact(store, i)
    bdir = str(tmp_path / "backups")
    bpath = store.backup(bdir)
    assert os.path.exists(bpath)
    chk = sqlite3.connect(bpath)
    assert chk.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 3
    assert chk.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    chk.close()
