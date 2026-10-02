"""Continuing-correction links: schema, lifecycle, and bounded re-check.

Model-free — no ONNX downloads. Exercises the storage layer directly and the
decision logic through the storage contract, so it runs in CI and on a bare
checkout.
"""
import sqlite3

import pytest

from limbic.store_sqlite import SQLiteStorage

DIMS = 8


def _vec(seed: float) -> list[float]:
    return [seed] * DIMS


@pytest.fixture
def store(tmp_path):
    s = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    yield s
    s.close()


def _mkfact(store, i, user="alice", trust=0.5):
    return store.store(
        content=f"fact number {i}",
        user_id=user,
        embedding=_vec(float(i) / 10),
        metadata={"trust_score": trust, "created_at": "2026-09-01T00:00:00+00:00"},
    )


def test_contradictions_table_exists(store):
    cols = {r[1] for r in store._db.execute("PRAGMA table_info(contradictions)")}
    assert cols == {"newer_fact_id", "older_fact_id", "created_at", "nli_label", "resolved_at"}


def test_link_and_read_back(store):
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    store.link_contradiction(newer, older)

    open_links = store.unresolved_contradictions("alice", limit=10)
    assert len(open_links) == 1
    link = open_links[0]
    assert link["newer_fact_id"] == newer
    assert link["older_fact_id"] == older
    assert link["nli_label"] == "contradiction"
    # the join must surface both sides' content and the older fact's trust
    assert link["older_content"] == "fact number 2"
    assert link["newer_content"] == "fact number 1"
    assert link["older_trust"] == 0.5


def test_link_is_idempotent(store):
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    store.link_contradiction(newer, older)
    store.link_contradiction(newer, older)
    assert store.count_open_contradictions("alice") == 1


def test_resolve_closes_link(store):
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    store.link_contradiction(newer, older)
    store.resolve_contradiction(newer, older)

    assert store.unresolved_contradictions("alice", limit=10) == []
    assert store.count_open_contradictions("alice") == 0
    # resolved rows persist (audit trail), just not returned as open
    n = store._db.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
    assert n == 1


def test_resolved_link_cannot_be_reopened(store):
    """A resolved pair must not come back to life if the write repeats."""
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    store.link_contradiction(newer, older)
    store.resolve_contradiction(newer, older)
    store.link_contradiction(newer, older)          # INSERT OR IGNORE
    assert store.count_open_contradictions("alice") == 0


def test_cascade_on_fact_delete(store):
    """Deleting either fact removes the link — no orphan rows."""
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    store.link_contradiction(newer, older)
    store.delete_fact(older)
    n = store._db.execute("SELECT COUNT(*) FROM contradictions").fetchone()[0]
    assert n == 0
    assert store.unresolved_contradictions("alice", limit=10) == []


def test_limit_is_respected_and_oldest_first(store):
    older_a, older_b, older_c = _mkfact(store, 1), _mkfact(store, 2), _mkfact(store, 3)
    newer = _mkfact(store, 4)
    for o in (older_a, older_b, older_c):
        store.link_contradiction(newer, o)

    links = store.unresolved_contradictions("alice", limit=2)
    assert len(links) == 2                      # bounded
    assert [l["older_fact_id"] for l in links] == [older_a, older_b]   # oldest first


def test_scoped_to_older_facts_owner(store):
    """Scoping follows the OLDER fact — the one being corrected.

    Cross-user links are unreachable in practice: remember() only runs NLI
    against facts returned by a user-scoped search, so both ends share an owner.
    The scoping is asserted anyway so a future change to that search cannot
    silently widen the re-check to another user's store.
    """
    newer = _mkfact(store, 1, user="alice")
    older = _mkfact(store, 2, user="bob")
    store.link_contradiction(newer, older)

    assert store.unresolved_contradictions("alice", limit=10) == []
    assert len(store.unresolved_contradictions("bob", limit=10)) == 1


def test_ttl_expiry_deletes_only_open(store):
    newer, older = _mkfact(store, 1), _mkfact(store, 2)
    newer2, older2 = _mkfact(store, 3), _mkfact(store, 4)
    store.link_contradiction(newer, older)
    store.link_contradiction(newer2, older2)
    store.resolve_contradiction(newer2, older2)

    # cutoff in the future expires everything still open
    deleted = store.delete_expired_contradictions("2099-01-01T00:00:00+00:00")
    assert deleted == 1                          # only the open one
    assert store.count_open_contradictions("alice") == 0


def test_storage_rejects_unknown_column_still(store):
    """Regression guard: the whitelist must still reject junk after edits."""
    fid = _mkfact(store, 1)
    with pytest.raises(ValueError):
        store.update(fid, **{"pinned": True})
