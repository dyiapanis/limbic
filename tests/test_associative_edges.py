"""Associative links (co-retrieval reinforcement) — storage contract.

Real SQLiteStorage on tmp_path: formation, canonical ordering, cap,
read-time decay, prune-on-touch, user isolation.
"""
import sqlite3

import pytest

from limbic.store_sqlite import SQLiteStorage, _now_iso


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStorage(path=str(tmp_path / "t.db"), dims=1024, collection_name="t")
    yield s
    s.close()


def _mk_fact(store, user_id="u1", content="fact"):
    # Real row via the public write path — FKs stay honest.
    return int(store.store(content=content, user_id=user_id,
                           embedding=[0.01] * 1024,
                           metadata={"trust_score": 0.9}))


def _edges(store):
    return store._db.execute(
        "SELECT fact_a, fact_b, user_id, weight FROM fact_edges ORDER BY fact_a, fact_b"
    ).fetchall()


def test_formation_canonical_pairs(store):
    a, b, c = _mk_fact(store), _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([c, a, b], "u1")  # unsorted input on purpose
    rows = _edges(store)
    # canonical (a<b): exactly 3 unordered pairs, no reversed duplicates
    assert [(r["fact_a"], r["fact_b"]) for r in rows] == [(a, b), (a, c), (b, c)]
    assert all(r["weight"] == 1.0 for r in rows)
    assert all(r["user_id"] == "u1" for r in rows)


def test_increment_capped(store):
    a, b = _mk_fact(store), _mk_fact(store)
    for _ in range(25):
        store.reinforce_edges([a, b], "u1")
    rows = _edges(store)
    assert rows[0]["weight"] == pytest.approx(SQLiteStorage._EDGE_CAP)


def test_user_isolation(store):
    a, b = _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([a, b], "u1")
    # u2 sees nothing, forms separate rows, never sees u1's
    assert store.edge_neighbours(a, [b], "u2") == {}
    store.reinforce_edges([a, b], "u2")
    rows = {(r["fact_a"], r["fact_b"], r["user_id"]) for r in _edges(store)}
    assert rows == {(min(a, b), max(a, b), "u1"), (min(a, b), max(a, b), "u2")}
    # isolation both directions of the pair key
    assert store.edge_neighbours(b, [a], "u1") == {a: pytest.approx(1.0 / SQLiteStorage._EDGE_CAP)}


def test_decay_computed_at_read(store):
    import datetime as _dt
    a, b = _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([a, b], "u1")
    # Forge a 60-day-old reinforcement: two half-lives → 0.25x effective.
    stale = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=60)).isoformat()
    store._db.execute("UPDATE fact_edges SET last_reinforced=?", (stale,))
    n = store.edge_neighbours(a, [b], "u1")
    assert n[b] == pytest.approx(0.25 / SQLiteStorage._EDGE_CAP, rel=0.2)


def test_prune_on_touch(store):
    import datetime as _dt
    a, b = _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([a, b], "u1")
    # Decay far past the prune floor, then touch: the link dies, not decrements.
    stale = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=300)).isoformat()
    store._db.execute("UPDATE fact_edges SET weight=1.0, last_reinforced=?", (stale,))
    store.reinforce_edges([a, b], "u1")
    assert _edges(store) == []


def test_neighbours_normalised_and_bounded(store):
    a, b = _mk_fact(store), _mk_fact(store)
    for _ in range(30):
        store.reinforce_edges([a, b], "u1")
    n = store.edge_neighbours(a, [b], "u1")
    assert n[b] == pytest.approx(1.0, rel=1e-4)  # capped weight → normalised at the cap


def test_reinforce_never_raises_on_missing_ids(store):
    store.reinforce_edges([None, 5], "u1")  # no pairs form; no crash
    store.reinforce_edges([], "u1")
    store.reinforce_edges([7], "u1")
    assert _edges(store) == []


def test_fk_cascade_removes_edges_with_fact(store):
    # delete_fact must take the edges with it — verified manually in review,
    # pinned here.
    a, b = _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([a, b], "u1")
    assert len(_edges(store)) == 1
    store.delete_fact(a)
    assert _edges(store) == []


def test_prune_boundary_just_above_floor_survives(store):
    # eff == the prune floor exactly is unreachable in practice: read-time
    # decay nudges any elapsed moment below it, so == floor prunes. Pin the
    # real boundary: just-above-floor survives and increments.
    a, b = _mk_fact(store), _mk_fact(store)
    store.reinforce_edges([a, b], "u1")
    store._db.execute(
        "UPDATE fact_edges SET weight=?, last_reinforced=?",
        (SQLiteStorage._EDGE_PRUNE_FLOOR * 2.0, _now_iso()),
    )
    store.reinforce_edges([a, b], "u1")
    rows = _edges(store)
    assert rows and rows[0]["weight"] == pytest.approx(
        SQLiteStorage._EDGE_PRUNE_FLOOR * 2.0 + 1.0, rel=1e-2)