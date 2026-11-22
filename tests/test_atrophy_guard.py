"""C4 atrophy recalibration tests — plan §C4.

Three jobs:
1. Pin the recalibrated operating point: threshold 0.05, grace 7 (the measured
   zombie band; the review window between mark and reap).
2. Document the root cause of zero-flags-ever: the band trusts were never
   candidates at the old 0.01 operating point and ARE candidates at 0.05.
3. Assert the reap guard end-to-end on the store: a reaped fact is gone
   (facts row + vec_index row); protected facts — retrieval_count >= 5,
   older_fact_id in an unresolved contradiction link, or excluded via
   reap_guard_excludes — stay intact, mark parked, forever until lifted.
"""
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import DEFAULT_PARAMS, TrustEngine  # noqa: E402

DIMS = 8

# Live fleet zombie-band trusts measured 2026-11-22 across 6 profiles
# (post-C2/C3 heal): 14 facts in [0.0139, 0.0488], zero below 0.01 — ever.
ZOMBIE_BAND_TRUSTS = [0.0139, 0.022, 0.0306, 0.0385, 0.0409, 0.0431, 0.0488]


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    yield s
    s.close()


@pytest.fixture()
def engine(tmp_path):
    return TrustEngine(str(tmp_path / "params.json"))


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _mkfact(store, i, rc=0, trust=0.5, user_id="grace"):
    return store.store(
        f"fact number {i} about topic {i % 3}",
        user_id=user_id,
        embedding=_norm([float((i * 7 + j) % 11 + 1) for j in range(DIMS)]),
        metadata={"trust_score": trust, "pinned": False, "retrieval_count": rc},
    )


def _old_mark(store, fid, days=8):
    """Stamp atrophy_marked_at backdated past a 7-day grace, in the _now_iso
    format the mark path itself uses."""
    iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    store._write_txn(
        lambda: store._db.execute(
            "UPDATE facts SET atrophy_marked_at = ? WHERE fact_id = ?", (iso, fid)
        )
    )


def _cutoff(grace_days=7):
    return (datetime.now(timezone.utc) - timedelta(days=grace_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _marked_at(store, fid):
    row = store._db.execute(
        "SELECT atrophy_marked_at FROM facts WHERE fact_id = ?", (fid,)
    ).fetchone()
    return row["atrophy_marked_at"] if row else None


# ── 1. Operating point ──────────────────────────────────────────────────────

def test_recalibrated_operating_point():
    """C4: threshold 0.01 → 0.05, grace 0 → 7. Grace ≥ 7 is the review
    window — the v1 mark→confirm→reap had no actor; grace is the mechanism."""
    assert DEFAULT_PARAMS["atrophy_threshold"]["value"] == 0.05
    assert DEFAULT_PARAMS["atrophy_grace_days"]["value"] == 7
    assert DEFAULT_PARAMS["max_atrophy_per_turn"]["value"] == 10


def test_band_trusts_candidates_at_005_not_at_001(engine, tmp_path):
    """Root-cause pin: every measured zombie-band trust WAS candidate-invisible
    at the old 0.01 operating point (why zero flags ever fired) and IS one at
    the recalibrated 0.05 (why the band is now actionable)."""
    created = "2026-09-01T00:00:00+00:00"
    assert all(engine.is_atrophy_candidate(trust_score=t, last_retrieved=None,
                                           created_at=created)
               for t in ZOMBIE_BAND_TRUSTS), "0.05 must reach the measured band"

    old_params = str(tmp_path / "old.json")
    with open(old_params, "w") as f:
        f.write('{"atrophy_threshold": {"value": 0.01}}')
    old = TrustEngine(old_params)
    assert old.is_atrophy_candidate(0.005, None, created_at=created), (
        "0.01 must still sweep fully-decayed facts (guard against over-recalibration)")
    assert not any(old.is_atrophy_candidate(trust_score=t, last_retrieved=None,
                                            created_at=created)
                   for t in ZOMBIE_BAND_TRUSTS), (
        "0.01 never reached the band — the zero-flags root cause, pinned")


# ── 2. Grace window ─────────────────────────────────────────────────────────

def test_fresh_mark_survives_grace_and_reaps_after(store):
    """A mark stamped now is inside the 7-day grace; past grace it reaps."""
    fid = _mkfact(store, 1)
    store.mark_atrophy(fid)
    assert store.sweep_atrophy(_cutoff(grace_days=7), limit=10) == 0
    assert store.get(fid) is not None

    _old_mark(store, fid, days=8)
    deleted = store.sweep_atrophy(_cutoff(grace_days=7), limit=10)
    assert deleted == 1
    assert store.get(fid) is None


def test_grace_zero_reaps_the_mark_within_its_turn(store):
    """Pins the v1 foot-gun: grace=0 deletes a fresh mark the same turn
    (isoformat stamp '…56.789+00:00' sorts below a '…56Z' cutoff) — the
    mark→confirm→reap 'confirm' had no actor. Recalibrated grace is 7."""
    fid = _mkfact(store, 1)
    store.mark_atrophy(fid)
    deleted = store.sweep_atrophy(_cutoff(grace_days=0), limit=10)
    assert deleted == 1
    assert store.get(fid) is None


# ── 3. Reap guard (protected facts stay intact) ──────────────────────────────

def test_guard_retrieval_count_ge_5(store):
    """rc >= 5 is never reaped: it surfaced 5+ times. rc < 5 reaps past grace."""
    protected = _mkfact(store, 1, rc=5)
    reaped = _mkfact(store, 2, rc=4)
    for fid in (protected, reaped):
        store.mark_atrophy(fid)
        _old_mark(store, fid)

    deleted = store.sweep_atrophy(_cutoff(), limit=10)
    assert deleted == 1
    assert store.get(protected) is not None
    assert store.get(reaped) is None
    assert _marked_at(store, protected) is not None, "protected mark stays parked"


def test_guard_unresolved_contradiction_link(store):
    """older_fact_id in an OPEN contradiction link is never reaped; once the
    link resolves, the same fact reaps past grace."""
    older = _mkfact(store, 1)
    newer = _mkfact(store, 2, trust=0.5)
    store.mark_atrophy(older)
    _old_mark(store, older)
    store.link_contradiction(newer, older)
    assert store.sweep_atrophy(_cutoff(), limit=10) == 0
    assert store.get(older) is not None

    store.resolve_contradiction(newer, older)
    deleted = store.sweep_atrophy(_cutoff(), limit=10)
    assert deleted == 1
    assert store.get(older) is None


def test_resolved_contradiction_does_not_protect(store):
    resolved_links = _mkfact(store, 1)
    store.mark_atrophy(resolved_links)
    _old_mark(store, resolved_links)
    store.link_contradiction(_mkfact(store, 2), resolved_links)
    store.resolve_contradiction(2, resolved_links)
    assert store.sweep_atrophy(_cutoff(), limit=10) == 1


def test_guard_exclude_ids(store):
    """reap_guard_excludes (chain-collateral, C2/C3 replay-blessed) is never
    reaped; the same fact without the exclusion is."""
    excluded = _mkfact(store, 1)
    store.mark_atrophy(excluded)
    _old_mark(store, excluded)

    assert store.sweep_atrophy(_cutoff(), limit=10, exclude_ids={excluded}) == 0
    assert store.get(excluded) is not None

    deleted = store.sweep_atrophy(_cutoff(), limit=10)
    assert deleted == 1
    assert store.get(excluded) is None


def test_reap_removes_fact_and_vec_row(store):
    """Post-run gate: the reaped fact is gone from facts AND vec_index; the
    protected neighbour is fully intact."""
    reaped = _mkfact(store, 1, rc=0)
    protected = _mkfact(store, 3, rc=7)
    store.mark_atrophy(reaped)
    store.mark_atrophy(protected)
    _old_mark(store, reaped)
    _old_mark(store, protected)

    store.sweep_atrophy(_cutoff(), limit=10)
    assert store.get(reaped) is None
    assert store._db.execute(
        "SELECT COUNT(*) FROM vec_index WHERE rowid = ?", (reaped,)
    ).fetchone()[0] == 0
    assert store.get(protected) is not None
    assert store._db.execute(
        "SELECT COUNT(*) FROM vec_index WHERE rowid = ?", (protected,)
    ).fetchone()[0] == 1


def test_max_atrophy_per_turn_cap_preserved(store):
    """max_atrophy_per_turn=10 is preserved: 12 unprotected marks → ≤10 go."""
    fids = [_mkfact(store, i, rc=0) for i in range(12)]
    for fid in fids:
        store.mark_atrophy(fid)
        _old_mark(store, fid)
    deleted = store.sweep_atrophy(_cutoff(), limit=10)
    assert deleted == 10


def test_user_scoping_still_applies(store):
    """The C4 guard must not have broken the per-user scoping contract."""
    mine = _mkfact(store, 1, rc=0, user_id="other")
    store.mark_atrophy(mine)
    _old_mark(store, mine)
    assert store.sweep_atrophy(_cutoff(), limit=10, user_id="grace") == 0
    assert store.sweep_atrophy(_cutoff(), limit=10, user_id="other") == 1


def test_unmark_atrophy_clears_recovered_fact(store):
    """C4 unmark: the sweep does not re-check trust, so a recovered mark is
    cleared by the decay pass — storage side, the mark must go to NULL and
    the fact is then reapable only on a FUTURE mark (fresh 7-day grace)."""
    fid = _mkfact(store, 1, rc=2, trust=0.02)
    store.mark_atrophy(fid)
    assert _marked_at(store, fid) is not None

    store.update(fid, trust_score=0.2)  # organic recovery (boost/heal)
    store.unmark_atrophy(fid)
    assert _marked_at(store, fid) is None
    assert store.sweep_atrophy(_cutoff(grace_days=-1), limit=10) == 0
