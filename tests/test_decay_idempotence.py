"""Decay idempotence — the bug this file exists to prevent.

`apply_decay` anchored only on `last_retrieved`. Because decay runs every turn,
the whole elapsed interval was re-billed on each pass: 27 days of real time
charged ~5 times, taking a healthy 0.69 to 0.036 (10x the correct value) and
pushing facts below the atrophy threshold inside one session. Five facts were
destroyed before this was caught.

The existing suite called `apply_decay` exactly once, which is why it passed.
Every test here calls it repeatedly — the only shape that catches
non-idempotent decay.

TESTING NOTE: arguments here mirror PRODUCTION, where all three timestamps are
populated from the row (last_retrieved, created_at, updated_at). Testing with
`created_at=None` makes the anchor list shorter than production and lets a
regression hide — a mutation test proved that mistake, which is why every case
below supplies a realistic created_at.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from limbic.trust import TrustEngine


@pytest.fixture()
def engine(tmp_path: Path) -> TrustEngine:
    return TrustEngine(str(tmp_path / "params.json"))


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_repeated_decay_at_same_instant_is_idempotent(engine: TrustEngine):
    """The regression. Ten passes over an already-charged interval == one pass.

    Production shape: a fact created long ago, retrieved recently, decayed
    before. The buggy version re-billed the 10-day-old retrieval every turn.
    """
    last_retrieved = _days_ago(10)
    created_at = _days_ago(60)

    once = engine.apply_decay(0.9, last_retrieved,
                              created_at=created_at, decayed_at=last_retrieved)

    value = once
    for _ in range(10):
        value = engine.apply_decay(value, last_retrieved,
                                   created_at=created_at, decayed_at=_now())

    assert value == pytest.approx(once, abs=1e-9), (
        f"decay compounded: one pass={once:.6f}, ten passes={value:.6f}"
    )


def test_decay_charges_elapsed_interval_once_not_per_call(engine: TrustEngine):
    """27 days of elapsed time must cost ~0.54x, not 0.54^5.

    The motivating case: a fact created 2026-08-11, evaluated 27 days later.
    """
    anchor = _days_ago(27)
    created = _days_ago(27)
    half_life = engine.get("decay_half_life_days")

    expected = 0.6892 * (0.5 ** (27 / half_life))
    got = engine.apply_decay(0.6892, anchor, created_at=created, decayed_at=anchor)

    assert got == pytest.approx(expected, rel=0.02)
    assert got > 0.25, f"decayed too far: {got:.4f} (correct ~{expected:.4f})"


def test_decay_actually_decays_over_real_time(engine: TrustEngine):
    """Idempotence must not mean 'never decays'."""
    anchor = _days_ago(30)
    result = engine.apply_decay(1.0, anchor, created_at=anchor, decayed_at=anchor)
    assert result == pytest.approx(0.5, abs=0.03)


def test_anchor_is_the_latest_signal(engine: TrustEngine):
    """A retrieval AFTER the last decay pass must move the anchor forward."""
    old_decay = _days_ago(60)
    recent_retrieval = _days_ago(1)

    stale = engine.apply_decay(1.0, _days_ago(60),
                               created_at=old_decay, decayed_at=old_decay)
    refreshed = engine.apply_decay(1.0, recent_retrieval,
                                   created_at=old_decay, decayed_at=old_decay)

    assert refreshed > stale
    assert refreshed > 0.95


def test_a_decay_pass_advances_the_anchor(engine: TrustEngine):
    """After decay runs, a second call for the SAME interval charges nothing.

    Passes `now` as decayed_at, exactly as the caller does after a write.
    """
    last_retrieved = _days_ago(30)
    created = _days_ago(30)

    first = engine.apply_decay(1.0, last_retrieved,
                               created_at=created, decayed_at=last_retrieved)
    second = engine.apply_decay(first, last_retrieved,
                                created_at=created, decayed_at=_now())

    assert second == pytest.approx(first, abs=1e-9)


def test_no_timestamp_returns_unchanged(engine: TrustEngine):
    assert engine.apply_decay(0.5, None, created_at=None, decayed_at=None) == 0.5


def test_backward_compatible_call_without_decayed_at(engine: TrustEngine):
    """Legacy callers (2 positional args) still decay rather than crash."""
    result = engine.apply_decay(1.0, _days_ago(30))
    assert result == pytest.approx(0.5, abs=0.03)


def test_fleet_survivor_scenario_no_longer_threatens_atrophy(engine: TrustEngine):
    """Repeated prefetch must not walk trust below threshold.

    8 facts sat at 0.0359 with a 3-day-old anchor. Under the bug, ~19 turns
    crossed the 0.01 atrophy threshold. With a fixed anchor, repeat traffic
    must hold steady once the day's decay is charged.
    """
    threshold = engine.get("atrophy_threshold")
    last_retrieved = _days_ago(3)
    created = _days_ago(32)

    trust = engine.apply_decay(0.0359, last_retrieved,
                               created_at=created, decayed_at=last_retrieved)
    charged = trust

    for _ in range(40):
        trust = engine.apply_decay(trust, last_retrieved,
                                   created_at=created, decayed_at=_now())

    assert trust == pytest.approx(charged, abs=1e-9)
    assert trust > threshold, (
        f"40 turns drove trust to {trust:.6f}, below atrophy threshold {threshold}"
    )

def test_zero_half_life_disables_decay(tmp_path: Path):
    """half_life <= 0 is the documented kill switch and must be honoured.

    Built from a params file (not engine._set, which clamps to the default
    param min of 7 days and would silently re-enable decay).
    """
    params = tmp_path / "zero.json"
    params.write_text(json.dumps(
        {"decay_half_life_days": {"value": 0, "min": 0, "max": 90}}
    ))
    engine = TrustEngine(str(params))
    assert engine.get("decay_half_life_days") == 0

    anchor = _days_ago(90)
    assert engine.apply_decay(1.0, anchor, created_at=anchor, decayed_at=anchor) == 1.0


def test_created_at_alone_decays_an_orphan(engine: TrustEngine):
    """A never-retrieved fact has no last_retrieved or decayed_at.

    created_at is its only anchor, so dropping it would freeze such facts at
    full trust forever.
    """
    created = _days_ago(30)
    result = engine.apply_decay(1.0, None, created_at=created, decayed_at=None)
    assert result == pytest.approx(0.5, abs=0.03), (
        f"orphan did not decay from created_at: {result:.4f}"
    )


def test_created_at_alone_is_idempotent_with_decayed_at(engine: TrustEngine):
    """Same orphan, but with a prior decay pass — must not re-charge."""
    created = _days_ago(30)
    first = engine.apply_decay(1.0, None, created_at=created, decayed_at=created)
    second = engine.apply_decay(first, None, created_at=created, decayed_at=_now())
    assert second == pytest.approx(first, abs=1e-9)
