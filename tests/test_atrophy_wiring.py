"""Atrophy-path wiring — the two bugs this file exists to prevent.

1. `is_atrophy_candidate` was called with `created_at` BOTH positionally and
   by keyword in the background scan. That raises TypeError, which the
   surrounding `except` logged at DEBUG — invisible in production — so the
   scan silently never marked a single fact. The two foreground paths always
   used keywords, so only one of three call sites was broken.

2. Temporal decay was non-idempotent, walking facts to ~44x below their
   correct value. Restoring walked values is a data operation, but the
   invariant tested here is the one that matters at runtime: a healthy fact
   must not be an atrophy candidate.

These assert on CALL SHAPE and OUTCOME, so a future refactor that reintroduces
a positional/keyword clash fails loudly instead of going quiet.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from limbic.trust import TrustEngine


@pytest.fixture()
def engine(tmp_path: Path) -> TrustEngine:
    return TrustEngine(str(tmp_path / "params.json"))


def test_signature_accepts_keyword_created_at(engine: TrustEngine):
    """The scan passes every argument by keyword — the signature must allow it."""
    sig = inspect.signature(TrustEngine.is_atrophy_candidate)
    params = list(sig.parameters)

    assert params[0] == "self"
    assert "trust_score" in params
    assert "last_retrieved" in params
    assert "created_at" in params

    # keyword-only invocation must work (the exact shape the scan uses)
    result = engine.is_atrophy_candidate(
        trust_score=0.5,
        last_retrieved=None,
        created_at="2026-09-08T00:00:00+00:00",
    )
    assert isinstance(result, bool)


def test_positional_plus_keyword_created_at_raises(engine: TrustEngine):
    """Documents the failure mode: the old call shape is a TypeError.

    This is deliberately asserting the *broken* shape fails — it pins down why
    the scan was dead, and will fail if someone "helpfully" makes the third
    positional parameter mean something else.
    """
    with pytest.raises(TypeError):
        engine.is_atrophy_candidate(0.009, None, False, created_at="2026-09-01T00:00:00+00:00")


def test_healthy_trust_is_not_a_candidate(engine: TrustEngine):
    """A fact at restored trust must NOT be marked — the guard that protects data."""
    created = "2026-09-08T00:00:00+00:00"
    for trust in (0.4665, 0.4559, 0.1097, 0.5):
        assert not engine.is_atrophy_candidate(
            trust_score=trust, last_retrieved=None, created_at=created
        ), f"trust {trust} wrongly flagged for atrophy"


def test_below_threshold_is_a_candidate(engine: TrustEngine):
    """The complement: genuinely low trust still gets swept. Not a no-op."""
    assert engine.is_atrophy_candidate(
        trust_score=0.005, last_retrieved=None, created_at="2026-09-08T00:00:00+00:00"
    )


def test_restored_values_clear_the_threshold(engine: TrustEngine):
    """Regression guard for the restore: the observed victim values, before and
    after. If a future decay change re-collapses them, this fails.

    C4 recalibration moved the operating point 0.01 → 0.05: the
    band trusts (0.010–0.035) that sat harmlessly above the old gate are NOW
    candidates by design (the measured zombie band is actionable), while
    healthy trusts stay safe.
    """
    threshold = engine.atrophy_threshold()
    created = "2026-09-08T00:00:00+00:00"

    # Pre-recalibration "safe" band values are NOW intentional sweep
    # candidates (documented in C4: the mark→grace→reap guard decides who
    # goes; these were the audit's zombie-band population).
    atrophy_band = [0.010155, 0.010258, 0.013667, 0.035897, 0.045, 0.0499]
    # Healthy trusts: never candidates, before or after the restore.
    safe = [0.05, 0.051, 0.1097, 0.4559, 0.4665, 0.4774]

    for v in atrophy_band:
        assert engine.is_atrophy_candidate(
            trust_score=v, last_retrieved=None, created_at=created
        ), f"{v} is below the recalibrated {threshold} and must be flagged"

    for v in safe:
        assert not engine.is_atrophy_candidate(
            trust_score=v, last_retrieved=None, created_at=created
        ), f"{v} is at/above {threshold} and must NOT be flagged"
