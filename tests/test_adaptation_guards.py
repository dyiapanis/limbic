"""Min-sample guards on TrustEngine._adapt (#19).

A handful of observations must not move adaptive parameters, and
below-threshold batches must accumulate across cycles instead of being
cleared every cycle.
"""
import pytest

from limbic.trust import TrustEngine


@pytest.fixture
def engine(tmp_path):
    e = TrustEngine(str(tmp_path / "params.json"))
    e._adaptation_interval = 1  # adapt on every tick — test speed only
    return e


def _ignore_signal(engine):
    engine.record_signal(1, was_continuation=False, was_correction=False)


def test_few_signals_do_not_move_floor(engine):
    floor0 = engine.get("relevance_floor")
    for _ in range(3):
        _ignore_signal(engine)
    engine.tick()
    assert engine.get("relevance_floor") == floor0


def test_signals_accumulate_until_threshold(engine):
    floor0 = engine.get("relevance_floor")
    for _ in range(3):
        _ignore_signal(engine)
    engine.tick()
    assert len(engine._signals) == 3  # below threshold → kept, not cleared
    for _ in range(7):
        _ignore_signal(engine)
    engine.tick()  # 10 signals, 100% ignore-rate > 0.5 → floor +0.02
    assert engine.get("relevance_floor") == floor0 + 0.02
    assert engine._signals == []  # adapted cycle consumed the sample


def test_single_maturity_does_not_move_t_half(engine):
    t0 = engine.get("maturation_t_half")
    engine.record_maturity(1.0)  # one observation — previously moved ±12h
    engine.tick()
    assert engine.get("maturation_t_half") == t0
    assert len(engine._maturity_times) == 1  # accumulates for the next cycle


def test_five_maturities_move_t_half(engine):
    t0 = engine.get("maturation_t_half")  # default 168h
    for _ in range(5):
        engine.record_maturity(1.0)  # avg 1h << t_half/2 → t_half −12
    engine.tick()
    assert engine.get("maturation_t_half") == t0 - 12
    assert engine._maturity_times == []  # consumed