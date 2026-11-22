"""C2 chain-replay gate — correction_floor vs unrestrained correction dock.

Replays the committed real-chain twins (eval/chain_replay.json fleet_fact_id
87/88, twin #87/#88) under both schedules and asserts the mechanism gates by
measurement:

- the unrestrained dock reproduces the fleet crater (post-dock trust to ~0);
- the floored dock cannot go below min(old, correction_floor) at any dock;
- floored-schedule links still resolve within correction_link_ttl (no
  repeat-clamp starvation — floor docking stops at the floor, which the tick
  predicate trusts as "landed").

Numbers are the 2026-11-22 measured ones; margins stay wide so param defaults
can evolve without brittleness. Run `python eval/chain_replay.py` for the
twin table this asserts.
"""
from pathlib import Path

from eval.chain_replay import (  # noqa: I100
    FLOOR,
    PENALTY,
    dock_floored as dock,
    replay_chain,
    twin_table,
)
from limbic import trust as trust_mod

# Measured evidence: the two twins twin #87/#88 were ground to these finals by
# the unrestrained dock (baseline of record).
TWINS = ("victim_twin_A", "victim_twin_B")


def test_fixture_carries_the_twin_pair():
    rows = {r["label"]: r for r in twin_table()}
    assert TWINS[0] in rows and TWINS[1] in rows


def test_baseline_reproduces_the_crater():
    rows = {r["label"]: r for r in twin_table()}
    for label in TWINS:
        row = rows[label]
        # actual pre-C2 final trust is in the 0.02 crater band
        assert row["actual_final_trust"] <= 0.03
        # the unrestrained dock regrinds through any floor to ~0
        assert row["baseline_min_post_dock_overall"] <= 0.05


def test_floored_dock_holds_the_crater_band():
    rows = {r["label"]: r for r in twin_table()}
    for label in TWINS:
        row = rows[label]
        # dock damage eliminated: post-dock trust stays a wide margin above
        # the crater band (0.26 measured: decay of the 0.30 dock floor)
        sel = row["selected_min_post_dock_overall"]
        assert sel is not None and sel >= 0.20
        assert sel > rows[label]["baseline_min_post_dock_overall"]


def test_floored_dock_never_docks_below_the_floor():
    # structural: the dock output never undercuts min(old, floor)
    values = [0.01, 0.05, 0.24, 0.28, 0.30, 0.40, 0.50, 0.75, 1.0]
    for old in values:
        if old <= FLOOR:
            assert dock(old, 0) == old          # dock is a no-op at/below floor
        else:
            assert dock(old, 0) == max(FLOOR, old - PENALTY)
        assert dock(old, 0) >= min(old, FLOOR)


def test_selected_schedule_resolves_links_within_ttl():
    for row in twin_table():
        if row["n_hits"] == 0:
            continue  # no links — nothing to starve
        assert row["selected_links_resolved_within_ttl"] is True


def test_floor_param_is_registered_and_cited():
    # the mechanism param exists with its measured default and floor value
    param = trust_mod.DEFAULT_PARAMS["correction_floor"]
    assert param["value"] == 0.30
    # resolution predicate trusts the dock floor as "landed"
    assert PENALTY < param["value"] <= 0.30
