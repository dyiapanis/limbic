#!/usr/bin/env python3
"""Chain replay — C2 correction-floor mechanism gate (measured 2026-11-22).

Replays the committed real-chain fixture (eval/chain_replay.json) under two
schedules and asserts the mechanism gates by measurement:

- baseline: `max(0, trust - correction_penalty)` — the pre-C2 dock, no floor.
- selected: the floored dock `max(min(trust, correction_floor), trust - penalty)`
  plus tick resolution "landed when trust <= correction_floor".

Observables per replay (identical decay/tick skeleton across schedules, so the
delta isolates the dock behaviour):
- `min_post_dock_overall` — the lowest trust right after any correction dock:
  the crater the schedule's docking can inflict (baseline regrinds through the
  floor to ~0.0 on the twins; the floored schedule cannot).
- `final_trust` — end-of-replay value (includes natural decay between and after
  hits; recovery from recalls/continuations is NOT modeled — conservative).
- `resolution_days` / `links_resolved_within_ttl` — every hit's link resolves
  inside correction_link_ttl (catches starvation under the floored schedule).

Simplification notes: victim starts at 0.5; decay charges daily off a
write-advanced anchor (mirrors updated_at); one maintenance tick per day; NLI
assumed to keep confirming (worst case for resolution timing); real stores
recover somewhat faster than this model, so selected numbers are lower bounds.
The per-store full harness that produced the selection measurement lives in
the private ops tree (limbic-ops/c2-replay-20261122); this module carries the
committed twins only.

Run as a script to print the twin table; tests/test_chain_replay.py asserts it.
"""
from __future__ import annotations

import json
import os
import sys
import types

# Script-mode runs (`python eval/chain_replay.py`) get eval/ as sys.path[0].
# `import limbic` needs a namespace-only alias for the repo dir (the checkout
# dir is not importable as `limbic` by name). Namespace-only, NOT exec'd: the
# package root pulls provider → agent.*, which has no shim outside Hermes —
# subtree imports (limbic.trust) are the pattern tests/conftest.py enables.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
if "limbic" not in sys.modules:
    _pkg = types.ModuleType("limbic")
    _pkg.__path__ = [_REPO]
    sys.modules["limbic"] = _pkg

from limbic import trust as trust_mod  # noqa: E402

PENALTY = float(trust_mod.DEFAULT_PARAMS["correction_penalty"]["value"])
FLOOR = float(trust_mod.DEFAULT_PARAMS["correction_floor"]["value"])
HALF_LIFE_DAYS = 30.0
TTL_DAYS = 30
MAX_PER_TURN = 3
INITIAL = 0.5

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chain_replay.json")


def dock_baseline(old, _n):
    return max(0.0, old - PENALTY)


def dock_floored(old, _n):
    return max(min(old, FLOOR), old - PENALTY)


def landed_baseline(t):
    return t < 0.25


def landed_floored(t):
    return t <= FLOOR


SCHEDULES = {
    "baseline": {"dock": dock_baseline, "landed": landed_baseline},
    "selected": {"dock": dock_floored, "landed": landed_floored},
}


def _decayed(value: float, days: float) -> float:
    return value * (0.5 ** (max(days, 0.0) / HALF_LIFE_DAYS))


def replay_chain(chain: dict, schedule_name: str) -> dict | None:
    """Deterministic replay of one victim's recorded hit pattern.

    The victim starts at 0.5; each recorded hit opens a link and docks the
    victim once; open links are re-checked on daily ticks (oldest first,
    bounded) and expire at the link TTL.
    """
    sk = SCHEDULES[schedule_name]
    hits = sorted(chain["hits"], key=lambda h: h["day"])
    if not hits:
        return None
    horizon = hits[-1]["day"] + TTL_DAYS + 2

    trust = INITIAL
    anchor = 0.0
    open_links: list[int] = []           # unprocessed hit indices
    resolutions: dict[int, float | None] = {}
    dock_outputs: list[float] = []       # trust right after every dock

    day = 1.0
    while day <= horizon + 1:
        # 1) write-time docks: hits landing during the past day, in order
        for idx, hit in enumerate(hits):
            if hit["day"] < day and idx not in resolutions and idx not in open_links:
                pre = _decayed(trust, day - anchor)
                new = sk["dock"](pre, idx)
                dock_outputs.append(new)
                trust = new
                anchor = day
                open_links.append(idx)
        # 2) maintenance tick (oldest first, bounded per turn)
        budget = MAX_PER_TURN
        for idx in sorted(open_links, key=lambda i: hits[i]["day"]):
            if budget <= 0:
                break
            pre = _decayed(trust, day - anchor)
            if sk["landed"](pre):
                resolutions[idx] = day
                open_links.remove(idx)
            else:
                new = sk["dock"](pre, idx)
                dock_outputs.append(new)
                trust = new
                anchor = day
            budget -= 1
        # 3) expiry
        for idx in list(open_links):
            if day - hits[idx]["day"] > TTL_DAYS:
                resolutions[idx] = None  # expired, never resolved
                open_links.remove(idx)
        day += 1.0

    for idx in open_links:
        resolutions[idx] = None

    return {
        "min_post_dock_overall": round(min(dock_outputs), 4),
        "final_trust": round(trust, 4),
        "links_resolved_within_ttl": all(d is not None for d in resolutions.values()),
        "resolution_days": {str(idx): (round(resolutions[idx], 2)
                                       if resolutions[idx] is not None else None)
                            for idx in sorted(resolutions)},
    }


def load_fixture() -> dict:
    with open(FIXTURE) as f:
        return json.load(f)


def twin_table() -> list[dict]:
    """One row per fixture chain: measured gate numbers, baseline vs selected."""
    fx = load_fixture()
    rows = []
    for chain in fx["chains"]:
        base = replay_chain(chain, "baseline")
        sel = replay_chain(chain, "selected")
        rows.append({
            "label": chain["label"],
            "retrieval_count": chain["retrieval_count"],
            "n_hits": len(chain["hits"]),
            "actual_final_trust": chain["actual_final_trust"],
            "baseline_min_post_dock_overall":
                base["min_post_dock_overall"] if base else None,
            "selected_min_post_dock_overall":
                sel["min_post_dock_overall"] if sel else None,
            "baseline_final_trust": base["final_trust"] if base else None,
            "selected_final_trust": sel["final_trust"] if sel else None,
            "selected_links_resolved_within_ttl":
                bool(sel and sel["links_resolved_within_ttl"]),
            "baseline_links_resolved_within_ttl":
                bool(base and base["links_resolved_within_ttl"]),
        })
    return rows


if __name__ == "__main__":
    for row in twin_table():
        print(row)
