#!/usr/bin/env python3
"""Ignore-signal harness — measured NO-GO for the text-channel ignore detector.

Question: should memory penalise facts that are repeatedly co-surfaced with a
busy topic but never earn a turn-boundary signal (no continuation, no
correction)? The trust engine's only positive channel is textual overlap, so
an ignore signal on that channel would target the "flat trust + hot riding"
cohort. This harness measures what that cohort actually contains.

Fixture: eval/ignore_signal_cohort.json — 135 facts (retrieval_count >= 8,
trust in the flat band across all six live fleet stores), hand-labelled by
the store owner. Labels: rider = conversational fragment / banner / task
echo; protected = rule / reference / log-fact.

Verdict (2026-10-23): the cohort is ~21% riders. The signal fails the 0.90
precision bar at 0.207 — it would drift the store's rulebook toward the
floor, because standing records NEVER earn textual continuations by nature.
Measured NO-GO; the mechanism is not implemented. Recorded here with the
cohort so no one re-derives it by surprise.

    Run: /path/to/env-venv-python eval/ignore_signal_eval.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PRECISION_BAR = 0.90

BANNERS = ("[context compaction", "[out-of-band", "[important:",
           "gateway message origin")
OPENERS = ("can you", "can i", "do you", "why ", "why,", "how ", "what ",
           "wtf", "not yet", "i want", "i need", "two things", "is that",
           "are you", "should we", "do we", "you recommended", "what i mean")


def fragment_gate(prefix: str) -> bool:
    """Write-time fragment-detector proxy: banner wrappers + interrogative /
    first-person intent openers, prefix-anchored to the first 80 chars."""
    t = prefix.strip().lower()
    if t.startswith(BANNERS):
        return True
    return any(t[:80].startswith(o) for o in OPENERS)


def main() -> int:
    fx = json.load(open(os.path.join(HERE, "ignore_signal_cohort.json")))
    facts = fx["facts"]
    n = len(facts)
    riders = [f for f in facts if f["lbl"] == "rider"]
    prot = [f for f in facts if f["lbl"] == "protected"]

    naive_p = len(riders) / n
    print(f"cohort: {n} facts ({fx['riders']} riders / {fx['protected']} protected), six fleet stores")
    print(f"[1] ignore-signal target class (flat-trust + hot-riding): "
          f"precision={naive_p:.3f}  ->  "
          f"{'PASS' if naive_p >= PRECISION_BAR else 'FAIL'} vs bar {PRECISION_BAR}")
    print("    verdict: the ignore signal as designed is NO-GO. The cohort is "
          "dominated by the store's own rulebook/log — standing records never "
          "earn textual continuations, so a text-channel ignore signal drifts "
          "the archive toward the floor, not the junk.")

    caught = [f for f in facts if f["flagged_by_gate"]]
    hit = [f for f in caught if f["lbl"] == "rider"]
    fp = [f for f in caught if f["lbl"] == "protected"]
    gp = len(hit) / max(1, len(caught))
    gr = len(hit) / max(1, len(riders))
    print(f"[2] write-time fragment gate (cohort proxy): flagged={len(caught)}  "
          f"precision={gp:.3f}  recall={gr:.3f}  ->  "
          f"{'PASS' if gp >= PRECISION_BAR else 'FAIL'}")
    print("    verdict: successor candidate — fragment hygiene at remember() "
          "time, not a recall-time ignore signal. Fixture-optimism caveat "
          "applies: gate classes were derived from this same cohort; a live "
          "phase (gate evidence rows on real writes, no trust movement) must "
          "confirm the false-block rate on genuine question-phrased facts "
          "before any GO.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())