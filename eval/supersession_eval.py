#!/usr/bin/env python3
"""Limbic supersession eval — the correction-pair acceptance gate (#19).

recall_eval.py --contested measures boundary confusion between DIFFERENT
facts; supersession targets a different failure: a stored correction losing
to the fact it corrects. This harness builds N stale→correction pairs
through the REAL write path (NLI links them), then asks each pair's
question. Reports:

  correction_top1 — the correction is the #1 result (the mechanism's purpose)
  stale_present  — the superseded fact appears anywhere in top-5
  links_formed   — write-time NLI actually linked the pairs

Deterministic corpus, real ONNX embedder + NLI, same runner as recall_eval.
Usage: eval/supersession_eval.py [--home DIR]
"""
import json
import os
import sys
import tempfile
import time

UID = "eval-user"
K = 5

# (domain, stale, correction, query) — near-identical sentence pairs with a
# swapped value, so NLI sees a genuine contradiction; domains are disjoint so
# pairs can't interfere with each other.
PAIRS = [
    ("fonts", "Her favourite font is JetBrains Mono",
     "Actually, her favourite font is Iosevka",
     "what is her favourite font?"),
    ("coffee", "The morning coffee is brewed with a dark Italian roast",
     "Actually, the morning coffee is a light Ethiopian roast",
     "which roast does the morning coffee use?"),
    ("phone", "The phone in the hallway is a corded landline",
     "Actually, the phone in the hallway is a mobile handset",
     "what kind of phone is in the hallway?"),
    ("car", "The family car is a silver estate wagon",
     "Actually, the family car is a blue hatchback",
     "what is the family car?"),
    ("desk", "The desk lamp is a warm filament bulb",
     "Actually, the desk lamp is a cool daylight LED",
     "what kind of bulb is the desk lamp?"),
    ("plant", "The office plant is a fern on the windowsill",
     "Actually, the office plant is a succulent on the shelf",
     "what is the office plant?"),
    ("music", "The alarm tone is a radio jingle",
     "Actually, the alarm tone is a soft chime",
     "what does the alarm sound like?"),
    ("meeting", "The team meeting is on Wednesday at noon",
     "Actually, the team meeting is on Thursday at three",
     "when is the team meeting?"),
    ("route", "The commute goes through the tunnel road",
     "Actually, the commute goes over the bridge road",
     "which road does the commute take?"),
    ("lunch", "Lunch on Fridays is soup at the counter",
     "Actually, lunch on Fridays is salad at the desk",
     "what is Friday lunch?"),
    ("editor", "The editor theme is a dark scheme at night",
     "Actually, the editor theme is a light scheme at night",
     "which editor theme is used at night?"),
    ("gym", "The gym session is a morning swim",
     "Actually, the gym session is an evening climb",
     "what is the gym session?"),
]


def _arg(name: str, default: str) -> str:
    argv = sys.argv[1:]
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else default
    return default


def main() -> int:
    home = _arg("--home", "") or tempfile.mkdtemp(prefix="limbic-supersession-")
    if not os.path.isdir(home):
        os.makedirs(home, exist_ok=True)
    print(f"eval home: {home}")

    # Same isolated-mode lane as recall_eval: insert paths in-process.
    sys.path.insert(0, os.environ.get("HERMES_AGENT_ROOT")
                    or os.path.expanduser("~/.hermes/hermes-agent"))
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from limbic.provider import LimbicMemoryProvider

    t0 = time.time()
    p = LimbicMemoryProvider()
    p.initialize("eval-session", hermes_home=home, user_id=UID, agent_identity="eval")
    p._get_user_id = lambda sid: UID
    print(f"provider initialised ({time.time() - t0:.1f}s)")

    # Ingest — real write path; NLI links each contradiction pair.
    t0 = time.time()
    for domain, stale, correction, _q in PAIRS:
        r1 = p.remember(stale, user_id=UID)
        if r1.get("error"):
            print(f"STORE FAILED: {r1['error']}: {stale}", file=sys.stderr)
            return 1
        r2 = p.remember(correction, user_id=UID)
        if r2.get("error"):
            print(f"STORE FAILED: {r2['error']}: {correction}", file=sys.stderr)
            return 1
    print(f"ingested {2 * len(PAIRS)} facts ({time.time() - t0:.1f}s)")

    # Query — real recall surface. Links are counted from the DB (the
    # recall results can't show them: the supersession filter removes the
    # stale fact before this loop sees the rows).
    links_formed = len(p._storage.unresolved_contradictions(UID, limit=100))
    correction_top1 = 0
    stale_top5 = 0
    for domain, stale, correction, query in PAIRS:
        results = p.recall(query, session_id="eval-session")
        rows = results.get("results", [])
        if not rows:
            print(f"  {domain:<8} ABSTAINED")
            continue
        top = rows[0].get("content")
        contents = [r.get("content") for r in rows]
        win = top == correction
        leak = stale in contents
        if win:
            correction_top1 += 1
        if leak:
            stale_top5 += 1
        print(f"  {domain:<8} correction_top1={win} stale_in_top5={leak}")

    n = len(PAIRS)
    result = {
        "n_pairs": n,
        "links_formed": links_formed,
        "correction_top1": round(correction_top1 / n, 4),
        "stale_present_top5": round(stale_top5 / n, 4),
    }
    print(f"\ncorrection_top1:  {result['correction_top1']}  (mechanism target)")
    print(f"stale in top-5:   {result['stale_present_top5']}")
    print(f"links formed:     {result['links_formed']}/{n}")
    print(f"EVAL_JSON={json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())