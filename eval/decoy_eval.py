#!/usr/bin/env python3
"""Limbic gate decoy eval — the false-positive observable for the recall
confidence floors (#19).

Both harness corpora only ask queries that HAVE a relevant fact, so nothing
measured the gate's abstain side. This harness ingests the base corpus and
asks decoy queries about topics ABSENT from it: the gate should abstain. The
observable for floor calibration:

  abstain_rate    — correct behaviour (nothing relevant exists)
  false_surface_* — when the gate does NOT abstain, how close the top
                    lookalike got (its relevance vs the floors)

Calibration trigger (watched, not campaigned): re-tune the floors when
false-surface rate crosses ~10% at a corpus milestone. The pressure variable
is facts-per-topic density, not total count.

Usage: eval/decoy_eval.py [--home DIR]   (same runner as recall_eval)
"""
import json
import os
import sys
import tempfile
import time

UID = "eval-user"
K = 5

# Topics absent from CORPUS (family/work/health/travel/finance/home/tech/
# social/food/hobby). Shared-vocabulary near-misses are deliberate — that
# is exactly the lookalike pressure being measured.
DECOYS = [
    "what is her favourite font?",
    "which ski wax works in cold snow?",
    "when does the Mars rover team meet?",
    "what breed is the neighbour's dog?",
    "how much was the guitar repair quote?",
    "where are the opera tickets?",
    "what time does the ferry leave on Sundays?",
    "who services the espresso machine?",
    "what is the licence plate of the van?",
    "how deep is the koi pond?",
    "which martial art does the dojo teach on Thursdays?",
    "what model is the garage door opener?",
]


def _arg(name: str, default: str) -> str:
    argv = sys.argv[1:]
    if name in argv:
        i = argv.index(name)
        return argv[i + 1] if i + 1 < len(argv) else default
    return default


def main() -> int:
    home = _arg("--home", "") or tempfile.mkdtemp(prefix="limbic-decoy-")
    if not os.path.isdir(home):
        os.makedirs(home, exist_ok=True)
    print(f"eval home: {home}")

    # Same isolated-mode lane as recall_eval; repo parent makes `limbic`
    # and `eval` importable.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.environ.get("HERMES_AGENT_ROOT")
                    or os.path.expanduser("~/.hermes/hermes-agent"))
    sys.path.insert(0, os.path.dirname(repo_root))
    sys.path.insert(0, repo_root)

    from eval.recall_eval import CORPUS  # reuse the base corpus — no duplication
    from limbic.provider import LimbicMemoryProvider

    t0 = time.time()
    p = LimbicMemoryProvider()
    p.initialize("eval-session", hermes_home=home, user_id=UID, agent_identity="eval")
    p._get_user_id = lambda sid: UID
    print(f"provider initialised ({time.time() - t0:.1f}s)")

    t0 = time.time()
    n = 0
    for domain, items in CORPUS.items():
        for content in items:
            res = p.remember(content, user_id=UID)
            if res.get("error"):
                print(f"STORE FAILED: {res['error']}: {content}", file=sys.stderr)
                return 1
            n += 1
    print(f"ingested {n} facts ({time.time() - t0:.1f}s)")

    floor = p._trust.get("relevance_floor")
    conf_floor = p._trust.get("recall_confidence_floor")
    print(f"floors: relevance={floor} confidence={conf_floor}")

    abstained = 0
    false_tops: list[float] = []
    for q in DECOYS:
        out = p.recall(q, session_id="eval-session")
        if out.get("status") == "no_confident_match":
            abstained += 1
            print(f"  ABSTAIN  {q}")
        else:
            # recall() strips scoring fields for the agent surface — the
            # lookalike cosine lives on the raw _search rows.
            raw = p._search(q, UID, limit=K)
            top = float(raw[0].get("relevance", 0.0) or 0.0) if raw else 0.0
            false_tops.append(top)
            print(f"  SURFACED {q}  top_relevance={top:.4f}  ({raw[0].get('content', '')[:50]})")

    total = len(DECOYS)
    result = {
        "n_facts": n,
        "n_decoys": total,
        "abstain_rate": round(abstained / total, 4),
        "false_surface_rate": round(len(false_tops) / total, 4),
        "false_top_max": round(max(false_tops), 4) if false_tops else None,
        "false_top_mean": round(sum(false_tops) / len(false_tops), 4) if false_tops else None,
        "relevance_floor": floor,
        "recall_confidence_floor": conf_floor,
    }
    print(f"\nabstain rate:         {result['abstain_rate']}  (correct behaviour)")
    print(f"false-surface rate:    {result['false_surface_rate']}  (calibrate floors if > ~0.10)")
    print(f"false-surface top cos: max={result['false_top_max']} mean={result['false_top_mean']}")
    print(f"EVAL_JSON={json.dumps(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())