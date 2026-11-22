#!/usr/bin/env python3
"""Limbic calibration gap-band probe — floor-blind cosine bands (C9 lane).

The shipped harnesses measure the GATE (recall_eval abstain/surface,
decoy_eval false-surface). This measures the MARGIN the recall
confidence floor (0.50) and relevance floor (0.25) sit inside, on the same
fixture and shipped embedder, with NO gate in the path:

  empty margin    — max cosine any absent-topic (decoy) query reaches
                    against any stored fact (the floor's lower bound)
  answerable side — per-query max cosine among the query's expected facts
                    (best-expected), and whether expected is the raw top-1

Deterministic: fixed corpus, fixed queries, real ONNX embedder.
Usage: eval/gap_band_probe.py   (same runner as recall_eval)
Output: human table + one EVAL_JSON line for record/diffing.
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

UID = "eval-user"

# Same isolated-mode lane as recall_eval: agent root + repo parent, so
# `limbic` (repo dir is the package) and `eval` both import.
sys.path.insert(0, os.environ.get("HERMES_AGENT_ROOT")
                or os.path.expanduser("~/.hermes/hermes-agent"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Same 10x10 corpus / 30 paraphrase queries / 12 decoys as the gate
# harnesses — imported, never duplicated.
from eval.recall_eval import CORPUS, QUERIES
from eval.decoy_eval import DECOYS
from limbic.provider import LimbicMemoryProvider


def main() -> int:
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    home = os.path.join(base, "limbic-bandprobe-home")
    if os.path.exists(home):
        shutil.rmtree(home)
    os.makedirs(home)

    p = LimbicMemoryProvider()
    p.initialize("eval-session", hermes_home=home, user_id=UID, agent_identity="eval")
    p._get_user_id = lambda sid: UID

    contents = [c for items in CORPUS.values() for c in items]
    t0 = time.time()
    for c in contents:
        res = p.remember(c, user_id=UID)
        if res.get("error") or res.get("status") == "rejected_fragment":
            print(f"REJECTED/FAILED: {res.get('status')} {res.get('reason', '')}: {c}", file=sys.stderr)
            return 1
    print(f"ingested {len(contents)} facts, zero rejections ({time.time() - t0:.1f}s)")

    emb = p._embedder
    V = np.stack([emb.embed(c) for c in contents]).astype(float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    offsets, acc = {}, 0
    for d in CORPUS:
        offsets[d] = acc
        acc += len(CORPUS[d])

    t0 = time.time()
    qvec = {}
    for q in set(DECOYS) | {qry for _, qry, _ in QUERIES}:
        e = np.asarray(emb.embed(q), dtype=float)
        qvec[q] = e / np.linalg.norm(e)
    print(f"embedded {len(qvec)} queries ({time.time() - t0:.1f}s)")

    empty = {q: (V @ qvec[q]).max() for q in DECOYS}
    worst_empty = max(empty.values())

    ans = {}
    for domain, q, idx in QUERIES:
        allow = np.zeros(len(contents), dtype=bool)
        for i in idx:
            allow[offsets[domain] + i] = True
        best_expected = float((V[allow] @ qvec[q]).max())
        top_any = (V @ qvec[q]).max()
        ans[q] = {"best_expected_cosine": round(best_expected, 4),
                  "top_any_cosine": round(top_any, 4)}
    worst_answerable = min(v["best_expected_cosine"] for v in ans.values())

    result = {
        "n_facts": len(contents),
        "n_decoys": len(DECOYS),
        "n_queries": len(QUERIES),
        "empty_margin": {k: round(v, 4) for k, v in sorted(empty.items())},
        "empty_margin_max": round(worst_empty, 4),
        "answerable_min": round(worst_answerable, 4),
        "per_query": ans,
    }
    print("\n=== floor bands (floor-blind) ===")
    print(f"empty margin (max cosine across decoys):  {worst_empty}")
    for k, v in sorted(empty.items()):
        print(f"  {v:.4f}  {k}")
    print(f"answerable minimum (max within expected): {worst_answerable}")
    below = [q for _, q, _ in QUERIES if ans[q]["best_expected_cosine"] < 0.50]
    print(f"answerable queries with expected-below-0.50: {len(below)}")
    for q in below:
        print(f"  {ans[q]['best_expected_cosine']}  {q}")
    print(f"EVAL_JSON={json.dumps({k: result[k] for k in ('n_facts','n_decoys','n_queries','empty_margin_max','answerable_min','empty_margin','per_query')})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
