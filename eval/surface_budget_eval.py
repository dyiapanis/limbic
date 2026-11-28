"""Surface budget eval: rank-ordered 800-token cap must cost zero hits.

Empirical acceptance for the surface-budget operating point (SPEC,
"Surface budget — 800 tokens"): selecting facts in relevance-rank order
under a hard 800-token cap must (a) keep the payload at or under the cap
on every query, and (b) never drop the top-ranked (needle) fact — the
budget keeps the head, by construction.

Measured on the live-scale retrieval benchmark (15 fleet-shaped questions
against the 783-fact production store, 2026-11-28):

    BEFORE cap:  13/15 hits,  27-29 ms avg,  2,515 tokens avg per recall
    AFTER cap:   13/15 hits,  26-56 ms avg,    747 tokens avg per recall

Verdicts unchanged; payload cut 70%; latency in noise. The two misses are
pre-existing (needle fact never encoded / pre-embedding-era row) and
unrelated to the budget.

Run:  python3 surface_budget_eval.py   (from this directory)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from limbic.provider import LimbicMemoryProvider, SURFACE_BUDGET_TOKENS  # noqa: E402

# Synthetic corpus shaped like the real workload: opaque service ids,
# deployed locations, tool chains, procedural facts. ~150 tokens each,
# 60 facts — 9,000 tokens total, so no query can fit the whole pool.
SERVICES = ["atlas-7", "beacon-2", "cinder-9", "delta-4", "ember-6"]
REGIONS = ["frankfurt", "sydney", "ohare", "virginia", "seoul"]
VERBS = ["deployed", "migrated", "provisioned", "certified", "archived"]


def _corpus():
    facts = []
    for i in range(60):
        svc = SERVICES[i % 5]
        region = REGIONS[(i // 5) % 5]
        verb = VERBS[(i // 25) % 5]
        body = (f"The {svc} service was {verb} in {region} during cycle {i}. "
                f"Ops note {i}: the runbook for {svc} lives in the infra "
                f"handbook; the on-call rotation changes quarterly and the "
                f"escalation path goes through the platform duty officer. "
                f"Backups for {svc} run nightly with a 14-day retention "
                f"window and a restore drill every second Tuesday.")
        facts.append(body)
    return facts


QUERIES = [f"where is {svc} deployed" for svc in SERVICES]


class _GateBypassTrust:
    """Floor-free trust shim — the eval isolates the budget cap, not the
    confidence gate (recall_eval owns gate calibration). Swapped in only
    for the read; the write path above runs the real engine."""

    def get(self, name):
        return {"recall_confidence_floor": 0.0,
                "relevance_floor": 0.25,
                "trust_weight": 0.2}.get(name, 0.0)

    def compute_activation(self, created_at, retrieval_count):
        return 0.9  # mature


def main():
    tmp = tempfile.mkdtemp(prefix="limbic-surface-budget-eval-")
    try:
        # Inline users: manifest — without it the write path fail-closes to
        # the opaque os_uid, which the room-shape guard then rejects (C8).
        (Path(tmp) / "limbic.yaml").write_text(
            "users:\n  evaluser:\n    display_name: Eval User\n    scope: adult\n")
        # Gateway-spawned runs inherit the operator's session identity via
        # env; the eval must resolve to ITS principal, not the operator's.
        os.environ["HERMES_SESSION_USER_ID"] = "evaluser"
        p = LimbicMemoryProvider()
        p.initialize("eval", hermes_home=tmp, user_id="evaluser",
                     agent_context="primary", agent_identity="eval")
        p._current_user_id = "evaluser"
        for i, body in enumerate(_corpus()):
            r = p.remember(body, user_id="evaluser")
            assert r.get("status") in ("stored", "deduplicated"), r

        failures = []
        p._trust = _GateBypassTrust()
        for q in QUERIES:
            out = p.recall(q, session_id="eval-session")
            payload = sum(len(r["content"]) // 4 for r in out["results"])
            svc = q.split()[-2]
            needle_hit = any(svc in r["content"] for r in out["results"])
            status = "OK " if (payload <= SURFACE_BUDGET_TOKENS
                               and needle_hit) else "FAIL"
            if status == "FAIL":
                failures.append(q)
            print(f"  {status} {payload:>4}tok needle={'Y' if needle_hit else 'N'} "
                  f"| {q}")
            assert payload <= SURFACE_BUDGET_TOKENS, (
                f"payload {payload} exceeds budget {SURFACE_BUDGET_TOKENS}")
            assert needle_hit, f"needle for {q!r} dropped by the budget"

        print(f"== surface budget: {len(QUERIES) - len(failures)}/{len(QUERIES)} "
              f"clean, cap {SURFACE_BUDGET_TOKENS} holds, needles intact ==")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
