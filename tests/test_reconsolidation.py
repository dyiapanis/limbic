"""Reconsolidation write-back tests.

An echo-served excerpt may be re-encoded as a real memory through the normal
write path — caller-distilled (plugin stays no-LLM), provenance-stamped,
fail-closed without a pending echo, redacted at the boundary, and subject to
the SAME gates and atrophy as any other fact.

Run:
    cd <repo-checkout> && PYTHONPATH= python -m pytest tests/test_reconsolidation.py -v
"""
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.identity import UserManifest  # noqa: E402
from limbic.provider import (  # noqa: E402
    LimbicMemoryProvider,
    _redact_secrets,
)
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8

MM_OWNER = "d" * 26

MANIFEST_USERS = {
    "owner": {"display_name": "Owner", "scope": "adult",
              "platform_ids": {"mattermost": MM_OWNER}, "guardians": []},
}

DISTILLED = "owner tunes sqlite WAL checkpoints on weekends"


class FakeEmbedder:
    """First-8-chars bag embedding — distinct texts keep distinct cosines."""

    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def _provider(tmp_path):
    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = type("N", (), {
        "classify": staticmethod(lambda a, b: ("neutral", 0.0))})()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    p._manifest = UserManifest("fixture", MANIFEST_USERS)
    p._session_guardian_of = {}
    p._initialised = True
    return p


def _serve_echo(p):
    """Simulate the read side: an echo excerpt was served to this user."""
    p._pending_echo["owner"] = True


def test_archive_write_back_admitted_with_provenance(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    r = p.remember(DISTILLED, user_id="owner", source="archive")
    assert r.get("fact_id"), f"write-back must be admitted: {r}"
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    row = con.execute(
        "SELECT source, trust_score, split_parent_id FROM facts "
        "WHERE fact_id = ?", (r["fact_id"],)).fetchone()
    con.close()
    assert row[0] == "archive", "provenance column stamped"
    assert row[1] == p._ARCHIVE_INITIAL_TRUST, "reduced initial trust, no surprise boost"
    assert row[2] is None, "atomic fact — never split"


def test_archive_write_requires_pending_echo(tmp_path):
    p = _provider(tmp_path)
    before = p._storage.count("owner")
    r = p.remember(DISTILLED, user_id="owner", source="archive")
    assert r.get("status") == "rejected_archive", "fail closed without an echo"
    assert p._storage.count("owner") == before, "nothing stored"
    bad = p.remember(DISTILLED, user_id="owner", source="forged")
    assert bad.get("status") == "rejected_source", "unknown provenance rejected"


def test_junk_excerpt_rejected_by_same_gate(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    r = p.remember("yes ok", user_id="owner", source="archive")
    assert r.get("status") == "rejected_fragment", "same admission bar as any memory"


def test_redaction_at_write_boundary(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    leaked = f"owner's api key is sk-abcdef1234567890abcdef and rotates monthly"
    r = p.remember(leaked, user_id="owner", source="archive")
    assert r.get("fact_id"), f"admitted: {r}"
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    stored = con.execute(
        "SELECT content FROM facts WHERE fact_id = ?", (r["fact_id"],)).fetchone()[0]
    con.close()
    assert "sk-abcdef1234567890" not in stored, "secret shape never reaches the store"
    assert "«redacted»" in stored


def test_archive_fact_skips_splitter(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    blob = ("owner reported build timings\n"
            "## Morning runs\n"
            "first block of the morning report with several details listed\n"
            "## Evening runs\n"
            "second block of the evening report with more details listed here")
    r = p.remember(blob, user_id="owner", source="archive")
    assert r.get("fact_id"), "splittable archive content stays one atomic fact"
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    n = con.execute("SELECT count(*) FROM facts").fetchone()[0]
    con.close()
    assert n == 1, "no parent/children fan-out on the archive lane"


def test_atrophy_applies_identically(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    arch = p.remember(DISTILLED, user_id="owner", source="archive")
    agent = p.remember("peer prefers plain-text commit messages over patches",
                        user_id="owner")
    assert arch.get("fact_id") and agent.get("fact_id")
    stale = "2026-06-01T00:00:00.000000Z"
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    con.execute("UPDATE facts SET trust_score = 0.005, last_retrieved = ?, "
                "updated_at = ? WHERE fact_id IN (?, ?)",
                (stale, stale, arch["fact_id"], agent["fact_id"]))
    con.commit()
    con.close()
    p._apply_temporal_decay("owner")
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    marks = dict(con.execute(
        "SELECT fact_id, atrophy_marked_at FROM facts WHERE fact_id IN (?, ?)",
        (arch["fact_id"], agent["fact_id"])).fetchall())
    con.close()
    assert marks[arch["fact_id"]] is not None, "archive fact atrophies"
    assert marks[agent["fact_id"]] is not None, "same rule as agent facts"


def test_repeat_question_hits_store_not_echo(tmp_path):
    p = _provider(tmp_path)
    _serve_echo(p)
    r = p.remember(DISTILLED, user_id="owner", source="archive")
    assert r.get("fact_id")
    p._session_user_id["sess-owner"] = "owner"
    resp = p.recall("sqlite WAL tuning", session_id="sess-owner")
    assert resp["results"], "re-asked question is answered from the store"
    assert "echo" not in resp, "echo tax paid once — no second serve"
