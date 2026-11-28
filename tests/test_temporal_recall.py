"""Temporal recall tests.

A time-shaped query (explicit patterns only) re-ranks the already-gated
pool by date proximity; non-temporal queries keep the composite ranking
byte-for-byte, and the relevance floor is never crossed by the boost.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.identity import UserManifest  # noqa: E402
from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8

MM_OWNER = "d" * 26

MANIFEST_USERS = {
    "owner": {"display_name": "Owner", "scope": "adult",
              "platform_ids": {"mattermost": MM_OWNER}, "guardians": []},
}

NEUTRAL_Q = "audit report status check"
ANCHORED_Q = "audit report schedule for November 2026"
DISTRACTOR = "audit report covers the networking stack"   # old, high trust
NEEDLE = "audit report covers the dns failover setup"  # fresh, low trust


class FakeEmbedder:
    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = (sum(x * x for x in v)) ** 0.5 or 1.0
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


def _seed(p, tmp_path):
    r = p.remember(DISTRACTOR, user_id="owner")
    assert r.get("fact_id"), r
    r2 = p.remember(NEEDLE, user_id="owner")
    assert r2.get("fact_id"), r2
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    con.execute("UPDATE facts SET trust_score = 0.9, "
                "created_at = '2026-05-01T09:00:00+00:00', "
                "updated_at = '2026-05-01T09:00:00+00:00' "
                "WHERE fact_id = ?", (r["fact_id"],))
    con.execute("UPDATE facts SET trust_score = 0.5, "
                "created_at = '2026-11-20T09:00:00+00:00', "
                "updated_at = '2026-11-20T09:00:00+00:00' "
                "WHERE fact_id = ?", (r2["fact_id"],))
    con.commit()
    con.close()
    return r["fact_id"], r2["fact_id"]


def test_detector_patterns(tmp_path):
    p = _provider(tmp_path)
    assert p._temporal_intent("when did we ship 0.9.6") == ("recency", None)
    assert p._temporal_intent("how long ago did the audit run") == ("recency", None)
    assert p._temporal_intent("plans for 2026-11-05")[0] == "anchor"
    assert p._temporal_intent("what changed in November 2026")[0] == "anchor"
    assert p._temporal_intent("review the Nov 2026 notes")[0] == "anchor"
    assert p._temporal_intent("plans for 2026-11")[0] == "anchor"
    assert p._temporal_intent("what did we decide about the budget") is None
    assert p._temporal_intent("whenever you like") is None
    assert p._temporal_intent("where is the report") is None
    assert p._temporal_intent("plans for 2026-13-01") is None  # invalid date
    assert p._temporal_intent("") is None


def test_temporal_promotion(tmp_path):
    p = _provider(tmp_path)
    distractor_id, needle_id = _seed(p, tmp_path)

    # Temporal query: the fresh low-trust fact overtakes the stale high-trust one
    gated = p._search(ANCHORED_Q, "owner")
    order = [f["fact_id"] for f in gated]
    assert order[0] == needle_id, f"needle must be promoted: {order}"
    assert order[1] == distractor_id

    # Non-temporal query: composite order stands — distractor first
    gated_n = p._search(NEUTRAL_Q, "owner")
    order_n = [f["fact_id"] for f in gated_n]
    assert order_n[0] == distractor_id, f"no temporal term on neutral: {order_n}"
    assert "temporal_score" not in gated_n[0]
    assert "filed" not in gated_n[0]


def test_nontemporal_ranking_byte_identical(tmp_path):
    p = _provider(tmp_path)
    _seed(p, tmp_path)
    saved = p._temporal_intent
    p._temporal_intent = lambda q: None
    try:
        disabled = [(f["fact_id"], f["composite_score"])
                    for f in p._search(ANCHORED_Q, "owner")]
        disabled_neutral = [(f["fact_id"], f["composite_score"])
                            for f in p._search(NEUTRAL_Q, "owner")]
    finally:
        p._temporal_intent = saved
    enabled = [(f["fact_id"], f["composite_score"])
               for f in p._search(NEUTRAL_Q, "owner")]
    assert disabled_neutral == enabled, "neutral ranking changed by the temporal path"


def test_render_filed_only_on_temporal(tmp_path):
    p = _provider(tmp_path)
    _seed(p, tmp_path)
    p._session_user_id["sess-owner"] = "owner"
    resp = p.recall(ANCHORED_Q, session_id="sess-owner")
    assert resp["status"] == "ok", resp
    first = resp["results"][0]
    assert "filed" in first, "temporal query renders the fact timestamp"
    assert first["filed"] == "filed November 20"
    neutral = p.recall(NEUTRAL_Q, session_id="sess-owner")
    assert all("filed" not in r for r in neutral["results"]), \
        "non-temporal query must not render timestamps"


def test_boost_never_crosses_floor(tmp_path):
    p = _provider(tmp_path)
    real_embed = p._embedder.embed
    def poisoned(content, is_query=False):
        v = real_embed(content, is_query)
        if not is_query and content.startswith("zzz"):
            return [-x for x in v[:4]] + v[4:]  # near-orthogonal to queries
        return v
    p._embedder.embed = poisoned
    r = p.remember("zzz quixotic calibrate meridian tones hover", user_id="owner")
    assert r.get("fact_id")
    con = sqlite3.connect(str(tmp_path / "limbic.db"))
    con.execute("UPDATE facts SET created_at = '2026-11-20T09:00:00+00:00' "
                "WHERE fact_id = ?", (r["fact_id"],))
    con.commit()
    con.close()
    p._session_user_id["sess-owner"] = "owner"
    resp = p.recall(ANCHORED_Q, session_id="sess-owner")
    assert resp["status"] == "no_confident_match", \
        "temporal intent must not admit a below-floor fact"
    assert resp["results"] == []
