"""Echo fallback tests (SPEC: echo fallback).

The gate stays silent; the archive echoes — one raw excerpt, labelled,
scope-inherited through the manifest, fail-closed on every unresolved
session class. Scope tests run against a mixed-sender fixture, never the
live store (review amendment).

Run:
    cd <repo-checkout> && PYTHONPATH= python -m pytest tests/test_echo_fallback.py -v
"""
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.identity import UserManifest  # noqa: E402
from limbic.provider import (  # noqa: E402
    _ECHO_LABEL,
    LimbicMemoryProvider,
    _redact_secrets,
    _strip_scaffold,
)
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8

MM_DEMETRI = "d" * 26   # canonical 26-char Mattermost id shape
MM_PEER = "k" * 26
NEEDLE = "peer hit the re.findall greedy-match trap while parsing fixtures"

MANIFEST_USERS = {
    "demetri": {"display_name": "Demetri", "scope": "adult",
                "platform_ids": {"mattermost": MM_DEMETRI}, "guardians": []},
    "peer": {"display_name": "Peer", "scope": "child",
              "platform_ids": {"mattermost": MM_PEER}, "guardians": []},
}


class FakeEmbedder:
    """First-8-chars bag embedding — distinct texts keep distinct cosines."""

    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def _core_schema(db):
    """Minimal same-shape replica of the Hermes core tables echo reads."""
    db.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, "
               "source TEXT NOT NULL, "
               "user_id TEXT, started_at REAL NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
               "session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, "
               "timestamp REAL NOT NULL DEFAULT 0, "
               "active INTEGER NOT NULL DEFAULT 1, "
               "_compressed_summary INTEGER NOT NULL DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS messages_fts_src (content, tool_name, tool_calls)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, tool_name, "
               "tool_calls, content='messages_fts_src', content_rowid='id')")


def _session(db, sid, source, user_id):
    db.execute("INSERT INTO sessions (id, source, user_id) VALUES (?,?,?)",
               (sid, source, user_id))


def _msg(db, sid, role, content, compressed=0):
    cur = db.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, "
        "_compressed_summary) VALUES (?,?,?,?,?)",
        (sid, role, content, 1795848796.0, compressed))
    db.execute("INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
               "VALUES (?,?,'','')", (cur.lastrowid, content))


def _core(tmp_path):
    """Open the sibling core archive and ensure the schema echo reads."""
    db = sqlite3.connect(str(tmp_path / "state.db"))
    _core_schema(db)
    return db


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


def _seed(tmp_path):
    """Store + mixed-sender core rows; returns (provider, db)."""
    p = _provider(tmp_path)
    db = _core(tmp_path)
    # questioner's own record: a scaffolded compaction summary holding the needle
    _session(db, "sess-demetri", "mattermost", MM_DEMETRI)
    _msg(db, "sess-demetri", "user",
         "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted.\n"
         "---\n" + NEEDLE)
    _msg(db, "sess-demetri", "user", "demetri asked about sqlite WAL tuning")
    # another user's record — echo must never cross
    _session(db, "sess-peer", "mattermost", MM_PEER)
    _msg(db, "sess-peer", "user", "peer's other private note about re.findall")
    # unattributed session — fail-closed
    _session(db, "sess-orphan", "mattermost", "")
    _msg(db, "sess-orphan", "user", "orphan row mentioning re.findall nobody owns")
    # excluded agent class
    _session(db, "sess-kanban", "kanban", "")
    _msg(db, "sess-kanban", "assistant", "kanban-internal re.findall chatter")
    # host frame — junk even in an owned session
    _msg(db, "sess-demetri", "assistant",
         "[System: Your previous response was truncated] re.findall")
    db.commit()
    return p, db


def test_scope_and_scaffold_strip(tmp_path):
    p, _ = _seed(tmp_path)
    echo = p._echo_fallback("re.findall", "demetri", "sess-demetri")
    assert echo is not None, "the questioner's own scaffolded needle must echo"
    assert NEEDLE in echo["excerpt"], "content beneath the scaffold is kept"
    assert "[CONTEXT COMPACTION" not in echo["excerpt"], "markers stripped"
    assert echo["scaffold_stripped"] is True
    assert "other private note" not in echo["excerpt"], "peer unreachable"
    assert echo["label"] == _ECHO_LABEL


def test_cross_user_unreachable(tmp_path):
    p, _ = _seed(tmp_path)
    echo = p._echo_fallback("re.findall", "peer", "sess-peer")
    assert echo is not None
    assert NEEDLE not in echo["excerpt"], "demetri's rows stay out of peer's echo"
    assert "other private note" in echo["excerpt"]


def test_unattributed_fail_closed(tmp_path):
    p = _provider(tmp_path)
    db = _core(tmp_path)
    _session(db, "sess-orphan", "mattermost", "")
    _msg(db, "sess-orphan", "user", "orphan re.findall row nobody owns")
    _session(db, "sess-kanban", "kanban", MM_DEMETRI)  # agent class, even owned
    _msg(db, "sess-kanban", "user", "kanban re.findall row")
    db.commit()
    assert p._echo_fallback("re.findall", "demetri", "sess-x") is None


def test_host_frame_excluded(tmp_path):
    p = _provider(tmp_path)
    db = _core(tmp_path)
    _session(db, "sess-demetri", "mattermost", MM_DEMETRI)
    _msg(db, "sess-demetri", "assistant",
         "[System: Your previous response was truncated] re.findall")
    db.commit()
    assert p._echo_fallback("re.findall", "demetri", "sess-demetri") is None


def test_feature_detect_degrades(tmp_path):
    p = _provider(tmp_path)
    db = _core(tmp_path)
    _session(db, "s", "mattermost", MM_DEMETRI)
    _msg(db, "s", "user", "a note")
    db.execute("DROP TABLE messages_fts")
    db.execute("DROP TABLE messages")
    db.commit()
    assert p._echo_fallback("anything", "demetri", "s") is None
    out = p.recall("anything", session_id="s")
    assert out["status"] == "no_confident_match"
    assert "echo" not in out, "abstention looks exactly like yesterday"


def test_prefetch_byte_identical(tmp_path):
    p, _ = _seed(tmp_path)
    with_archive = p.prefetch("re.findall")
    assert _ECHO_LABEL not in (with_archive or ""), "prefetch never echoes"
    db = _core(tmp_path)
    db.execute("DROP TABLE messages_fts")
    db.execute("DROP TABLE messages")
    db.commit()
    assert p.prefetch("re.findall") == with_archive, "prefetch never reads archive"


def test_excerpt_counts_against_budget(tmp_path):
    p = _provider(tmp_path)
    db = _core(tmp_path)
    _session(db, "s", "mattermost", MM_DEMETRI)
    _msg(db, "s", "user", "wal tuning note " + "x" * 5000)
    db.commit()
    from limbic.provider import _estimate_tokens
    echo = p._echo_fallback("wal tuning", "demetri", "s")
    assert echo is not None
    assert _estimate_tokens(echo["excerpt"]) <= 800
    assert echo["excerpt"].endswith("…")


def test_recall_abstention_carries_echo(tmp_path):
    p, _ = _seed(tmp_path)
    p._get_user_id = lambda sid: "demetri"
    out = p.recall("re.findall", session_id="sess-demetri")
    assert out["status"] == "no_confident_match"
    assert out.get("echo", {}).get("label") == _ECHO_LABEL
    assert NEEDLE in out["echo"]["excerpt"]


def test_near_miss_abstention_carries_echo(tmp_path):
    p, _ = _seed(tmp_path)
    p._get_user_id = lambda sid: "demetri"
    emb = [1.0] + [0.0] * (DIMS - 1)
    p._embedder = type("E", (), {
        "embed": staticmethod(lambda q, is_query=False: emb)})()
    low = [{"fact_id": 1, "content": "mediocre", "relevance": 0.29,
            "composite_score": 0.10, "vector_score": 0.29}]
    p._storage.search = staticmethod(lambda *a, **kw: low)
    p._storage.search_fts = staticmethod(lambda *a, **kw: [])
    p._storage.search_graph = staticmethod(lambda *a, **kw: [])
    p._storage.get_embedding = staticmethod(lambda *a, **kw: None)
    out = p.recall("re.findall", session_id="sess-demetri")
    assert out["status"] == "no_confident_match"
    assert NEEDLE in out.get("echo", {}).get("excerpt", "")


def test_echo_served_telemetry(tmp_path):
    p = _provider(tmp_path)
    db = _core(tmp_path)
    _session(db, "s", "mattermost", MM_DEMETRI)
    _msg(db, "s", "user", "a wal tuning note")
    db.commit()
    before = p._storage.get_metrics().get("echo_served", 0)
    p._echo_fallback("wal tuning", "demetri", "s")
    after = p._storage.get_metrics().get("echo_served", 0)
    assert after == before + 1


def test_scaffold_strip_unit():
    # scaffolding lines die; every prose line survives (including wrapper
    # bodies and compaction summary prose — that prose IS the record)
    row = ("[PRIOR CONTEXT — for reference only; not a new message]\n"
           "leave it\n\n"
           "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]\n\n"
           "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
           "compacted into the summary below.\n"
           "instruction prose\n"
           "more prose\n"
           "--- END OF CONTEXT SUMMARY — respond to the message below, "
           "not the summary above ---\nREAL MESSAGE")
    out = _strip_scaffold(row)
    assert "leave it" in out and "REAL MESSAGE" in out
    assert "instruction prose" in out
    assert "CONTEXT COMPACTION" not in out
    assert "Earlier turns were compacted" not in out
    assert "PRIOR CONTEXT" not in out
    assert "END OF CONTEXT SUMMARY" not in out
    # nested markers strip line-by-line
    row2 = ("[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
            "compacted into the summary below.\n"
            "SUMMARY ONE\n"
            "--- END OF CONTEXT SUMMARY — respond to the message below ---\n"
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were "
            "compacted into the summary below.\n"
            "SUMMARY TWO\n"
            "--- END OF CONTEXT SUMMARY — respond to the message below ---\n"
            "REAL MESSAGE")
    out2 = _strip_scaffold(row2)
    assert "SUMMARY ONE" in out2 and "SUMMARY TWO" in out2
    assert "REAL MESSAGE" in out2
    assert "CONTEXT COMPACTION" not in out2
    assert out2.count("---") == 0


def test_redaction_floor():
    assert _redact_secrets("key sk-abcdef0123456789abcdef end") == "key «redacted» end"
    out = _redact_secrets("hdr Authorization: Bearer abcdef0123456789abcd trailing")
    assert "abcdef0123456789abcd" not in out and "«redacted»" in out
    assert _redact_secrets("no secrets here") == "no secrets here"
