"""Sprint C1 — `limbic quality` instrument regression tests.

The metric SQL lives verbatim in cli.py (pinned definitions). These tests
build a tiny store with the production column shape and assert each pinned
definition: trust buckets + zombie band, never-recalled/stale-30d window,
contradiction density, orphan entities, length buckets, room-id user shape,
and the VACUUM INTO backup roundtrip.
"""
import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SCHEMA = """
CREATE TABLE facts (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL, user_id TEXT NOT NULL,
    trust_score REAL DEFAULT 0.5, retrieval_count INTEGER DEFAULT 0, last_retrieved TEXT,
    matured_at TEXT, atrophy_marked_at TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE entities (entity_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT DEFAULT 'UNKNOWN');
CREATE TABLE mentions (
    fact_id INTEGER NOT NULL, entity_id TEXT NOT NULL,
    PRIMARY KEY (fact_id, entity_id),
    FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id));
CREATE TABLE contradictions (
    newer_fact_id INTEGER NOT NULL, older_fact_id INTEGER NOT NULL,
    created_at TEXT NOT NULL, nli_label TEXT NOT NULL, resolved_at TEXT,
    PRIMARY KEY (newer_fact_id, older_fact_id));
"""

ROOM = "!abc:matrix.org"


def _iso(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _load_cli():
    """Import cli.py without triggering limbic/__init__ (which pulls provider.py
    → agent.*, unavailable outside Hermes) — same pattern as test_model_pins."""
    if "limbic_cli" in sys.modules:
        return sys.modules["limbic_cli"]
    spec = importlib.util.spec_from_file_location("limbic_cli", REPO / "cli.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["limbic_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


def _store(tmp_path: Path) -> Path:
    """Five facts: len 35/<40, 100+500/40-599, 700/600-1099, 1200/>=1100;
    trust one per bucket: 0.01 zombie, 0.15, 0.25, 0.55, 0.90."""
    db = tmp_path / "limbic.db"
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    rows = [
        # content, user, trust, rc, last_retrieved, matured, created
        ("x" * 35, "u1", 0.01, 0, None, None, _iso(100)),
        ("y" * 100, "u1", 0.15, 0, _iso(1), None, _iso(90)),
        ("z" * 700, "u1", 0.25, 3, _iso(40), None, _iso(80)),
        ("w" * 1200, ROOM, 0.55, 10, _iso(1), "2026-01-01 00:00:00", _iso(70)),
        ("v" * 500, "u2", 0.90, 1, _iso(1), "2026-01-02 00:00:00", _iso(60)),
    ]
    con.executemany(
        "INSERT INTO facts (content, user_id, trust_score, retrieval_count, last_retrieved,"
        " matured_at, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        [(*r, r[6]) for r in rows],
    )
    con.execute("INSERT INTO entities (entity_id, name) VALUES ('e1', 'Mentioned')")
    con.execute("INSERT INTO entities (entity_id, name) VALUES ('e2', 'Tombstone')")
    con.execute("INSERT INTO mentions (fact_id, entity_id) VALUES (3, 'e1')")
    # one unresolved link (2←3), one resolved (3←4)
    con.execute("INSERT INTO contradictions VALUES (3, 2, ?, 'contradiction', NULL)", (_iso(5),))
    con.execute("INSERT INTO contradictions VALUES (4, 3, ?, 'contradiction', ?)", (_iso(2), _iso(1)))
    con.commit()
    con.close()
    return db


def test_quality_metric_definitions(tmp_path):
    cli = _load_cli()
    r = cli.quality_report(_store(tmp_path))
    assert r["facts_total"] == 5 and r["users_total"] == 3
    t = r["trust"]
    assert t["buckets"] == {"<0.05": 1, "0.05-0.20": 1, "0.20-0.40": 1, "0.40-0.60": 1,
                            "0.60-0.80": 0, ">=0.80": 1, "none": 0}
    assert t["lt_0_02"] == 1 and t["zombie_lt_0_05_share"] == 0.2
    nr = r["never_recalled"]
    assert nr["count"] == 2 and nr["stale_30d"] == 2  # NULL + 40-day-old anchor
    c = r["contradictions"]
    assert c["total"] == 2 and c["unresolved"] == 1 and c["facts_involved"] == 3
    assert c["density_per_fact"] == 0.4 and c["events_per_involved_fact"] == 0.667
    e = r["entities"]
    assert e == {"total": 2, "orphans": 1, "orphan_ratio": 0.5, "mentions_rows": 1}
    assert r["length"]["buckets"] == {"<40": 1, "40-599": 2, "600-1099": 1, ">=1100": 1}
    assert r["room_shaped"] == {"user_ids": [ROOM], "facts": 1}
    assert r["liveness"] == {"matured": 2, "atrophy_marked": 0}


def test_quality_report_is_read_only(tmp_path):
    cli = _load_cli()
    db = _store(tmp_path)
    before = db.read_bytes()
    cli.quality_report(db)
    assert db.read_bytes() == before and set((db.parent).glob("*-wal*")) == set()


def test_backup_store_roundtrip(tmp_path):
    cli = _load_cli()
    db = _store(tmp_path)
    out = cli.backup_store(db, tmp_path / "backups", "tprof")
    b = Path(out["path"])
    assert out["integrity"] == "ok" and out["facts"] == 5 and b.exists()
    assert b.name.startswith("tprof-limbic-") and b.suffix == ".db"
    # timestamped copies never overwrite each other
    import time
    time.sleep(1.1)
    out2 = cli.backup_store(db, tmp_path / "backups", "tprof")
    assert out2["path"] != out["path"] and Path(out2["path"]).exists()


def test_cli_quality_end_to_end(tmp_path, capsys, monkeypatch):
    cli = _load_cli()
    home = tmp_path / "herb"
    (home / "logs").mkdir(parents=True)
    db = _store(home)
    out = tmp_path / "herb.json"
    cli.main(["--hermes-home", str(home), "quality",
              "--backup-dir", str(tmp_path / "bk"), "-o", str(out)])
    parsed = json.loads(out.read_text())
    assert parsed["profile"] == "herb" and parsed["facts_total"] == 5
    assert parsed["backup"]["path"].startswith(str(tmp_path / "bk") + "/herb-limbic-")
    # store path is derived from --hermes-home when --db is absent
    assert parsed["store"] == str(db)
    captured = capsys.readouterr().out
    assert "herb.json" in captured
    # missing store exits 1
    missing = tmp_path / "empty-home"
    missing.mkdir()
    try:
        cli.main(["--hermes-home", str(missing), "quality"])
        raise AssertionError("missing store must exit non-zero")
    except SystemExit as e:
        assert e.code == 1
