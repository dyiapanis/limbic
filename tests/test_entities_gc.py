"""gc_entities — entity-graph hygiene sweep (junk + orphan classes).

The sweep mirrors the extraction junk filter over already-stored rows and
purges orphaned mention/entity rows. Every class is asserted both against the
sweep's own live measurement AND an independent SQL re-measure of the same
store — the same gate the deploy-verification step asserts at fleet scale.
"""
import json
import math
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.entities import junk_matcher  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402

DIMS = 8


def _vec(i: int) -> list[float]:
    v = [float(i + j % 7 + 1) for j in range(DIMS)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    yield s
    s.close()


ORPHAN_ENTITIES_SQL = (
    "SELECT COUNT(*) FROM entities "
    "WHERE entity_id NOT IN (SELECT DISTINCT entity_id FROM mentions)"
)


def _legacy_shape(store):
    """Two live facts plus a mentionless-orphaned one, with junk entities
    attached as if written before the extraction filter existed."""
    a = store.store("Grace ran the 2026 race", user_id="grace", embedding=_vec(1),
                    metadata={"trust_score": 0.5})
    b = store.store("Grace wrote the report again", user_id="grace", embedding=_vec(2),
                    metadata={"trust_score": 0.5})
    c = store.store("Wombats sleep in burrows", user_id="grace", embedding=_vec(3),
                    metadata={"trust_score": 0.5})
    store.store_entities(a, [{"name": "Grace", "type": "PERSON"},
                             {"name": "then", "type": "UNKNOWN"}])
    store.store_entities(b, [{"name": "One", "type": "CARDINAL"},
                             {"name": "10", "type": "CARDINAL"}])
    store.store_entities(c, [{"name": "Wombats", "type": "PERSON"}])
    # a fact deletion with FK cascade removes the mention row — the entity
    # survives mentionless, exactly the shape the sweep's orphan class targets
    store.delete_fact(c)
    return a, b


def _independent_counts(store):
    """Same classes, measured independently of gc_entities (name-list for
    junk, the quality-report SQL for orphans)."""
    junk = store._db.execute(
        "SELECT COUNT(*) FROM entities WHERE LOWER(name) IN ('one', '10', 'then')"
    ).fetchone()[0]
    orphans = store._db.execute(ORPHAN_ENTITIES_SQL).fetchone()[0]
    orph_m = store._db.execute(
        "SELECT COUNT(*) FROM mentions WHERE fact_id NOT IN (SELECT fact_id FROM facts)"
    ).fetchone()[0]
    return {"junk_entities": junk, "orphan_entities": orphans, "orphan_mentions": orph_m}


def test_dry_run_measures_without_deleting(store):
    _legacy_shape(store)
    matcher = junk_matcher()
    dry = store.gc_entities(junk_match=matcher, apply=False)
    independent = _independent_counts(store)
    assert dry == independent  # run-time measurement, asserted externally
    assert dry["junk_entities"] == 3 and dry["orphan_entities"] == 1
    # nothing was deleted
    assert store._db.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 5


def test_sweep_purges_junk_cascades_and_orphans(store):
    a, b = _legacy_shape(store)
    out = store.gc_entities(junk_match=junk_matcher(), apply=True)
    assert out["junk_entities"] == 3
    assert out["junk_mentions_deleted"] == 3  # a/…, b/One, b/10 cascade
    assert out["orphan_entities_deleted"] == 1  # Wombats
    assert out["remaining_orphan_entities"] == 0
    names = {row["name"] for row in store._db.execute("SELECT name FROM entities")}
    assert names == {"Grace"}  # junk gone even though fact b still lives
    # facts and their surviving mentions untouched
    assert store.get(a)["content"] == "Grace ran the 2026 race"
    assert store.get(b)["content"] == "Grace wrote the report again"
    assert store._db.execute(
        "SELECT COUNT(*) FROM mentions WHERE fact_id = ?", (a,)).fetchone()[0] == 1


def test_sweep_is_idempotent(store):
    _legacy_shape(store)
    first = store.gc_entities(junk_match=junk_matcher(), apply=True)
    second = store.gc_entities(junk_match=junk_matcher(), apply=True)
    assert first["junk_entities"] == 3
    assert second["junk_entities"] == 0
    assert second["orphan_entities"] == 0 and second["orphan_mentions"] == 0


def test_sweep_without_matcher_still_cleans_orphans(store):
    store.gc_entities(junk_match=None, apply=True)
    assert store.gc_entities(junk_match=None, apply=True)["orphan_entities"] == 0


def _make_db(tmp_path):
    """A store for the CLI tests (same legacy shape as the sweep tests)."""
    db_path = tmp_path / "limbic.db"
    s = SQLiteStorage(str(db_path), dims=DIMS)
    _legacy_shape(s)
    s.close()
    return db_path


def _cli_args(tmp_path, db_path, gc=False, apply=False):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return SimpleNamespace(hermes_home=str(home), db=str(db_path), gc=gc, apply=apply)


def test_cli_entities_report_is_read_only(tmp_path, capsys):
    db_path = _make_db(tmp_path)
    from limbic import cli as cli_mod

    cli_mod.cmd_entities(_cli_args(tmp_path, db_path, gc=False))
    data = _first_json(capsys.readouterr().out)
    assert data["applied"] is False
    assert data["junk_entities"] == 3 and data["orphan_entities"] == 1
    assert _independent_counts(SQLiteStorage(str(db_path), dims=DIMS))["junk_entities"] == 3


def _first_json(text: str):
    """Parse the first JSON object in the CLI output (pretty-printed, possibly
    followed by a hint line)."""
    return json.JSONDecoder().raw_decode(text.strip())[0]


def test_cli_entities_gc_dry_run_then_apply(tmp_path, capsys):
    db_path = _make_db(tmp_path)
    from limbic import cli as cli_mod

    cli_mod.cmd_entities(_cli_args(tmp_path, db_path, gc=True, apply=False))
    out = capsys.readouterr()
    data = _first_json(out.out)
    assert data["applied"] is False
    assert "Dry run" in out.out.splitlines()[-1]

    cli_mod.cmd_entities(_cli_args(tmp_path, db_path, gc=True, apply=True))
    data = _first_json(capsys.readouterr().out)
    assert data["applied"] is True
    assert data["junk_entities"] == 3 and data["junk_mentions_deleted"] == 3
    assert data["remaining_orphan_entities"] == 0

    s = SQLiteStorage(str(db_path), dims=DIMS)
    names = {row["name"] for row in s._db.execute("SELECT name FROM entities")}
    s.close()
    assert names == {"Grace"}
