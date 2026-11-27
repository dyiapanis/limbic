"""Limbic CLI — operator commands for the limbic memory provider.

Usage: hermes limbic <cmd> [options]

Available commands:
    stats       Show per-user or aggregate statistics
    export      Export all facts for a user (GDPR-style)
    forget      Remove all data for a user (right to be forgotten)
    seed        Inject a fact without conversation
    reindex     Re-embed all facts with the bundled model
    fts-rebuild Rebuild the FTS5 full-text index
    entities-backfill  Populate entities/mentions for facts written while NLP was disabled
    entities    Entity-graph hygiene: counts; --gc --apply sweeps junk/orphans
    quality     Store-quality metrics (read-only) + optional VACUUM INTO backup
    status      Show provider health and config summary
    languages   Show or set entity-extraction languages (e.g. en,fr,el)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# Version — single source of truth is plugin.yaml (read once at import)
PLUGIN_VERSION = "0.5.1"


def _plugin_version() -> str:
    try:
        with open(Path(__file__).with_name("plugin.yaml")) as f:
            m = re.search(r"^version:\s*(.+)$", f.read(), re.M)
            if m:
                return m.group(1).strip().strip("'\"")
    except Exception:
        pass
    return PLUGIN_VERSION


def _get_provider():
    """Resolve the active Limbic provider via the memory manager."""
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
        provider_name = config.get("memory", {}).get("provider", "")
        if provider_name != "limbic":
            print(f"ERROR: Active memory provider is '{provider_name}', not 'limbic'.")
            sys.exit(1)

        from agent.memory_manager import MemoryManager
        mgr = MemoryManager()
        # The manager may not have providers loaded in CLI context
        # Fall back to direct instantiation
        from .provider import LimbicMemoryProvider
        provider = LimbicMemoryProvider()
        return provider
    except Exception as e:
        print(f"ERROR: Could not initialise Limbic provider: {e}")
        sys.exit(1)


def _resolve_hermes_home(args) -> str:
    """Resolve HERMES_HOME from args or env."""
    import os
    home = getattr(args, "hermes_home", None) or os.environ.get("HERMES_HOME", "")
    if home:
        return home
    # Fallback to current profile
    try:
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return str(Path.home() / ".hermes" / "profiles" / "default")


def _resolve_agent_identity() -> str:
    """Resolve the active profile name for namespace scoping.

    The gateway injects agent_identity via agent_init.py, but CLI calls
    bypass that path. Without this, the provider defaults to 'default'
    and queries a different namespace than the gateway —
    seeing 0 facts even when the gateway stored facts under the profile
    namespace (e.g. ns=hermes has 78 facts, ns=default has 0).
    """
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "default"


def cmd_stats(args):
    """Show per-user or aggregate statistics."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-stats", hermes_home=hermes_home, agent_identity=agent_identity)

    user_id = getattr(args, "user", None)
    result = provider.stats(user_id)
    print(json.dumps(result, indent=2, default=str))


def cmd_export(args):
    """Export all facts for a user."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-export", hermes_home=hermes_home, agent_identity=agent_identity)

    user_id = getattr(args, "user", None)
    if not user_id:
        print("ERROR: --user required")
        sys.exit(1)

    result = provider.export(user_id)
    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(json.dumps(result, indent=2, default=str))
        print(f"Exported to {output}")
    else:
        print(json.dumps(result, indent=2, default=str))


def cmd_forget(args):
    """Remove all data for a user."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-forget", hermes_home=hermes_home, agent_identity=agent_identity)

    user_id = getattr(args, "user", None)
    if not user_id:
        print("ERROR: --user required")
        sys.exit(1)

    confirm = getattr(args, "confirm", False)
    if not confirm:
        print(f"WARNING: This will remove ALL memory data for user '{user_id}'.")
        print("Run again with --confirm to proceed.")
        sys.exit(1)

    result = provider.forget(user_id)
    print(json.dumps(result, indent=2, default=str))


def cmd_forget_fact(args):
    """Remove a single fact by ID."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-forget-fact", hermes_home=hermes_home, agent_identity=agent_identity)

    user_id = getattr(args, "user", None)
    if not user_id:
        print("ERROR: --user is required for forget-fact (ownership check).")
        sys.exit(1)
    result = provider.forget_fact(args.fact_id, user_id=user_id)
    print(json.dumps(result, indent=2, default=str))


def cmd_seed(args):
    """Inject a fact without conversation."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-seed", hermes_home=hermes_home, agent_identity=agent_identity)

    user_id = getattr(args, "user", None)
    content = getattr(args, "content", None)
    if not user_id or not content:
        print("ERROR: --user and --content required")
        sys.exit(1)

    result = provider.seed(user_id, content)
    print(json.dumps(result, indent=2, default=str))


def cmd_reindex(args):
    """Re-embed all facts with the bundled ONNX model."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-reindex", hermes_home=hermes_home, agent_identity=agent_identity)

    new_dims = getattr(args, "dims", None)

    # Build embedder — bundled ONNX only (create_embedder ignores any
    # provider/model fields, so we don't accept them).
    from .embeddings import create_embedder, EMBEDDING_DIMS
    new_embedder = create_embedder(provider._config)
    new_dims_val = int(new_dims) if new_dims else EMBEDDING_DIMS

    # Re-embed + re-stamp: the stamp tells the store which embedder produced
    # these vectors (the recover path the stale_embedder warning points at —
    # without it the flag would never clear).
    from .model_pins import embedder_stamp
    result = provider._storage.reindex(
        new_embedder, new_dims_val, embedder_stamp=embedder_stamp()
    ) if provider._storage else {"error": "storage not initialised"}
    print(json.dumps(result, indent=2, default=str))


def cmd_fts_rebuild(args):
    """Rebuild the FTS5 full-text index from the facts table."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-fts-rebuild", hermes_home=hermes_home, agent_identity=agent_identity)

    storage = provider._storage
    if not storage:
        print("ERROR: storage not initialised")
        sys.exit(1)
    storage.rebuild_fts()
    print("FTS index rebuilt.")


def cmd_entities_backfill(args):
    """Backfill entities/mentions for facts written while extraction was disabled (issue #18).

    Idempotent: store_entities uses INSERT OR IGNORE + content-hash ids, so re-runs
    add nothing. Refuses when the spaCy pipeline is unavailable — reports the fix
    instead of pretending.
    """
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-entities-backfill", hermes_home=hermes_home, agent_identity=agent_identity)

    storage = provider._storage
    if not storage:
        print("ERROR: storage not initialised")
        sys.exit(1)
    extractor = provider._entity_extractor
    if not extractor or not extractor.available:
        print("ERROR: entity extraction unavailable — spaCy model not installed. Fix:")
        print("  " + sys.executable + " -m spacy download en_core_web_sm")
        sys.exit(2)

    def _counts():
        # Direct SQL: entities/mentions table sizes (SQLiteStorage owns the schema)
        try:
            e = storage._db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            m = storage._db.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
            return e, m
        except Exception:
            return (0, 0)

    users = storage.facts_per_user() or []
    scanned = new_e = new_m = 0
    for u in users:
        user_id = u.get("user_id", "unknown")
        exported = storage.export_user(user_id)
        facts = exported.get("facts", []) if isinstance(exported, dict) else []
        for f in facts:
            fid, content = f.get("fact_id"), f.get("content")
            if not fid or not content:
                continue
            scanned += 1
            entities = extractor.extract(content)
            if not entities:
                continue
            before = _counts()
            storage.store_entities(fid, entities)
            after = _counts()
            new_e += max(0, after[0] - before[0])
            new_m += max(0, after[1] - before[1])
    print(f"OK: scanned {scanned} facts — new entities {new_e}, new mentions {new_m}")


def cmd_entities(args):
    """Entity-graph hygiene: counts by class; --gc --apply sweeps.

    Default is a read-only report (junk entities, orphan mentions, orphan
    entities measured live against the store — never a cached audit
    snapshot). --gc previews the same classes; --gc --apply deletes (junk
    entities first with their mention rows, then both orphan classes) and
    re-verifies against a live re-measure. Never initialises the provider —
    direct store handle, same resolution as cmd_quality.
    """
    hermes_home = _resolve_hermes_home(args)
    db_path = getattr(args, "db", None) or str(Path(hermes_home) / "limbic.db")
    if not Path(db_path).exists():
        print(f"ERROR: store not found: {db_path}")
        sys.exit(1)

    from .entities import junk_matcher
    from .store_sqlite import SQLiteStorage

    extras: list[str] = []
    yaml_path = Path(hermes_home) / "limbic.yaml"
    if yaml_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(yaml_path.read_text()) or {}
            extras = [str(w) for w in (cfg.get("entity_junk_words") or [])]
        except Exception:
            pass

    storage = SQLiteStorage(db_path)
    try:
        sweep = bool(getattr(args, "gc", False))
        if getattr(args, "apply", False) and not sweep:
            print("NOTE: --apply without --gc does nothing — the report ran read-only.")
        result = storage.gc_entities(
            junk_match=junk_matcher(extras),
            apply=bool(sweep and getattr(args, "apply", False)),
        )
    finally:
        storage.close()

    result["store"] = db_path
    result["applied"] = bool(sweep and getattr(args, "apply", False))
    print(json.dumps(result, indent=2, default=str))
    if sweep and not getattr(args, "apply", False):
        print("Dry run — run again with --apply to delete.")


def cmd_status(args):
    """Show provider health and config summary."""
    provider = _get_provider()
    hermes_home = _resolve_hermes_home(args)
    agent_identity = _resolve_agent_identity()
    provider.initialize(session_id="cli-status", hermes_home=hermes_home, agent_identity=agent_identity)

    storage = provider._storage
    config = provider._config

    status = {
        "provider": "limbic",
        "version": _plugin_version(),
        "available": provider.is_available(),
        "hermes_home": hermes_home,
        "storage_backend": config.get("storage", {}).get("provider", "sqlite"),
        "collection": getattr(storage, "_collection", "unknown"),
        "total_facts": storage.count() if storage else 0,
        "embedding": config.get("embedding", {}),
        "facts_per_turn": config.get(
            "facts_per_turn", config.get("context_budget", 20)
        ),
        "entity_extraction": {
            "enabled": bool(provider._entity_extractor and provider._entity_extractor.available),
            "languages": getattr(provider._entity_extractor, "_languages", None),
        },
    }
    print(json.dumps(status, indent=2, default=str))


def cmd_languages(args):
    """Show or set entity-extraction languages (persisted to $HERMES_HOME/limbic.yaml)."""
    import os
    from .entities import EntityExtractor

    hermes_home = _resolve_hermes_home(args)
    config_path = os.path.join(hermes_home, "limbic.yaml")

    def _current() -> list[str]:
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f) or {}
                return (
                    cfg.get("languages")
                    or cfg.get("nlp", {}).get("languages")
                    or ["en"]
                )  # canonical: languages (legacy: nlp.languages)
            except Exception:
                pass
        return ["en"]  # plugin default

    if getattr(args, "set", None):
        langs = [l.strip() for l in str(args.set).split(",") if l.strip()]
        invalid = [l for l in langs if l not in EntityExtractor.MODEL_MAP]
        if invalid:
            print(f"ERROR: unknown language code(s): {', '.join(invalid)}")
            print(f"Available: {', '.join(sorted(EntityExtractor.MODEL_MAP))}")
            sys.exit(1)
        provider = _get_provider()
        provider.save_config({"languages": ",".join(langs)}, hermes_home)
        print(f"Saved languages: {', '.join(langs)} -> {config_path}")
        # auto-install models for the selected languages
        import subprocess
        for l in langs:
            model = EntityExtractor.MODEL_MAP[l]
            print(f"Installing {model} ...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model], check=False)
        return
    # show: list available with plain names, mark selected
    current = set(_current())
    print(f"Configured: {', '.join(_current())}")
    print("Available:")
    for code in sorted(EntityExtractor.MODEL_MAP):
        mark = " *" if code in current else "  "
        print(f"{mark} {code:4s}  {EntityExtractor.MODEL_MAP[code]}")
    print("\nSet with: hermes limbic languages --set en,it  (then models install automatically)")


def cmd_setup(args):
    print("Setting up Limbic — downloading models with progress:\n")
    from .embeddings import OnnxEmbedder
    from .nli import LimbicNLI
    print("[1/3] Embedding model (~570MB)...")
    OnnxEmbedder()
    print("  ✓ embedding ready")
    print("[2/3] NLI model (~430MB)...")
    LimbicNLI()
    print("  ✓ NLI ready")
    print("[3/3] Entity models...")
    from .entities import EntityExtractor
    import subprocess
    langs = ["en"]
    try:
        import os, yaml
        cfg_path = os.path.join(_resolve_hermes_home(args), "limbic.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            langs = (
                cfg.get("languages")
                or cfg.get("nlp", {}).get("languages")
                or ["en"]
            )  # canonical: languages (legacy: nlp.languages)
            if isinstance(langs, str):
                langs = [l.strip() for l in langs.split(",")]
    except Exception:
        pass
    for l in langs:
        model = EntityExtractor.MODEL_MAP.get(l, "")
        if not model:
            continue
        print(f"  {l} -> {model} ...")
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=False)
    print("\n✓ Setup complete. Run `hermes limbic status` to verify.")


# ── quality (store-quality sprint C1) ───────────────────────────────
#
# Pinned metric definitions — the exact SQL the report runs. All reads are
# read-only against limbic.db (the sanctioned forensic lane; this module
# never writes a store — the only file it creates is a fresh VACUUM INTO
# backup, and only when --backup-dir is given). `last_retrieved` is stored
# ISO-8601 UTC ('YYYY-MM-DDTHH:MM:SS...'), so the stale-window cutoff is
# strftime'd into the same T-format for a correct byte comparison.
ROOM_ID_RE = re.compile(r"![^:]+:.+")
ORPHAN_ENTITIES_SQL = (
    "SELECT COUNT(*) FROM entities "
    "WHERE entity_id NOT IN (SELECT DISTINCT entity_id FROM mentions)"
)
STALE_30D_SQL = (
    "SELECT COUNT(*) FROM facts WHERE last_retrieved IS NULL "
    "OR last_retrieved < strftime('%Y-%m-%dT%H:%M:%S', datetime('now','-30 days'))"
)
CONTRADICTION_EVENT_SQL = (
    "SELECT (SELECT COUNT(*) FROM contradictions), "
    "(SELECT COUNT(*) FROM contradictions WHERE resolved_at IS NULL), "
    "(SELECT COUNT(DISTINCT fact_id) FROM "
    "(SELECT newer_fact_id AS fact_id FROM contradictions "
    "UNION SELECT older_fact_id FROM contradictions))"
)
TRUST_BUCKETS_SQL = (
    "SELECT COUNT(*), COUNT(trust_score), MIN(trust_score), MAX(trust_score), "
    "COALESCE(SUM(trust_score < 0.02), 0), "
    "COALESCE(SUM(trust_score < 0.05), 0), "
    "COALESCE(SUM(trust_score >= 0.05 AND trust_score < 0.20), 0), "
    "COALESCE(SUM(trust_score >= 0.20 AND trust_score < 0.40), 0), "
    "COALESCE(SUM(trust_score >= 0.40 AND trust_score < 0.60), 0), "
    "COALESCE(SUM(trust_score >= 0.60 AND trust_score < 0.80), 0), "
    "COALESCE(SUM(trust_score >= 0.80), 0) FROM facts"
)
LENGTH_BUCKETS_SQL = (
    "SELECT COUNT(*), "
    "COALESCE(SUM(LENGTH(content) < 40), 0), "
    "COALESCE(SUM(LENGTH(content) >= 40 AND LENGTH(content) < 600), 0), "
    "COALESCE(SUM(LENGTH(content) >= 600 AND LENGTH(content) < 1100), 0), "
    "COALESCE(SUM(LENGTH(content) >= 1100), 0) FROM facts"
)


def quality_report(db_path) -> dict:
    """Store-quality metrics for one limbic.db — read-only SQL, definitions pinned above."""
    import statistics
    from datetime import datetime, timezone
    from urllib.parse import quote

    con = sqlite3.connect(f"file:{quote(str(db_path))}?mode=ro", uri=True)
    try:
        total, have_trust, t_min, t_max, lt2, lt5, b1, b2, b3, b4, b5 = con.execute(TRUST_BUCKETS_SQL).fetchone()
        trusts = [r[0] for r in con.execute("SELECT trust_score FROM facts WHERE trust_score IS NOT NULL")]
        users_total = con.execute("SELECT COUNT(DISTINCT user_id) FROM facts").fetchone()[0]
        ever, unresolved, involved = con.execute(CONTRADICTION_EVENT_SQL).fetchone()
        never_recalled = con.execute("SELECT COUNT(*) FROM facts WHERE retrieval_count = 0").fetchone()[0]
        stale_30d = con.execute(STALE_30D_SQL).fetchone()[0]
        orphans = con.execute(ORPHAN_ENTITIES_SQL).fetchone()[0]
        entities_total = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        mentions_rows = con.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
        l1, l2, l3, l4, l5 = con.execute(LENGTH_BUCKETS_SQL).fetchone()
        users = [
            {"user_id": r[0], "facts": r[1], "room_shaped": bool(ROOM_ID_RE.fullmatch(r[0] or ""))}
            for r in con.execute("SELECT user_id, COUNT(*) AS n FROM facts GROUP BY user_id ORDER BY n DESC, user_id")
        ]
        matured, atrophy_marked = con.execute(
            "SELECT COALESCE(SUM(matured_at IS NOT NULL), 0), "
            "COALESCE(SUM(atrophy_marked_at IS NOT NULL), 0) FROM facts").fetchone()
        newest = con.execute("SELECT MAX(created_at) FROM facts").fetchone()[0]
    finally:
        con.close()

    def share(n, of):
        return round(n / of, 4) if of else 0.0

    room = [u for u in users if u["room_shaped"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": str(db_path),
        "facts_total": total,
        "users_total": users_total,
        "newest_fact": newest,
        "trust": {
            "mean": round(statistics.fmean(trusts), 4) if trusts else None,
            "median": round(statistics.median(trusts), 4) if trusts else None,
            "min": t_min, "max": t_max,
            "buckets": {"<0.05": lt5, "0.05-0.20": b1, "0.20-0.40": b2, "0.40-0.60": b3,
                        "0.60-0.80": b4, ">=0.80": b5, "none": total - have_trust},
            "lt_0_02": lt2,
            "zombie_lt_0_05_share": share(lt5, total),
        },
        "never_recalled": {
            "count": never_recalled, "share": share(never_recalled, total),
            "stale_30d": stale_30d, "stale_30d_share": share(stale_30d, total),
        },
        "contradictions": {
            "total": ever, "unresolved": unresolved, "facts_involved": involved,
            "density_per_fact": round(ever / total, 3) if total else None,
            "events_per_involved_fact": round(ever / involved, 3) if involved else None,
        },
        "entities": {"total": entities_total, "orphans": orphans,
                     "orphan_ratio": share(orphans, entities_total), "mentions_rows": mentions_rows},
        "length": {
            "buckets": {"<40": l2, "40-599": l3, "600-1099": l4, ">=1100": l5},
            "shares": {"<40": share(l2, l1), "40-599": share(l3, l1),
                       "600-1099": share(l4, l1), ">=1100": share(l5, l1)},
        },
        "users": users,
        "room_shaped": {"user_ids": [u["user_id"] for u in room], "facts": sum(u["facts"] for u in room)},
        "liveness": {"matured": matured, "atrophy_marked": atrophy_marked},
    }


def backup_store(db_path, dest_dir, profile=None) -> dict:
    """Timestamped VACUUM INTO backup of one store.

    Safe against a live WAL writer — the source is only ever read (a consistent
    snapshot); VACUUM INTO writes the fresh destination file, never the source.
    Verified before returning: integrity_check ok and fact count matches source.
    """
    from datetime import datetime, timezone
    from urllib.parse import quote

    src = Path(db_path)
    dest_dir = Path(dest_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"{profile or src.parent.name}-{src.stem}-{ts}{src.suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{quote(str(src))}?mode=ro", uri=True)
    try:
        con.execute("VACUUM INTO ?", (str(dest),))
        src_n = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    finally:
        con.close()
    chk = sqlite3.connect(f"file:{quote(str(dest))}?mode=ro", uri=True)
    try:
        integrity = chk.execute("PRAGMA integrity_check").fetchone()[0]
        dest_n = chk.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    finally:
        chk.close()
    if integrity != "ok" or dest_n != src_n:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"ERROR: backup failed verification ({dest.name}): integrity={integrity}, facts {src_n}->{dest_n}")
    return {"path": str(dest), "facts": dest_n, "integrity": integrity}


def _load_principals(path):
    """Parse a users.yaml manifest → the set of principal-scoped user_id values:
    canonical keys + every platform_ids value (matrix user ids kept raw and
    @-stripped). Pass the fleet manifest via --principals; no principal names
    are ever written into this repo."""
    import yaml

    with open(path) as f:
        manifest = yaml.safe_load(f) or {}
    users = manifest.get("users", {}) or {}
    ids = {u for u in users if isinstance(u, str)}
    for u in users.values():
        if not isinstance(u, dict):
            continue
        for v in (u.get("platform_ids") or {}).values():
            if isinstance(v, str):
                ids.add(v)
                if v.startswith("@"):
                    ids.add(v[1:])
    return ids


def cmd_quality(args):
    """Store-quality metrics report (read-only; sprint C1 instrument).

    Deliberately does NOT initialise the provider — pure sqlite reads on the
    resolved store, nothing the profile's running gateway can notice.
    """
    hermes_home = _resolve_hermes_home(args)
    db_path = getattr(args, "db", None) or str(Path(hermes_home) / "limbic.db")
    if not Path(db_path).exists():
        print(f"ERROR: store not found: {db_path}")
        sys.exit(1)
    principals_path = getattr(args, "principals", None)
    if principals_path and not Path(principals_path).exists():
        print(f"ERROR: principals manifest not found: {principals_path}")
        sys.exit(2)
    result = quality_report(db_path)
    result["profile"] = Path(db_path).parent.name
    if principals_path:
        principals = _load_principals(principals_path)
        other = [u for u in result["users"] if u["user_id"] not in principals]
        result["user_scoping"] = {
            "non_principal_ids": [u["user_id"] for u in other],
            "non_principal_facts": sum(u["facts"] for u in other),
            "definition": "user_id values absent from the --principals manifest (canonical keys + platform_ids)",
        }
    backup_dir = getattr(args, "backup_dir", None)
    if backup_dir:
        result["backup"] = backup_store(db_path, backup_dir, result["profile"])
    output = getattr(args, "output", None)
    if output:
        Path(output).write_text(json.dumps(result, indent=2, default=str))
        print(f"Quality report written to {output}")
    else:
        print(json.dumps(result, indent=2, default=str))


# ── Argparse registration ──────────────────────────────────────────


def register_cli(subparser) -> None:
    """Build the `hermes limbic` argparse tree.

    Called by discover_plugin_cli_commands() at setup time.
    """
    parser = subparser
    parser.add_argument(
        "--hermes-home",
        help="HERMES_HOME path (defaults to current profile)",
        default=None,
    )

    subs = parser.add_subparsers(dest="limbic_command", required=True)

    # stats
    p_stats = subs.add_parser("stats", help="Show per-user or aggregate statistics")
    p_stats.add_argument("--user", help="User ID (omit for aggregate)")
    p_stats.set_defaults(func=cmd_stats)

    # export
    p_export = subs.add_parser("export", help="Export all facts for a user")
    p_export.add_argument("--user", required=True, help="User ID to export")
    p_export.add_argument("--output", "-o", help="Write to file instead of stdout")
    p_export.set_defaults(func=cmd_export)

    # forget
    p_forget = subs.add_parser("forget", help="Remove all data for a user")
    p_forget.add_argument("--user", required=True, help="User ID to forget")
    p_forget.add_argument("--confirm", action="store_true", help="Confirm destructive operation")
    p_forget.set_defaults(func=cmd_forget)

    # forget-fact
    p_ff = subs.add_parser("forget-fact", help="Remove a single fact by ID")
    p_ff.add_argument("fact_id", type=int, help="Fact ID to remove")
    p_ff.add_argument("--user", help="Refuse if fact belongs to another user")
    p_ff.set_defaults(func=cmd_forget_fact)

    # seed
    p_seed = subs.add_parser("seed", help="Inject a fact without conversation")
    p_seed.add_argument("--user", required=True, help="User ID to seed")
    p_seed.add_argument("--content", required=True, help="Fact content to store")
    p_seed.set_defaults(func=cmd_seed)

    # reindex — the bundled ONNX embedder is the only provider; --dims is
    # the one meaningful override (it must match the model actually bundled).
    p_reindex = subs.add_parser("reindex", help="Re-embed all facts (bundled ONNX model)")
    p_reindex.add_argument("--dims", type=int, help="Embedding dimensions (e.g. 1024)")
    p_reindex.set_defaults(func=cmd_reindex)

    # fts-rebuild
    p_fts = subs.add_parser("fts-rebuild", help="Rebuild the FTS5 full-text index")
    p_fts.set_defaults(func=cmd_fts_rebuild)

    # entities-backfill (issue #18)
    p_eb = subs.add_parser("entities-backfill", help="Populate entities/mentions for facts written while NLP was disabled")
    p_eb.set_defaults(func=cmd_entities_backfill)

    # entities — entity-graph hygiene: junk + orphan counts, optional --gc sweep
    p_ent = subs.add_parser("entities", help="Entity-graph hygiene counts; --gc sweeps junk/orphans")
    p_ent.add_argument("--db", help="Store path override (default: <hermes-home>/limbic.db)")
    p_ent.add_argument("--gc", action="store_true",
                       help="Sweep mode: junk + orphan classes (dry run unless --apply)")
    p_ent.add_argument("--apply", action="store_true",
                       help="Confirm the destructive sweep (with --gc)")
    p_ent.set_defaults(func=cmd_entities)

    # status
    p_status = subs.add_parser("status", help="Show provider health and config")
    p_status.set_defaults(func=cmd_status)

    # quality — store-quality sprint: read-only metrics + optional VACUUM INTO backup
    p_quality = subs.add_parser("quality", help="Store-quality metrics (read-only) + optional VACUUM INTO backup")
    p_quality.add_argument("--db", help="Store path override (default: <hermes-home>/limbic.db)")
    p_quality.add_argument("--principals", help="users.yaml manifest path: classifies user_id scoping (canonical keys + platform_ids are principal)")
    p_quality.add_argument("--backup-dir", help="Timestamped VACUUM INTO backup of the store into this directory")
    p_quality.add_argument("-o", "--output", help="Write the report JSON to a file")
    p_quality.set_defaults(func=cmd_quality)

    # languages
    p_lang = subs.add_parser("languages", help="Show or set entity-extraction languages")
    p_lang.add_argument("--set", help="Comma-separated language codes (e.g. en,fr,el)")
    p_lang.set_defaults(func=cmd_languages)

    # setup — one-shot install with progress: HF models + spacy models
    p_setup = subs.add_parser("setup", help="Download all models with progress")
    p_setup.set_defaults(func=cmd_setup)


def main(argv=None):
    """Standalone entry: `python3 -m limbic.cli quality …` — same argparse tree
    the Hermes integration mounts via register_cli(subparser)."""
    parser = argparse.ArgumentParser(prog="limbic", description="Limbic operator CLI")
    register_cli(parser)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
