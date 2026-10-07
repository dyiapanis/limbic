"""Limbic CLI — operator commands for the limbic memory provider.

Usage: hermes limbic <cmd> [options]

Available commands:
    stats       Show per-user or aggregate statistics
    export      Export all facts for a user (GDPR-style)
    forget      Remove all data for a user (right to be forgotten)
    seed        Inject a fact without conversation
    reindex     Re-embed all facts with the bundled model
    fts-rebuild Rebuild the FTS5 full-text index
    status      Show provider health and config summary
    languages   Show or set entity-extraction languages (e.g. en,fr,el)
"""
from __future__ import annotations

import json
import re
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
        "context_budget": config.get("context_budget", 20),
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
                return cfg.get("nlp", {}).get("languages", ["en"])
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
        provider.save_config({"nlp_languages": ",".join(langs)}, hermes_home)
        print(f"Saved nlp.languages: {', '.join(langs)} -> {config_path}")
        print("spaCy models download automatically on next restart (first use per language, ~14.5MB each).")
    else:
        print(f"Configured: {', '.join(_current())}")
        print(f"Available: {', '.join(sorted(EntityExtractor.MODEL_MAP))}")
        print("Set with: hermes limbic languages --set en,fr,el")


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

    # status
    p_status = subs.add_parser("status", help="Show provider health and config")
    p_status.set_defaults(func=cmd_status)

    # languages
    p_lang = subs.add_parser("languages", help="Show or set entity-extraction languages")
    p_lang.add_argument("--set", help="Comma-separated language codes (e.g. en,fr,el)")
    p_lang.set_defaults(func=cmd_languages)
