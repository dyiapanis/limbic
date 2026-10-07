#!/usr/bin/env python3
"""Limbic reindex — re-embed all facts with the bundled ONNX embedding model.

Used when switching embedding models or after a dimension change.
Creates a new vec_index, re-embeds all facts, swaps atomically.

Usage:
    python3 reindex.py --agent <profile-name>
    python3 reindex.py --all
"""
import argparse
import os
import sys
import time

# honour HERMES_HOME (multiplex/routed homes); default keeps standalone behaviour
_HERMES_HOME = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
sys.path.insert(0, os.path.join(_HERMES_HOME, "plugins"))


def _current_stamp() -> str | None:
    """Stamp for the bundled embedder the reindex is about to write."""
    try:
        from limbic.model_pins import embedder_stamp
        return embedder_stamp()
    except Exception:
        return None


def reindex_agent(agent: str) -> dict:
    """Reindex a single agent's Limbic facts with the bundled ONNX embedder."""
    from limbic.embeddings import create_embedder, EMBEDDING_DIMS
    from limbic.store_sqlite import SQLiteStorage

    hermes = _HERMES_HOME
    db_path = os.path.join(hermes, "profiles", agent, "limbic.db")

    if not os.path.exists(db_path):
        # _HERMES_HOME may point at a profile dir (multiplex homes) while the
        # DB actually lives at ~/.hermes/profiles/<agent>/limbic.db — fall
        # back to the always-canonical layout before giving up.
        alt = os.path.join(os.path.expanduser("~"), ".hermes", "profiles", agent, "limbic.db")
        if os.path.exists(alt):
            db_path = alt
        else:
            return {"agent": agent, "error": "no limbic.db found", "status": "skipped"}

    # Build embedder (bundled ONNX — Arctic Embed 2.0 L int8)
    embedder = create_embedder({})

    # Open storage with current dims for reading
    storage = SQLiteStorage(db_path, dims=EMBEDDING_DIMS, collection_name=f"limbic_{agent}")

    # Count before
    count_before = storage.count()
    print(f"  {agent}: {count_before} facts to reindex")

    # Run reindex — passes the embedder stamp so the store records the model
    # that produced these vectors (mandatory-reindex update mechanism)
    stamp = _current_stamp()
    result = storage.reindex(embedder, EMBEDDING_DIMS, embedder_stamp=stamp)
    storage.close()

    return {"agent": agent, "embedder_stamp": stamp, **result}


def main():
    parser = argparse.ArgumentParser(description="Limbic reindex — re-embed all facts with bundled ONNX model")
    parser.add_argument("--agent", help="Agent profile name (e.g. hermes)")
    parser.add_argument("--all", action="store_true", help="Reindex all agents")
    args = parser.parse_args()

    if args.all:
        # All profile directories under ~/.hermes/profiles that have a limbic.db
        hermes = os.path.expanduser("~/.hermes")
        profiles_root = os.path.join(hermes, "profiles")
        agents = sorted(
            d for d in os.listdir(profiles_root)
            if os.path.exists(os.path.join(profiles_root, d, "limbic.db"))
        )
    elif args.agent:
        hermes = os.path.expanduser("~/.hermes")
        db_path = os.path.join(hermes, "profiles", args.agent, "limbic.db")
        if not os.path.exists(db_path):
            sys.exit(f"No limbic.db at {db_path} — nothing to reindex.")
        agents = [args.agent]
    else:
        parser.error("Specify --agent <name> or --all")

    if not agents:
        sys.exit("No profiles with a limbic.db found — nothing to reindex.")

    from limbic.embeddings import EMBEDDING_MODEL_ID, EMBEDDING_DIMS
    print(f"Limbic reindex — model={EMBEDDING_MODEL_ID} dims={EMBEDDING_DIMS}")
    print(f"Agents: {', '.join(agents)}")
    print()

    # Warm up the model
    print("Warming up embedding model...")
    from limbic.embeddings import create_embedder
    _ = create_embedder({}).embed("warmup")
    print("Model ready.")
    print()

    results = []
    for agent in agents:
        print(f"=== {agent} ===")
        t0 = time.time()
        result = reindex_agent(agent)
        result["elapsed_total"] = round(time.time() - t0, 1)
        results.append(result)

        if result.get("status") == "complete":
            print(f"  ✓ Reindexed: {result.get('reindexed', 0)} facts, {result.get('errors', 0)} errors, {result['elapsed_total']}s")
        else:
            print(f"  ✗ {result.get('error', 'unknown error')}")
        print()

    # Summary
    print("=== Summary ===")
    total_reindexed = sum(r.get("reindexed", 0) for r in results if r.get("status") == "complete")
    total_errors = sum(r.get("errors", 0) for r in results if r.get("status") == "complete")
    failed = [r["agent"] for r in results if r.get("status") not in ("complete", "skipped")]

    print(f"  Reindexed: {total_reindexed} facts")
    print(f"  Errors: {total_errors}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    else:
        print("  All agents: OK")


if __name__ == "__main__":
    main()