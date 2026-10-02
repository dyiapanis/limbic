---
name: using-limbic
description: "Use when the active memory provider is Limbic. Covers remember/recall tool usage, fact quality guidance, and operator CLI commands."
version: 1.0.0
author: Limbic
license: PolyForm-Noncommercial-1.0.0
metadata:
  hermes:
    tags: [memory, limbic, remember, recall, facts]
---

# Using Limbic Memory

Limbic is the active memory provider. It persists facts across sessions and injects relevant context into each turn via `prefetch()`. You interact with it through two tools: `remember` and `recall`.

## Tools

### `remember(content)`

Store a durable fact. Write **declarative statements**, not instructions or conversation fragments.

**Good:**
- `User prefers concise responses`
- `Project uses pytest with xdist`
- `Server runs Debian 13, kernel 6.12`

**Bad (will be rejected or decay quickly):**
- `Always respond concisely` (imperative — phrased as a directive to yourself)
- `User said they like that` (conversation fragment, no durable value)
- `Phase 1 done, PR #1234 merged` (ephemeral task state — stale in a week)
- `Fixed bug X yesterday` (ephemeral work log)

**What belongs in memory:**
- User preferences and communication style
- Environment facts (OS, tools, project structure)
- Stable conventions and workflows
- Tool quirks and lessons learned

**What does NOT belong:**
- Task progress, session outcomes, completed-work logs
- PR numbers, issue numbers, commit SHAs
- Temporary TODO state
- Anything that will be stale in 7 days

### `recall(query)`

Search stored facts by semantic similarity. Returns matching facts ranked by composite score (embedding similarity + trust + activation). Use this when you need to check what's already known before asking the user to repeat themselves.

## How Memory Surfaces

At the start of each turn, Limbic runs `prefetch()` automatically. It searches for facts relevant to the current user message and injects them as a `<limbic-context>` block in the system prompt. You don't need to call `recall()` to see your memories — they appear automatically.

The `recall()` tool is for explicit searches when you want to verify something specific.

## Trust and Atrophy

Facts are not permanent by default. Limbic manages their lifecycle:

- **Trust score** — starts at 0.50, rises with retrieval and continuation signals, falls with corrections and interference from newer facts.
- **Temporal decay** — unretrieved facts slowly decay through an Ebbinghaus forgetting curve. Orphans (never-retrieved facts) eventually fade.
- **Corrections** — if the user says "no", "wrong", "actually", or "that's not right" about a topic matching a stored fact, that fact's trust is penalised.
- **Atrophy** — facts below the trust threshold are marked, then physically deleted after a grace period. This is automatic and organic.
- **No pinning** — durable facts earn no exemption; high trust just means they surface more often while used, and they fade like anything else when they stop being used.

You don't manage this lifecycle. Just write good facts and let Limbic handle the rest.

## Operator CLI

```bash
hermes limbic status                          # provider health summary
hermes limbic stats [--user <user_id>]       # per-user or aggregate statistics
hermes limbic export --user <user_id> [-o <file>]  # export all facts (JSON)
hermes limbic forget --user <user_id> --confirm     # wipe all facts for a user (irreversible)
hermes limbic forget-fact <fact_id> [--user <user_id>]  # delete ONE fact by ID (2026-08-29)
hermes limbic seed --user <user_id> --content "..."  # inject a fact from the operator
hermes limbic reindex [--dims N]                    # re-embed all facts (bundled model)
hermes limbic fts-rebuild                       # rebuild the FTS5 index
hermes limbic languages [--set en,fr,el]        # show/set entity-extraction languages
```

**Notes:**
- `forget-fact` is the surgical delete path — use it over `forget` whenever only one fact is wrong.
- `seed` is the operator injection path — use it to bootstrap a new user or restore facts.
- `reindex` re-embeds all facts with the bundled model. Required if you change embedding dimensions.

## ⛔ Never edit `limbic.db` directly

All reads and writes go through the plugin — the CLI verbs above or the provider API. Never `sqlite3 limbic.db "DELETE ..."` and never bare-copy the file while a gateway runs.

1. **WAL race** — running gateways hold the DB in WAL mode. A bare-file copy misses WAL-held rows and silently loses data (2026-08-29: 3 facts lost this way during a restore).
2. **Index consistency** — facts have companion rows in `vec_index` (KNN) and `facts_fts` (BM25). Raw SQL that skips companion-row cleanup leaves facts invisible to search or orphaned index rows.
3. **Trust invariants** — decay and atrophy assume all writes went through `storage.py` under its lock.
4. **Missing verb?** Request it (file an issue) instead of improvising SQL.

To inspect a user's facts, use `hermes limbic export --user <uid>` — offline-safe JSON read, no lock contention.

## Configuration

Limbic reads config from `$HERMES_HOME/limbic.yaml`:

```yaml
embedding:
  provider: local          # bundled ONNX — no endpoint option
  model: Snowflake/snowflake-arctic-embed-l-v2.0
  dims: 1024

nlp:
  languages: [en]          # spaCy models for entity extraction
```

Set via `hermes memory setup` (interactive) or edit `limbic.yaml` directly.

## Storage

- **`limbic.db`** — single SQLite file per profile, WAL mode
- **sqlite-vec** — vector KNN search (brute-force linear scan, no ANN index)
- **FTS5** — full-text search for BM25 hybrid retrieval
- Backup: `hermes limbic export` or the storage-level backup (`VACUUM INTO`) — do NOT raw-copy a live WAL database (see the policy section above)