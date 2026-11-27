![LIMBIC — Self-Organising Memory for AI Agents](assets/banner.png)

<p align="center"><strong>Limbic</strong><br>Self-organising, adaptive memory for Hermes Agents</p>

<p align="center">
[![tests](https://github.com/dyiapanis/limbic/actions/workflows/tests.yml/badge.svg)](https://github.com/dyiapanis/limbic/actions/workflows/tests.yml)
[![license](https://img.shields.io/github/license/dyiapanis/limbic)](https://github.com/dyiapanis/limbic/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![platforms](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20Windows-informational)](https://github.com/dyiapanis/limbic/actions/workflows/tests.yml)
[![hermes](https://img.shields.io/badge/hermes-plugin%20marketplace-8A2BE2)](https://hermes-agent.nousresearch.com/marketplace/limbic)
</p>

---

## How It Works

Limbic is inspired by the brain's limbic system:

- **Hippocampus** (encoding) — `remember()` embeds, checks interference, and stores
- **Amygdala** (salience) — trust scoring, self-tunes from use
- **Thalamus** (retrieval) — `recall()` searches and gates by relevance
- **Atrophy** — memories superseded, contradicted and unused fade and are progressively forgotten.

The agent sees three tools (`remember`, `recall`, `forget`). The operator sees one file (`limbic.db`). Everything inside is self-organising.

**What it does:**
- New memories start quiet and become easier to recall the more they're used
- Similar memories compete — newer ones supersede older ones, older ones can block similar new ones
- Novel facts get a trust boost; familiar ones don't
- Corrections update trust without ever editing the stored fact — and never below a floor (0.30 default), so a bad edit can't erase a memory
- Use keeps a memory alive, recall resets its decay clock, and reinforcement lifts its trust, so what matters rises while what does not fades.
- A background sweep marks memories that have faded beyond recall and removes them after a review window (7 days) — being recalled during that window saves them.
- Every memory is attributed to whoever said it: raw platform, room or host-minted ids are refused at the boundary, not minted into junk buckets
- If nothing relevant exists for a query, recall says so (`no_confident_match`) rather than guessing
- Relevance thresholds and decay rates tune themselves from use
- Maturity is handled atomically by SQLite triggers

**Research foundations:** [HippoRAG](https://arxiv.org/abs/2405.14831) (NeurIPS 2024), Human-Inspired Memory Architecture (Microsoft 2026), StructMemEval, HEMA — see [SPEC.md](./SPEC.md) for details.

---

## Quick Start

**Platforms:** Linux, macOS and Windows, CI-tested on all three. On Windows the model cache lives at `%USERPROFILE%\.cache\limbic`, and cross-process lock files are a POSIX feature — absent there, so the store initialises lock-free (WAL mode still protects the single-writer, single-user case). All dependencies ship prebuilt wheels for all three platforms; nothing to compile. One interpreter requirement: `sqlite-vec` loads through SQLite's loadable-extension API, so the Python must be built with `--enable-loadable-sqlite-extensions` — Hermes ships such a build, as do most Linux distributions, but python.org's macOS installers do not.

**1 — Install** from the Hermes catalog:

```bash
hermes plugin install limbic
hermes limbic setup   # pre-downloads the ~1GB ONNX models with progress
```

First `remember()` also triggers the same download lazily — `Downloading models…` appears in chat — but `setup` pre-warms with a bar so the first use is instant.

**2 — Enable** (if not using the catalog installer) — in `~/.hermes/profiles/<name>/config.yaml`:

```yaml
memory:
  provider: limbic            # activates Limbic as the memory provider
  memory_enabled: false       # disables the built-in MEMORY.md store
  user_profile_enabled: false # disables the built-in USER.md store
```

**3 — Verify:**

```bash
hermes limbic status        # shows storage, embedding, entity_extraction
```

The agent now has `remember(content)` and `recall(query)`. No indexing, no server. `setup` already enables entity extraction (English) — no extra step for single-language use.

**Multilingual** defaults to English (`en`). To add more:

```bash
hermes limbic languages              # lists available, marks current with *
hermes limbic languages --set it     # saves + auto-downloads with progress
hermes limbic languages --set en,it  # multiple — lowest common set for multi-user
```

When no model is present, `status` shows `entity_extraction: disabled` and memory still works; the next `remember()` will prompt `Downloading models…` in chat. Single config `languages` covers all users in the profile.

Config file `limbic.yaml` is optional (defaults in [Configuration](#configuration)).

---

## Configuration

Each profile has a `limbic.yaml` that sets environmental bounds. The system adapts within these limits:

```yaml
languages: [en] #language support, separate by comma
facts_per_turn: 20 #max facts surfaced per turn
```

---

## Multi-User Mode

**How isolated should memory be?** Pick the weakest boundary that solves your problem:

- **Separate Hermes profiles** — hard isolation, different DBs and agents. Use for different products, tenants, or users who must never share context.
- **One profile + `users.yaml`** — one agent, one store, a separate memory bucket per user: what you store and recall is always attributed to the person speaking. The manifest adds display names and guardian semantics (a guardian may write into a ward's scope; others may not) — and its shape is the same whether there are two users or twenty.

Without the manifest the same isolation still happens: every platform account that talks to the agent gets its own bucket, keyed by their full platform ID (`@grace:hs1` ≠ `@grace42:evil.org`; a single user simply sees one bucket). Adding the manifest later is purely additive — buckets resolve per `user_id`, never per configuration edit.

Enable multi-user mode in `limbic.yaml` with an inline `users:` block — one
config file per deployment (identity only; limbic reads `display_name`,
`scope`, `platform_ids`, `guardians`; roles are ignored here, see
access-control):

```yaml
# limbic.yaml
users:
  grace:
    display_name: Grace
    scope: adult
    platform_ids:
      matrix: "@grace:example.org"
      telegram: "123456789"
    guardians: []
  bob:
    display_name: Bob
    scope: child
    guardians: [grace]
    platform_ids:
      matrix: "@bob:example.org"
```

Alternatively, copy the shipped example next to the plugin code (or set
`users_manifest:` in `limbic.yaml` to any path) — the file layer remains
honoured for existing deployments, with the inline block taking precedence:

```bash
cp ~/.hermes/plugins/limbic/users.yaml.example \
   ~/.hermes/plugins/limbic/users.yaml
```

Nouns defined:

- `platform_ids` — every chat-platform account that belongs to one person;
  all of them resolve to the same memory bucket
- `scope` — a coarse audience tag carried by the memory record (e.g. `adult`,
  `child`). It labels records injected into context; it is **not** an access
  control and does not gate tools by itself
- `guardians` — other users who may *write into* this user's scope (e.g. a
  parent for a child); everyone else stays in their own bucket

The file is deployment-private (gitignored); `users.yaml.example` ships in
its place. Per-profile overrides: `$HERMES_HOME/users.yaml` beats the plugin
directory copy. Other plugins may read the same manifest for their own
identity needs (workspace delivery routing, credential prefixes) — it is the
one identity file in the stack.

Identity is guarded in depth: the memory boundary refuses ids that are
room-shaped or host-minted (18+ character opaque ids and `!`-room ids), and
re-attributes any raw id that resolves in the manifest. Short opaque ids in
single-user bare mode still become their own bucket (no manifest to resolve
against) — add `users.yaml` for strict attribution. Resolving unregistered
ids to a person, `os_uid:` mapping for local peers, and fail-closed unmapped
homes are stack-level guarantees (Hermes host lane), not plugin claims.

---

## Operator Functions

For the human running Hermes (`hermes limbic ...`):

```
hermes limbic forget <user_id>          # right to be forgotten
hermes limbic forget-fact <fact_id> --user <user_id>   # remove a single wrong fact
hermes limbic export <user_id>          # data export (lists fact IDs)
hermes limbic seed <user_id> <content>  # inject facts without conversation
hermes limbic reindex                   # re-embed all facts with bundled model
hermes limbic stats [user_id]           # per-user or aggregate metrics
hermes limbic status [user_id]          # atrophy candidates, storage health
hermes limbic entities                  # entity-graph counts
hermes limbic entities --gc             # sweep junk/orphan entities (dry run; add --apply to commit)
```

### Deleting or correcting a wrong fact

When a memory is wrong, the fix depends on why it is wrong. Two paths,
by intent:

- **The fact was wrong from the start** (a misheard name, a bad extraction):
  remove it by ID — `hermes limbic export <user_id>` lists fact IDs,
  `hermes limbic forget-fact <fact_id> --user <user_id>` deletes it (the
  ownership check refuses another user's fact). Every deletion, on every
  path, writes a hash-only tombstone receipt — proof of what was removed,
  never a copy of the content itself.
- **The world changed** (a newer value supersedes an old one): just store the
  correction. Write-time NLI links the pair, the old fact's trust is
  penalised, and recall prefers the correction whenever both are relevant
  to the query. The fade is still gradual — trust dynamics and maturation
  (~a week at the default curve) — but the correction wins from the day
  it is stored, not after the week.

**When a plugin upgrade changes the bundled embedder, reindexing is
required.** Limbic ships its embedding model, and the store records which
embedder produced its vectors (the embedder stamp). If an upgrade ships a
different bundled model, `status` and `stats` report `stale_embedder: true`,
retrieval quality is degraded, and one command realigns the store (run on the
host where Hermes is installed — the script imports Hermes core):

```bash
python3 ~/.hermes/plugins/limbic/reindex.py --agent <profile-name>
python3 ~/.hermes/plugins/limbic/reindex.py --all          # every profile
```

---

## Architecture

See [SPEC.md](./SPEC.md) for the full architecture — data model, lifecycle mechanisms, trust adaptation, retrieval pipeline, atrophy, and roadmap. See [FAQ.md](./FAQ.md) for common follow-up questions.

## Hardware Requirements

Hermes Agent runs at 1 GB RAM / 1 CPU / 500 MB disk. Limbic adds:

- **RAM (per profile):** ~700 MB resident once the memory is active — each profile loads its own ONNX embedder session; sessions are not shared across profiles. The NLI cross-encoder loads lazily on the first contradiction check and adds to that.
- **Disk (shared):** ~1 GB model cache, downloaded once for ALL profiles (Arctic int8 ~570 MB + NLI ~430 MB). The English NER pipeline ships inside the plugin (~15 MB, no download). Per profile, the only growth is `limbic.db` — a few MB per 1,000 memories.
- **CPU (shared):** 2 cores recommended. Embedding is ~17–19 ms int8 ONNX on CPU (measured on a minimalist non-GPU server); the NLI gate runs at write time only. No GPU.

See SPEC.md §Performance budget / §Scale projections for measured numbers.

