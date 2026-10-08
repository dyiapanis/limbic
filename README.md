![LIMBIC — Self-Organising Memory for AI Agents](assets/banner.png)

Self-Organising Memory for AI Agents

**Version 0.6.0** · Author: D Yiapanis · License: PolyForm Noncommercial 1.0.0 (source-available; commercial use requires a separate license)

Limbic is a self-organising memory system for AI agents. It stores facts, learns which ones matter through interaction, and surfaces the right context at the right time — without ever calling an LLM. Designed for [Hermes Agent](https://hermes-agent.nousresearch.com) and implements the upstream `MemoryProvider` ABC.

---

## How It Works

Limbic is modelled on the brain's limbic system:

- **Hippocampus** (encoding) — `remember()` embeds content, checks interference, computes Bayesian surprise, and stores the engram.
- **Amygdala** (salience) — trust scoring engine. All parameters self-tune through interaction signals: continuation, correction, retrieval frequency, temporal decay.
- **Thalamus** (retrieval) — `recall()` and `prefetch()` run vector similarity search, activation gating, priming, and relevance floor filtering.
- **Synaptic Atrophy** (gradual organic removal) — unused facts decay in trust, are marked for atrophy, and physically removed after a grace period. No explicit pruning required.

The agent sees two tools (`remember`, `recall`). The operator sees one file (`limbic.db`). Everything inside is self-organising.

**Key mechanisms:**
- **Activation maturation** — new facts start silent, mature via sigmoid over time + retrieval
- **Interference-based forgetting** — retroactive (new supersedes old) and proactive (old blocks new)
- **Bayesian surprise** — novel facts get a trust boost
- **Priming-based reconsolidation** — corrections propagate through trust dynamics, no content ever modified
- **Synaptic atrophy** — mark-and-sweep gradual removal of unused facts. Virtual decay (query-time) + physical removal (sweep)
- **Adaptive parameters** — self-tuning relevance floor, decay half-life, and maturation curve
- **Database triggers** — auto-maturity handled atomically by a SQLite trigger
- **ABC compliance** — inherits `MemoryProvider` ABC from Hermes core. Implements `get_config_schema`, `save_config`, `system_prompt_block`, `on_pre_compress`, `on_memory_write`.

**Research foundations:** [HippoRAG](https://arxiv.org/abs/2405.14831) (NeurIPS 2024), Human-Inspired Memory Architecture (Microsoft 2026), StructMemEval, HEMA.

---

## Storage Architecture

Limbic uses **SQLite** as its storage backend:

- **`limbic.db`** — single-file SQLite database with WAL mode
- **sqlite-vec** — vector KNN similarity search (cosine distance). Brute-force linear scan, no ANN index; practical to ~10k facts per profile.
- **FTS5** — built-in full-text search for BM25 hybrid retrieval
- **SQLite triggers** — auto-maturity fires atomically on write
- **WAL mode** — durable writes, survives ungraceful restarts, concurrent reader access

---

## Quick Start

```bash
pip install sqlite-vec onnxruntime tokenizers pyyaml spacy
```

The plugin lives at `~/.hermes/plugins/limbic/`. Hermes loads it automatically from `plugin.yaml`.

**Choose your languages** — entity extraction runs spaCy NER per configured language (default: English only):

```bash
hermes limbic languages --set en,fr,el   # persists to limbic.yaml; models download on first use
```

**First-use downloads** (~1.1GB total, cached locally, one-time):
- Embedding model — Arctic Embed 2.0 L int8 (~570MB; Snowflake's own quantized ONNX, retrieval-equivalent to fp32 — see Model section)
- NLI model — MiniLMv2-L6-mnli-xnli (~430MB)
- spaCy NER models — `en_core_web_sm` + `fr_core_news_sm` (~14.5MB each)

**Minimal config** (`~/.hermes/profiles/<name>/limbic.yaml`):

```yaml
storage:
  provider: sqlite
embedding:
  provider: local
  model: Snowflake/snowflake-arctic-embed-l-v2.0
  dims: 1024
```

**Disable Hermes' built-in memory** — otherwise the built-in MEMORY.md store runs alongside Limbic and the agent gets two competing memory systems:

```yaml
# ~/.hermes/profiles/<name>/config.yaml  (Hermes config, not limbic.yaml)
memory:
  provider: limbic            # activates Limbic as the memory provider
  memory_enabled: false       # disables the built-in MEMORY.md store
  user_profile_enabled: false # disables the built-in USER.md store
```

The `memory:` block lives in the profile's `config.yaml`; `provider: limbic` activates Limbic but does not by itself disable the built-in store — `memory_enabled: false` does.

**First use** — the agent calls two tools:

- `remember(content)` — store a fact
- `recall(query)` — search memory

No indexing step, no schema setup, no server to start.

---

## Configuration

Each profile has a `limbic.yaml` that sets environmental bounds. The system adapts within these limits:

```yaml
storage:
  provider: sqlite

embedding:
  provider: local              # bundled ONNX model — no endpoint, no external server
  model: Snowflake/snowflake-arctic-embed-l-v2.0
  dims: 1024

nlp:
  languages: [en]               # spaCy models for entity extraction (en, fr, el, …)

context_budget: 20              # ceiling on facts surfaced per turn
fragment_gate_chars: 60         # remember() rejects content below BOTH fragment gates
fragment_gate_words: 3
```

Adaptive parameters (decay half-life, relevance floor) are not static knobs — the trust engine tunes them in `limbic_params.json`.

If recall is poor, change the environment:
- Too much noise? Reduce `context_budget` (smaller pot)
- Facts fading too fast? The decay half-life is adaptive — feed the system steady retrieval traffic on the facts that matter
- System not learning? Feed it more experiences (more water)

After changing embedding dimensions — or upgrading the plugin when the bundled
embedder has changed — reindex existing facts. The store records which embedder
produced its vectors (the embedder stamp); after an embedder-changing update,
`stats` reports `stale_embedder: true` and retrieval quality is degraded until
reindex is run (run on the host where Hermes is installed — the script imports Hermes core):

```bash
python3 ~/.hermes/plugins/limbic/reindex.py --agent <profile-name>
python3 ~/.hermes/plugins/limbic/reindex.py --all          # every profile
```

---

## Multi-User Mode

Limbic scopes memories per user automatically. Resolution order for every
`user_id` the host passes in:

1. **`users.yaml` present** → multi-user mode: canonical buckets, guardian
   semantics (a guardian may write into a ward's scope; others may not),
   display names in injected context
2. **No manifest, one user seen** → single-user mode: all memories attribute
   to that raw platform ID (scoping still correct — one bucket)
3. **No manifest, multiple users seen** → each user STILL gets an isolated
   bucket under their raw ID (full MXID or platform handle — never stripped,
   `@alice:hs1` ≠ `@alice42:evil.org`), with a log warning that multi-user
   mode is misconfigured

Enable multi-user mode by copying the shipped example next to the plugin code
(or set `users_manifest:` in `limbic.yaml` to any path):

```bash
cp ~/.hermes/plugins/limbic/users.yaml.example \
   ~/.hermes/plugins/limbic/users.yaml
```

```yaml
# users.yaml — identity only; limbic reads display_name, scope,
# platform_ids, guardians. Roles are ignored here (see access-control).
users:
  alice:
    display_name: Alice
    scope: adult
    platform_ids:
      matrix: "@alice:example.org"
      telegram: "123456789"
    guardians: []
  bob:
    display_name: Bob
    scope: child
    guardians: [alice]
    platform_ids:
      matrix: "@bob:example.org"
```

The file is deployment-private (gitignored); `users.yaml.example` ships in
its place. Per-profile overrides: `$HERMES_HOME/users.yaml` beats the plugin
directory copy. Other plugins may read the same manifest for their own
identity needs (workspace delivery routing, credential prefixes) — it is the
one identity file in the stack.

---

## Operator Functions

---

## Operator Functions

```
forget(user_id)          — right to be forgotten
export(user_id)          — GDPR-style data export
seed(user_id, content)   — inject facts without conversation
reindex(dims?)            — re-embed all facts with the bundled model
stats(user_id?)          — per-user or aggregate store metrics
status(user_id?)         — atrophy candidates, storage health
```

---

## Philosophy

Memory in an agent should behave like a plant, not a database. You water it (feed experiences), prune it (correct mistakes), and let it grow toward the light (what gets retrieved, survives). The system doesn't need a schema, a migration plan, or a human tuning retrieval weights — it discovers what matters through use.

---

## License

PolyForm Noncommercial 1.0.0. Free for personal, research, educational, and noncommercial use. Commercial use requires a separate license from the author.

---

## Architecture

See [SPEC.md](./SPEC.md) for the full architecture — data model, lifecycle mechanisms, trust adaptation, retrieval pipeline, atrophy, and roadmap.
