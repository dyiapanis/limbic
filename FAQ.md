# Limbic FAQ

Answers to common follow-up questions after installing Limbic. For install steps see [README Quick Start](README.md#quick-start); for technical detail see [SPEC.md](SPEC.md).

---

## What problem does it solve?

LLM agents forget. Limbic gives an agent local, self-organising long-term memory: facts are embedded, scored for trust/relevance, and re-surfaced automatically in later sessions. No cloud, no API keys, no external vector DB — SQLite + two bundled ONNX models.

## How does it solve it?

The agent gets two tools (`remember`/`recall`). Writes pass through a trust engine (corrections demote stale claims), write-time NLI contradiction detection, and entity linking. Reads are hybrid keyword+vector search with time decay. See SPEC.md §Hippocampus/§Thalamus for mechanics.

## What is the impact of installing?

One directory (~700 MB, mostly models, downloaded lazily on first use) + one SQLite DB. CPU-only inference (int8 ONNX) — no GPU required.

## Does it allow for "dreaming"?

No offline dream phase. NLI runs at write time (inside `remember()`); every turn applies decay, atrophy sweeps and re-checks. Content is never modified after storage — nothing is rewritten while you sleep. The closest analogue is metabolism between conversations: time-based change with no agent present. See SPEC.md §Reconsolidation.

## How does it play with other memory modules?

Hermes loads ONE memory provider at a time (`memory.provider`). Limbic replaces the default store; it doesn't co-exist with another provider in the same profile.

## Performance requirements?

Runs on a minimalist 2-core, non-GPU server: embedding is int8 ONNX on CPU, NLI gate only spins at write time, storage is embedded SQLite. See SPEC.md §Performance budget / §Scale projections for measured numbers.

## What are the dependencies for?

- **onnxruntime** — local model inference (embedder + NLI)
- **tokenizers/sentencepiece** — tokenizer runtime
- **numpy** — vector math
- **sqlite3** — storage (stdlib)
- **spacy** — entity extraction (optional, per-language models via `hermes limbic setup`)

No network calls after the one-time model download; models are sha-pinned.

## Are embeddings/NLI available via OpenRouter?

No — both models run locally as bundled ONNX (offline, no API). No remote option by design.

## Telegram group with topics — how to keep memories separate?

Memory is per Hermes profile. For 6 people sharing one group:

- **One profile + `users.yaml`** — Limbic recognises the user and scopes memories (e.g. "Alice likes tacos"). Soft separation: family-doctor model — knows all members, doesn't volunteer others' info normally.
- **Separate profiles** — hard isolation, each agent has its own DB. Use if no leakage is acceptable.

Only bots *mentioned* in a topic saving memories is controlled by wake-words/platform filtering in Hermes, not limbic-side — Limbic stores what the profile's agent is given.

## Where do files go?

- Plugin code: `~/.hermes/plugins/limbic/`
- DB: `~/.hermes/profiles/<name>/limbic.db`
- Models cache: `~/.cache/huggingface/` + `~/.cache/limbic/` (or profile cache)
- Config: `~/.hermes/profiles/<name>/config.yaml` + `limbic.yaml`

No `pip install` outside Hermes — plugin manager creates its own venv and installs `pip_dependencies` there. `hermes limbic setup` downloads models with progress; no manual venv.

## What happens during `hermes update`?

Plugin code updates via `hermes plugin update`; DB is untouched. If a new version changes the embedder, `status` shows `stale_embedder: true` — run `hermes limbic reindex` once.

## How is it backed up?

DB is a single SQLite file. `hermes backup` includes `~/.hermes/profiles/<name>/limbic.db` and `limbic.yaml`. Restore by copying back; no scripts needed. Model cache is re-downloadable.

## Example multi-session walkthrough?

TBD — concrete fleet walkthrough on roadmap (six-agent use tracked in #17).
