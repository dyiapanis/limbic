# Limbic — Opaque Memory System Specification

**Version 0.6.2** · Author: D Yiapanis · License: PolyForm Noncommercial 1.0.0

## Overview

Limbic is a self-organising memory system for AI agents. It learns what matters, forgets what doesn't, and surfaces relevant context automatically. The agent doesn't manage memory — it experiences it.

The system is modelled on the brain's limbic system. Each component maps to a brain region. The agent interacts with two simple tools. The operator backs up one file. Everything inside is self-organising.

### Design Principles

- **Opaque** — the agent and operator don't inspect or manipulate internal state. The system self-organises through experience.
- **Self-improving** — adaptive parameters discover optimal values through interaction signals, not manual tuning.
- **Self-contained** — ships with two bundled ONNX models (embedding + NLI). No external services required. No endpoint configuration. Models download from HuggingFace on first use and cache locally.
- **User-scoped** — user_id filtering enforced at code level on all reads and writes.
- **Identity-agnostic** — consumes a user_id string and optional user_profile dict from the host. No dependency on any specific identity plugin.
- **Storage-agnostic** — operates against a storage interface (`LimbicStorage` ABC). SQLite is the storage backend. The brain (hippocampus, amygdala, thalamus) is unchanged across storage implementations.
- **Zero LLM cost** — no extraction on writes, no reasoning on reads. All lifecycle mechanisms are deterministic algorithms operating on embeddings and scores.
- **Token-conscious** — every fact in the context block earns its place. Nothing is free-riding. Empty context is valid.

---

## Research Foundations

Limbic's design is informed by the following research:

**HippoRAG (NeurIPS 2024)** — Hippocampal indexing theory. Knowledge graph + PageRank as the hippocampal index. Single-step retrieval matches iterative retrieval at 10-30x lower cost. Informs: the graph as an index, not just storage. Entity graph is active — spaCy NER extracts entities on every write, graph traversal is wired into prefetch retrieval alongside vector and FTS5 paths.

**Human-Inspired Memory Architecture (Microsoft, May 2026)** — Six cognitive mechanisms: sleep-phase consolidation, interference-based forgetting, engram maturation, reconsolidation upon retrieval, entity knowledge graphs, hybrid multi-cue retrieval. Key findings: deduplication (not summarisation) is the dominant mechanism; decay half-life ~29 days optimal for production; maturation sigmoid (silent → retrievable over ~1 week); reconsolidation window ~60 min; interference-based forgetting (retroactive 0.6, proactive 0.4); Bayesian surprise as importance signal. Informs: reconsolidation, activation maturation, interference forgetting, Bayesian surprise.

**HEMA (April 2025)** — Dual-memory: compact summary + vector episodic store. Semantic forgetting through age-weighted pruning reduces latency 34%. Informs: temporal decay as pruning mechanism.

**StructMemEval (arxiv 2502.13649)** — Simple flat retrieval outperforms complex memory hierarchies on standard benchmarks. Entity traversal benefits emerge at >1,000 facts. Informs: MVP shipped with flat + entity graph (three-path hybrid: vector + FTS5 + graph traversal). Graph path runs alongside flat retrieval from day one — benefits scale naturally as fact count grows past 1,000.

**A-Mem (arxiv 2502.12110)** — Zettelkasten-style linked notes. 85-93% token reduction. Informs: linked-notes architecture if flat retrieval insufficient (backlog).

**Cognee** — Six-stage pipeline with semantic triples and self-improving knowledge graph. Informs: LLM extraction and graph refinement (backlog).

**Steve Kinney / Zylos surveys (2026)** — Three-axis framework (Forms, Functions, Dynamics). Production consensus: hybrid vector-graph stores becoming standard. Multi-tenancy must be designed from start. Informs: user scoping from day one, hybrid retrieval as target architecture.

---

## Architectural Reference — The Limbic System

The system is opaque like a brain. You can't inspect your own neurons, but you can tend to the environment — sleep, nutrition, activity — and the brain self-organises. Limbic works the same way. You feed it experiences through conversation. You set environmental conditions through config. The system grows.

**Structure: four regions, seven mechanisms.** Every mechanism is owned by exactly one region. Maturation, reconsolidation, and atrophy are *processes*, not structures — they are listed under the region that performs them, never as peers of a region. Prompt injection is not a mechanism; it is what the thalamus produces.

The authoritative mapping — region → process → code symbol — is `limbic/docs/anatomy-map.md`, verified against the plugin source by `limbic/docs/verify_anatomy_map.py`. The sections below are the narrative; where they and the map disagree, the map wins.

### Hippocampus — Encoding

**Brain function:** Converts experience into memory.

**Limbic component:** The write path. `remember(content)` receives raw text. Internally:
- Embedding generated via bundled ONNX model (Arctic Embed 2.0 L, 1024 dims)
- user_id resolved from session context (not passed as parameter)
- Guardian case: if content is about another user (e.g. "Bob is interested in Harry Potter"), fact is stored under that user, not the speaker
- **NLI-aware interference and correction detection** — for each similar existing fact (similarity > 0.60), an NLI cross-encoder classifies the relationship as contradiction, entailment, or neutral. The label determines the action (see NLI-aware flow below).
- Bayesian surprise computed — novel facts get trust boost
- Stored in SQLite with embedding, content, user_id, trust, activation, timestamps
- **Named entities extracted via spaCy NER** — EntityRuler with custom fleet patterns (user names, project names, products) layered on top of statistical NER. Entities stored in `entities` table, fact↔entity links in `mentions` junction table. This is the hippocampal spatial map — each fact is located relative to the entities it mentions.
- Initial trust: 0.5 (or 0.6 if high surprise)
- Initial activation: 0.03 (silent engram)

#### NLI-aware interference and correction flow

For each similar existing fact, the NLI cross-encoder (MiniLMv2-L6-mnli-xnli, 100+ languages) classifies the relationship between the existing fact's content and the new content. The cross-encoder is noisy on unrelated pairs, so it only fires above similarity 0.70 — below that, similarity-band logic alone decides:

| Similarity | NLI label | Action | Penalty |
|---|---|---|---|
| >0.85 | contradiction | Store correction + penalise old | correction_penalty (0.15) |
| >0.85 | entailment | Store new (no penalty) | 0 |
| >0.85 | neutral | Store new + interference penalty on old | interference_penalty (0.20) |
| 0.70-0.85 | contradiction | Store correction + penalise old | correction_penalty (0.15) |
| 0.70-0.85 | entailment | Deduplicate (skip — same information) | n/a |
| 0.70-0.85 | neutral | Deduplicate (skip — same topic) | n/a |
| 0.60-0.70 | n/a (NLI not run) | Store new (no penalty) | 0 |
| <0.60 | n/a | Store new (no action) | 0 |

**Contradictions are never deduplicated** — the correction must be stored so it can mature and replace the stale fact through trust dynamics. This is the reconsolidation trigger: NLI computes prediction error (the existing fact predicted one state, the new fact observes another), which opens the reconsolidation window.

**No fact is exempt from contradiction.** Every fact, however durable, takes the correction penalty when NLI detects a contradiction. There is deliberately no exemption class — the measured outcome that ruled one out is under Amygdala.

**This runs at write time**, so it works in all session types — not just continuous gateway sessions. This closes the correction propagation gap that existed when correction detection was regex-only and read-time.

**Why NLI, not embeddings or regex:** Embeddings capture topic, not stance — "I eat meat" and "I am vegan" have high cosine similarity because both are about diet. Regex catches explicit negation ("no", "wrong") but misses declarative state-changes ("The old engine is decommissioned", "The browser engine is Camofox now"). NLI (cross-encoder) processes both texts together and captures their interaction — it detects semantic contradiction regardless of linguistic form.

The agent never sees fact IDs, payloads, or internal state. It said "remember this" and the system did.

### Amygdala — Salience (Adaptive)

**Brain function:** Tags memories with emotional weight. Over time, memories that aren't reactivated fade.

**Limbic component:** Trust scoring engine. Entirely internal. No exposed parameters. Owns everything that moves trust after the write, plus maturation and the atrophy criterion.

**Mechanisms (distinct, not one bucket):**

- **Implicit signals** — continuation (user stays on topic), correction (user says "no" or "wrong"), ignore (topic change). Per turn.
- **Interference-based forgetting** — write-time, two directions. Retroactive: when a new fact is similar to an existing one (>0.85 cosine), the old fact's trust decreases. Proactive: when a mature fact blocks a newer silent fact (>0.7 cosine, newer created_at), the old fact's trust is penalised at 0.4× the interference rate. Based on Microsoft paper's interference model.
- **Bayesian surprise** — facts whose embeddings are far from the centroid of existing facts get a trust boost (0.5 → 0.6) at write time. Novel information is more memorable. Based on Microsoft paper's dopaminergic signalling model.
- **Temporal decay** — Ebbinghaus curve; the decay clock resets on retrieval, not on decay application. Decay rate adapts per domain. The operation is **idempotent**: its anchor is the most recent of `last_retrieved`, `created_at`, and the previous decay pass (`updated_at`), so repeated application over an already-charged interval adds nothing. Required because decay runs every turn — anchoring only on `last_retrieved` re-billed the full elapsed interval each pass.
- **Maturation** (a process, owned here) — silent → retrievable via the activation sigmoid: `1 / (1 + exp(-(hours - adjusted_t_half) / k))`. Retrieval shifts the curve left, so use shortens the silent period. `t_half` adapts toward observed average time-to-maturity.
- **Atrophy criterion** (a process, owned here) — what makes a fact eligible for removal. The sweep itself runs in the thalamus's maintenance step.

**Adaptive parameters** — decay rate, relevance floor, and the maturation curve drift toward what produces better recall quality. Parameters start at defaults.

Nobody configures `retrieval_boost: 0.02`. The system discovers what works.

**No exemption class.** A pinned/exempt mechanism was operated and produced 79% saturation (182 of 231 facts pinned), pin state that survived correction, and regrowth to pre-repair levels within ~30h of being zeroed. An immortal fact class is incompatible with natural retention; the measured outcome is in the paper (Section 4.4.2). There is no mechanism that skips decay or the correction penalty.

### Thalamus — Routing and Gating (Adaptive)

**Brain function:** The relay station. Routes sensory input. Decides what to load into conscious awareness. Not a dumper — a gatekeeper.

**Limbic component:** Prefetch — routing and gating before each LLM call. Entirely internal.

Three-path hybrid retrieval (cortex + semantic association + hippocampal map):

1. **Vector similarity search** (cortex — pattern matching) — sqlite-vec brute-force KNN, filtered by user_id
2. **Full-text search** (semantic association) — FTS5 BM25, keyword overlap between query and fact content
3. **Entity graph traversal** (hippocampal spatial map) — spaCy NER extracts entities from the query, graph traversal finds all facts linked to those entities via the `mentions` junction table. Graph results that aren't already in the vector/FTS candidate pool are merged in with a moderate score (0.3) so they pass the relevance floor.

Each path catches what the others miss. Results are fused via Reciprocal Rank Fusion (RRF) for vector + FTS, then graph results are merged in. The composite score is computed across all candidates.

- **Over-fetch** — search requests `limit * 5` results so the activation gate has a larger pool to filter. Without this, a crowd of recent silent facts can dominate the top-K and starve recall of active facts.
- **Relevance floor** — facts below a composite score threshold don't inject. The floor adapts: if injected facts lead to corrections or topic changes, the floor rises. If continuation rate is high but context is sparse, the floor lowers.
- **`context_budget` is a ceiling, not a target.** Some turns produce 3 facts. Some produce 1. Some produce 0.
- **Silent-facts fallback** — if no active facts pass the relevance gate, high-relevance silent facts (activation < 0.5 but embedding score >= floor) are returned as fallback so recall isn't empty when all matching facts are still maturing.
- What surfaces is the system's best guess at what's relevant, ranked by signals the agent never sees

### Entorhinal Cortex — The Gateway (Spatial Coordinate System)

**Brain function:** Sits between the hippocampus and the cortex. Grid cells create a coordinate system — it's how the brain knows that two memories encoded at different times are in the same neighbourhood of memory space.

**Limbic component:** The entity graph's cross-references. Facts linked to "alice" and facts linked to "marathon" are connected through their shared entity — the intersection of two entity regions. The entorhinal cortex doesn't store content; it stores the coordinate relationships that make navigation possible.

When the thalamus routes a query, the entity graph path traverses these coordinates. `search_graph(["alice", "marathon"])` activates the intersection of two regions — facts linked to both entities. This is structural navigation, not fuzzy similarity. A fact linked to "alice" is exactly about Alice, not about someone with a similar name.

The entity graph tables (`entities` + `mentions`) are the coordinate system. Each entity is a coordinate. Each fact-entity link places a fact at that coordinate. Multiple links place a fact at the intersection of coordinates. This is how the brain knows "marathon training" and "the 2026 race" are in the same neighbourhood, even if encoded at different times — they share coordinates.

#### Maturation — Engram Development  ·  owned by Amygdala

*Not a region. A process: the activation sigmoid that moves a fact from silent to retrievable.*

**Brain function:** Engrams form immediately but remain "silent" before becoming explicitly retrievable.

**Limbic component:** Activation maturation. Entirely internal.

- New memories start silent (activation ~0.03)
- They prime related memories without surfacing themselves
- Over time + retrieval, they mature via sigmoid
- Retrieval accelerates maturation — each retrieval shifts the curve left
- Below threshold (activation < 0.5): silent, not injected. Above threshold: eligible for injection.
- Maturation curve adapts — t_half adjusts toward observed average time-to-maturity, recorded when facts first cross activation 0.5

Based on Kitamura et al. (2017) engram stabilisation research and Microsoft paper's maturation dynamics.

#### Reconsolidation — Organic Correction  ·  owned by Hippocampus

*Not a region. A process: write-time NLI contradiction detection, firing inside `remember()`.*

**Brain function:** Retrieved memories become labile — modifiable for a window. New information during this window blends with existing content. Prevents stale facts.

**Limbic component:** Corrections are detected at write time by NLI and act through trust dynamics. Entirely internal. **No content is ever modified.**

The system never edits a fact's content. It never blends, merges, appends, or replaces text. Instead, corrections work through trust and activation dynamics — the same mechanisms that govern everything else.

**How it works:**

Correction detection operates through two channels:

1. **Write-time NLI (primary):** When a new fact is stored, the NLI cross-encoder classifies the relationship between the new content and each similar existing fact (similarity > 0.60). If NLI returns "contradiction", the existing fact's trust is penalised immediately at write time. This works in all session types — continuous gateway sessions, single-shot dispatches, cron jobs, delegated tasks.

2. **Read-time regex (secondary):** During prefetch, the `_evaluate_trust_signals()` method checks the user's message for correction patterns ("no", "wrong", "actually") against previously injected facts. If a correction is detected and the user's message overlaps with an injected fact's content, the fact's trust is penalised. This provides a secondary signal in continuous gateway sessions where the user explicitly corrects the agent.

Once a correction is detected (through either channel):

1. A correction is stored as a new silent fact (activation ~0.03) via `remember()`
2. A **contradiction link** is recorded between the new fact and the contradicted one
3. The mature fact's trust has already been penalised at write time (NLI) or at read time (regex)
4. On each tick, the link is re-checked: if the mature fact's trust has recovered,
   the penalty is re-asserted; if it has fallen below the relevance floor, the link
   resolves — the correction has landed
5. The correction fact matures over time and begins surfacing
6. The system has self-corrected — old fact faded, new fact grew. No one touched the content.

#### The continuing-correction channel

Write-time NLI is **one-shot**: it penalises the older fact once and stops. If that
trust later recovers through continuation signals, the correction is silently undone,
and if NLI was unavailable at write time the contradiction is never noticed at all.

A content-free correction needs a channel that persists after the write. That channel
is an explicit **contradiction link** between the fact pair, stored in the
`contradictions` junction table (reusing the `mentions` pattern — composite PK,
`ON DELETE CASCADE` on both sides).

The re-check runs on the periodic tick, bounded by `max_corrections_per_turn` (default 3):

| Condition | Action |
|---|---|
| Older fact's trust below the relevance floor | resolve — the correction landed |
| NLI no longer returns `contradiction` | resolve — no longer true |
| Trust recovered **and** NLI still contradicts | re-assert the penalty (one write) |

Links expire after `correction_link_ttl_days` (default 30) if neither condition fires.

**Bounds:** at most `max_corrections_per_turn` trust writes and NLI calls per turn —
O(1) rather than the O(silent facts × neighbours) of the read-time channel it replaces.
Both terminal conditions close the link, so the pass is self-limiting.

**This is not read-time priming.** No silent fact scans for neighbours, no composite-score
boosts. The link is directional and explicit, and it records what write-time NLI already
decided rather than performing a second detection pass.

**Why this is the right shape.** In the brain, reconsolidation happens when a memory is reactivated and new information is present. You don't open the brain and rewrite the engram — the old memory becomes less salient, the new one becomes more salient. Limbic does the same, but the correction is detected at write time (NLI) rather than during a reactivation window. Trust and activation dynamics carry it from there.

There is no labile-window timestamp and no content editing of any kind — no blending, no number replacement, no appended "Updated:" suffixes. Write-time NLI is the trigger; trust and activation dynamics do the rest.

**The correction penalty and `is_correction()` detection already exist in the trust engine.** NLI is what connects them to the right fact: the cross-encoder identifies which existing fact the new content contradicts, and the penalty is applied there. One mechanism, organic, no content manipulation.

Based on Nader et al. (2000) reconsolidation research — reinterpreted through the organic philosophy: the system adjusts salience, not content.

#### Synaptic Atrophy — Gradual Organic Removal  ·  criterion Amygdala, sweep Thalamus

*Not a region. A process: the amygdala decides eligibility, the thalamus's maintenance step sweeps in bounded batches.*

**Brain function:** Unused synapses weaken and are pruned. Not a conscious decision — a background metabolic process. Neurons that don't fire, don't wire.

**Limbic component:** Mark-and-sweep atrophy. Entirely internal. No exposed parameters.

Unlike the virtual forgetting already implemented (temporal decay, interference, reconsolidation), atrophy **physically removes** facts from the database. But it does so organically:

**Mark phase:** During `prefetch()`, after applying temporal decay to every retrieved fact, any fact meeting these criteria is marked:
- `trust_score < atrophy_threshold` (default 0.01 — trust-only, near zero)

The horizon check (`atrophy_horizon_multiplier × decay_half_life_days`) is disabled by default (multiplier = 0). Trust alone determines atrophy — a fact's lifetime through the 30-day decay curve IS its grace period. If trust has decayed to near-zero, the fact has had its entire lifetime to prove relevant and didn't.

For never-retrieved facts (orphans), `created_at` is used as the reference timestamp for decay, ensuring orphans still decay through time alone.

Marking sets `atrophy_marked_at = now()` on the record.

**Sweep phase:** During `tick()` (every turn), `_sweep_atrophy()` checks for facts whose `atrophy_marked_at` exceeds `atrophy_grace_days` (default 0 — no separate grace; the fact's lifetime through the decay curve is the grace period). It deletes them in bounded batches (`max_atrophy_per_turn` = 10).

A background scan (`_scan_and_sweep_gated_facts()`) also runs every turn, querying the DB directly for facts below the atrophy threshold that haven't surfaced in search results. This catches facts that are too irrelevant to even appear in prefetch candidates.

**Why no separate grace period:** The fact's lifetime — from creation, through the 30-day Ebbinghaus decay curve, until trust drops below the atrophy threshold — IS the grace period. It's not a parameter. Setting `atrophy_grace_days = 0` is correct — there's no need for an additional wait after the fact has already had its entire lifetime to prove itself.

**All parameters are adaptive:**
- `atrophy_threshold` — if too many facts are marked, the threshold rises; if the store grows unbounded, it lowers
- `atrophy_horizon_multiplier` — if facts are removed too aggressively, the multiplier rises
- `atrophy_grace_days` — if operators report accidental loss, grace extends; if storage pressure is high, it shortens
- `max_atrophy_per_turn` — if compaction pressure is high, the batch size lowers

**This is not pruning.** Pruning is an explicit operator command (`hermes limbic prune`). Atrophy is a metabolic process — the system forgets unused memories the same way a brain forgets them: gradually, irreversibly, and without instruction.

#### Context Injection — What the Thalamus Produces

*Not a region and not a mechanism. Injection is the output of routing, not an eighth component.*

**Brain function:** What's loaded into active processing right now. Not all memories — just what the thalamus routed in and the amygdala flagged as relevant.

**Limbic component:** The `<limbic-context>` block injected into the system prompt.

**Two states of memory:**

1. **Ambient** — facts that exist in the system, available when needed, but not actively injected. Not recited every turn.

2. **Focused** — facts the thalamus routes in because they're relevant AND cross the relevance floor. These earn their token cost.

Nothing is "always injected." Every fact earns its place through relevance to the current turn. The distinction is "ambient vs focused," and the thalamus decides based on what you're actually talking about.

**Empty context is valid.** If nothing crosses the relevance floor, the block is empty or absent.

**Activation as injection gate:**
- High activation + high composite score → focused (injected)
- High activation + low composite score → ambient (available, not injected)
- Low activation → silent (not injected)

**Identity directive in context block:**

The prefetch pipeline prepends an identity directive to the context block. This tells the agent who they're talking to and enforces user-separation:

```
You're chatting with {display_name} ({scope}). Recalled user preferences shape your responses.
User-separation rule: You are talking to {display_name}. Do not reference, mention, or apply
other users' personal data. If recalled context or conversation history mentions another user,
do not surface it to {display_name}.
```

This is not a fact — it is a directive injected by the thalamus alongside the facts it routes in.

---

## What the Agent Sees

Two tools:

```
remember(content: str) → {"status": "stored"}
recall(query: str) → {"results": [...], "count": N, "status": "ok", "top_score": S}
recall(query: str) → {"results": [], "count": 0, "status": "no_confident_match", "top_score": S}
```

**Recall-confidence gate (abstention):** when the best fact's RELEVANCE — the raw
cosine similarity, NOT the trust-modulated composite — falls below the adaptive
`recall_confidence_floor` (default 0.40), recall reports `no_confident_match`
instead of surfacing a near-miss — read-side metacognition. The same top-relevance
guard applies to prefetch: if even the best fact can't clear the floor, nothing is
injected that turn (per-fact `relevance_floor` filtering is unchanged on top of
it). The gate is deliberately cosine-gated: composite conflates match confidence
with fact trust — natural queries about recently-stored (low-trust) facts would
falsely abstain (measured: 5/10 natural probes abstained via composite vs 0/10 via
cosine, independent review 2026-10-10). Calibration (live 429-fact store):
answerable top-cosine 0.63–0.95 vs unanswerable 0.15–0.30 — the floor sits mid-gap.
The agent should treat an abstention as "memory has nothing on this," not as
silence to paper over: answer from what you know, don't guess and don't attribute
anything to memory.

And a context block that appears before turns where memory is relevant:

```
<limbic-context>
You're chatting with alice (adult). Recalled user preferences shape your responses.
User-separation rule: You are talking to alice. Do not reference, mention, or apply
other users' personal data. If recalled context or conversation history mentions another
user, do not surface it to alice.
Project architecture discussion — evaluating opaque Limbic design.
Adaptive parameter engine — self-tuning trust and retrieval weights.
</limbic-context>
```

No categories. No trust scores. No fact IDs. Just context. Only what's relevant. Sometimes nothing.

The agent tunes the system through interaction:
- "Remember that I'm vegan" → fact stored, starts maturing
- "No, that's wrong" → correction signal, trust drops on stale fact, new fact matures to replace it
- "Actually my goal is 3:50 now" → new fact stored, primes old "sub 4hr" fact, old fact fades
- Repeatedly talking about training → facts retrieved more, trust rises, activation matures faster

---

## What the Operator Sees

```
~/.hermes/profiles/<agent>/limbic.db
~/.hermes/profiles/<agent>/limbic_params.json
```

One database file. One parameters file. Back them up. Restore them. Never open them.

### Operator functions (CLI/script only, not agent tools)

```
forget(user_id: str) → {"removed": N, "status": "complete"}
export(user_id: str) → {"facts": [...], "fact_count": N}
seed(user_id: str, content: str) → {"status": "stored"}
reindex(provider: str, **config) → {"reindexed": N, "status": "complete"}
stats(user_id: str | None) → {"total_facts": N, "avg_trust": F, "matured": N, ...}
```

- `forget` — right to be forgotten. Removes all data for that user.
- `export` — data export / debug window. GDPR-style.
- `seed` — inject facts without conversation. For migration or operator-initiated context loading.
- `reindex` — re-embed all facts with the bundled model (optionally at new dims). Creates a new `vec_index`, re-embeds, swaps atomically, drops the old index. The FTS and fact rows are untouched.
- `stats` — analytical self-evaluation. Per-user or aggregate metrics for performance monitoring.

### Config (environmental conditions)

```yaml
# Storage — SQLite embedded
storage:
  provider: sqlite

# Embedding — bundled ONNX (Arctic Embed 2.0 L, in-process, no external services)
embedding:
  provider: local
  model: Snowflake/snowflake-arctic-embed-l-v2.0
  dims: 1024

# NLI — contradiction detection (bundled ONNX, in-process)
nli:
  model: MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli

# Entity extraction — spaCy models per language
nlp:
  languages: [en]

# Environmental conditions — system adapts within these bounds
context_budget: 20           # like pot size — ceiling on facts in awareness per turn
fragment_gate_chars: 60      # conversational fragments are not durable facts
fragment_gate_words: 3

# Adaptive parameters (decay half-life, relevance floor) live in
# limbic_params.json — managed by the trust engine, not static knobs.
```

If recall is poor, you change the environment:
- Too much noise? Reduce `context_budget` (smaller pot)
- Facts fading too fast? The decay half-life is adaptive — feed the system steady retrieval traffic on the facts that matter
- System not learning? Feed it more experiences (more water)

---

## Three-Layer Decoupling

| Layer | Interface | Default | Alternatives |
|---|---|---|---|
| **Identity** | `user_id: str` + `user_profile: dict` | User manifest (users.yaml) | LDAP, OAuth, anything |
| **Storage** | `LimbicStorage` class | SQLite embedded (one file, WAL mode) | Interface exists for future backends |
| **Embedding** | `embed(content) → vector` | Bundled ONNX (Arctic Embed 2.0 L, in-process) | Interface exists for future backends |

Each layer is independently swappable. The brain operates against interfaces, not implementations.

### Storage interface

```python
class LimbicStorage:
    def store(self, content, user_id, embedding, metadata) -> int
    def search(self, embedding, filters, limit) -> list[dict]
    def get(self, fact_id) -> dict | None
    def update(self, fact_id, **fields) -> None
    def delete_user(self, user_id) -> int
    def export_user(self, user_id) -> dict
    def count(self, user_id: str | None = None) -> int
    def reindex(self, new_embedder, new_dims: int) -> dict
    # + search_fts, store_entities/search_graph, analytical rows, maintenance,
    #   and contradiction-link methods — storage.py is the full contract
    def close(self) -> None
```

### Embedding interface

```python
class LimbicEmbedder:
    def embed(self, content: str) -> Optional[list[float]]
    @property
    def dims(self) -> int
```

Two implementations:
- `OnnxEmbedder` — bundled Arctic Embed 2.0 L (ONNX, in-process, zero external services). Caches by content hash. No endpoint option.

---

## Bundled Models

The system ships with two ONNX models, both running in-process on CPU via onnxruntime. No external services. No endpoint configuration. Models download from HuggingFace on first use and cache locally.

### Embedding Model — Arctic Embed 2.0 L

**Model:** `Snowflake/snowflake-arctic-embed-l-v2.0`
- 1024 dimensions
- 74 languages (including English, French, Chinese, Arabic, Greek)
- ~570 MB model cache (int8-quantized ONNX, single self-contained file: model_int8.onnx)
- Cache hygiene: at every model load (each process start), files in the cache directory that
  are not in the pin manifest are deleted — stale orphans left by a previous model or
  quantization (e.g. a fp32 `model.onnx` + `model.onnx_data` pair from a 0.5.x install, ~2.1GB)
  are reclaimed on the first start after upgrade; the database and Limbic state are never touched
- Apache-2.0 license
- ~10-15ms inference on CPU (ONNX runtime)
- 8192 token max sequence (truncated to 512 for facts — sufficient for memory content)
- XLM-RoBERTa base architecture
- Query prefix: "query: " recommended for retrieval queries; no prefix for stored facts
- Mean pooling + L2 normalisation

**Why this model:**
- Multilingual
- 1024 dimensions — richer representations than smaller multilingual MiniLM models
- 8192 token context — effectively unlimited for fact-length content
- Apache-2.0 — no licensing concerns for distribution
- XLM-RoBERTa base — same architecture family as the NLI model
- Pre-built ONNX export available on HuggingFace — no conversion needed

**Switching embedding models:**
- Requires reindex: `python3 reindex.py --agent <name>` or `python3 reindex.py --all`
- Reindex drops the old vec_index, re-embeds all facts through the new model, creates a new vec_index atomically
- One command, automated, documented

### NLI Model — MiniLMv2-L6-mnli-xnli

**Model:** `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`
- 3-class output: entailment, neutral, contradiction
- 100+ languages (including Greek, English, French, Chinese, Arabic)
- ~430 MB model cache (ONNX), downloaded on first contradiction-check — not at install or startup
  (the provider holds only a lightweight classifier shell until a write with similarity > 0.70
  triggers the first classification; the same non-pinned-file GC policy as the embedder applies)
- MIT license
- ~3-5ms per pair on CPU (ONNX runtime)
- 107M parameters (distilled from XLM-RoBERTa-large)
- XLM-RoBERTa base architecture (same family as embedding model)
- Trained on XNLI (15 languages) + MNLI
- 87% accuracy on MNLI mismatched

**Why this model:**
- Same architecture family as the embedding model (XLM-RoBERTa) — shared tokenizer vocabulary
- Small enough to bundle (430 MB vs 560 MB for mDeBERTa-v3-base)
- Fast enough for write-time inference (~15-25ms for 5 NLI calls)
- 100+ languages — matches the embedding model's coverage
- MIT license — no licensing concerns

**Why NLI, not embeddings or regex for contradiction detection:**
- Embeddings (bi-encoder) capture topic, not stance — "I eat meat" and "I am vegan" have high cosine similarity
- Regex catches explicit negation but misses declarative state-changes
- NLI (cross-encoder) processes both texts together and captures their interaction — detects semantic contradiction regardless of linguistic form

---

## Privacy

- **User scoping** — enforced at code level (WHERE clause on every retrieval). Verified by code review, not DB inspection.
- **Right to be forgotten** — `forget(user_id)` removes all data for that user. Operator function.
- **Data export** — `export(user_id)` returns all stored data. GDPR-style.
- **Backup** — the database file contains all users' data. Backup is all-or-nothing per agent. Selective per-user backup via `export`.
- **No cloud data exfiltration** — bundled model runs in-process; no endpoint option exists.
- **Guardian access** — guardians (e.g. a parent) can see their own facts plus their guarded users' facts. Enforced via `user_id_any` filter in queries.

---

## User Scoping (Internal)

- user_id resolved from session context by the host identity system
- Identity resolution chain: user manifest (users.yaml — canonical name, scope, guardian_of) → Matrix homeserver suffix stripping → raw user_id
- All writes tagged with current user_id automatically (database field)
- All reads filtered by current user_id automatically (WHERE clause)
- Guardian case: searches use `user_id_any: [user_id] + list(guardian_of)` so a guardian sees both their own and their dependents' facts
- Subject detection: if a fact's content starts with another user's name as subject (e.g. "Bob is interested in..."), it is stored under that user, not the speaker

---

## Baseline Seeding (Genetic Predisposition)

On first interaction with a new user_id:
- Host passes user_profile dict to Limbic's `initialize()`
- Limbic ingests as initial memories (identity-level facts: canonical name, display name, scope, role, email)
- On subsequent sessions, existing memories → no re-seeding

The identity system provides the genetic baseline. The system grows from there through experience.

---

## Backup and Restore

Backup paths, in order of preference:
- **Host backup pipeline** — `backup_paths()` hands the database path to Hermes backup; the store's `backup()` uses `VACUUM INTO`, consistent while the gateway is writing.
- **Per-user selective backup** — `hermes limbic export --user <uid>` (JSON, offline-safe).
- **Cold copy** — `cp limbic.db limb_params.json` only while the gateway for that profile is stopped. Raw-copying a live WAL database yields a torn image.

Restore: copy the two files back with the gateway stopped. No rebuild needed — one database file plus one parameters file contains all vectors, content, metadata, and adaptive parameters.

---

## Token Cost Model

Every fact in the context block costs tokens. The system earns that cost.

| Scenario | Facts injected | Est. tokens | When |
|---|---|---|---|
| Deep domain conversation | 3-5 | 80-150 | Relevant facts cross the floor |
| Casual check-in | 1-2 | 30-60 | Minimal context needed |
| Self-contained question | 0 | 0 | Conversation doesn't need memory |
| Topic shift | 3-5 | 80-150 | New domain enters focus, old goes ambient |

Compare to fixed injection systems: ~1200-1500 tokens per turn regardless of relevance.

---

## Tuning Philosophy

You can't micromanage a plant's DNA, composition, or growth. But you can water it, tend to it, affect its environment. The plant does the rest.

You can't micromanage your brain's neurons. But sleep consolidates memories, nutrition affects neurogenesis, exercise increases BDNF. You tend to the environment and the brain self-organises.

Limbic is the same. You feed it experiences (water). You set environmental conditions (sunlight, soil). The system grows. You don't inspect the roots. You don't edit the DNA. You tend to the environment and observe what grows.

---

## Technical Details

### 1. Hippocampus — Encoding (SQLite Schema)

Each fact is a row in the `facts` table:

```python
{
    "fact_id": 42,               # INTEGER PRIMARY KEY AUTOINCREMENT
    "embedding": [1024 floats],  # BLOB — float32 vector stored via sqlite-vec (bundled model dims)
    "content": "City Marathon 2026 — sub 4hr goal, race day Aug 30",
    "user_id": "alice",
    "trust_score": 0.89,         # decayed value (apply_decay writes this every turn)
    "retrieval_count": 3,
    "last_retrieved": "2026-07-14T22:00:00Z",
    "matured_at": "2026-07-08T10:00:00Z",  # set by trigger when retrieval_count crosses 5
    "atrophy_marked_at": null,          # set when fact enters atrophy pipeline
    "created_at": "2026-07-01T10:00:00Z",
    "updated_at": "2026-07-14T22:00:00Z"
}
```

Note: `activation` is NOT a stored column — it is always computed dynamically via `compute_activation(created_at, retrieval_count)` at query time.

SQLite configuration:
- Vector dimensions: 1024 (bundled model) — set at init, changes require reindex
- Distance: Cosine (via sqlite-vec vec0 virtual table with `distance_metric=cosine`)
- Vector table: `CREATE VIRTUAL TABLE vec_index USING vec0(facts.embedding FLOAT[1024] distance_metric=cosine)` — persistent. **No ANN index:** `vec0` is queried as a linear scan. Practical to ~10k facts per profile; an ANN index is future work in sqlite-vec.
- KNN search: `SELECT fact_id, distance FROM vec_index WHERE embedding MATCH ? ORDER BY distance LIMIT ?`
- Distance returned by KNN is converted to score: `score = 1.0 - distance` (distance 0 = identical, score 1.0)
- Full-text search: FTS5 virtual table `facts_fts` synced via triggers for BM25 hybrid retrieval
- WAL mode: durable writes, concurrent reader access, survives ungraceful restarts
- User scoping: `WHERE user_id = ?` on every query (indexed column)

**Entity graph tables (entorhinal cortex — hippocampal spatial map):**

```sql
-- Entity registry — each named entity (person, org, product, etc.)
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,        -- md5(name.lower())[:12]
    name TEXT NOT NULL UNIQUE,
    type TEXT DEFAULT 'UNKNOWN'         -- spaCy NER label: PERSON, ORG, GPE, PRODUCT, etc.
);

-- Fact ↔ Entity junction — which facts mention which entities
CREATE TABLE IF NOT EXISTS mentions (
    fact_id INTEGER NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (fact_id, entity_id),
    FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

-- Contradiction links — the continuing-correction channel.
-- One row per NLI-detected contradiction. Re-checked on the periodic tick so a
-- correction whose penalty was outvoted by later continuations gets re-asserted.
CREATE TABLE IF NOT EXISTS contradictions (
    newer_fact_id INTEGER NOT NULL,     -- the fact carrying the correction
    older_fact_id INTEGER NOT NULL,     -- the fact being corrected
    created_at    TEXT NOT NULL,
    nli_label     TEXT NOT NULL,        -- 'contradiction' at write time
    resolved_at   TEXT,                 -- NULL while the link is live
    PRIMARY KEY (newer_fact_id, older_fact_id),
    FOREIGN KEY (newer_fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
    FOREIGN KEY (older_fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE
);
```

Entity extraction runs on every `remember()` call via spaCy NER. Language pipelines are config-driven (`nlp.languages`, default `en_core_web_sm`; spaCy models download on first use for each configured language). A custom EntityRuler adds deterministic patterns for known entities (user names, project names, products) that statistical NER misses. Graph traversal in `search_graph()` finds all fact_ids linked to query-extracted entities, merged into the retrieval candidate pool alongside vector and FTS5 results.

Adaptive parameters stored as a local JSON file alongside the database (`limbic_params.json`):

```python
{
    "trust_weight":         {"value": 0.20, "min": 0.05, "max": 0.40},
    # Continuing correction — bounded re-check of open contradiction links
    "max_corrections_per_turn": {"value": 3,  "min": 1,  "max": 10},
    "correction_link_ttl_days": {"value": 30, "min": 7,  "max": 90},
    "decay_half_life_days": {"value": 30,   "min": 7,    "max": 90},
    "retrieval_boost":      {"value": 0.02, "min": 0.005, "max": 0.05},
    "continuation_boost":   {"value": 0.03, "min": 0.01, "max": 0.08},
    "correction_penalty":   {"value": 0.15, "min": 0.01, "max": 0.20},
    "maturation_t_half":    {"value": 168,  "min": 24,    "max": 336},  # hours
    "maturation_k":         {"value": 48,   "min": 12,    "max": 96},
    "relevance_floor":      {"value": 0.25, "min": 0.15, "max": 0.60},
    "interference_penalty": {"value": 0.20, "min": 0.02, "max": 0.25},
    "surprise_boost":       {"value": 0.10, "min": 0.02, "max": 0.20},
    "surprise_percentile":  {"value": 0.90, "min": 0.80, "max": 0.99},
    # Atrophy (gradual organic removal)
    "atrophy_threshold":           {"value": 0.01, "min": 0.001, "max": 0.20},
    "atrophy_horizon_multiplier":  {"value": 0,    "min": 0,    "max": 10},   # 0 = no horizon check, trust-only
    "atrophy_grace_days":          {"value": 0,    "min": 0,    "max": 30},   # 0 = no separate grace period; the decay curve is the grace
    "max_atrophy_per_turn":        {"value": 10,   "min": 1,    "max": 20},
}
```

### 2. Amygdala — Salience (Adaptive Parameter Engine)

#### Adaptation mechanism

Every N turns (default N=50), the system evaluates accumulated continuation/correction signals:

**Relevance floor adaptation:**
```
continuation_rate = fraction of injected facts where user continued on topic
correction_rate = fraction where user corrected
ignore_rate = 1.0 - continuation_rate - correction_rate

if ignore_rate > 0.5: relevance_floor += 0.02
if continuation_rate > 0.7 and total < 3 * interval: relevance_floor -= 0.01
clamp to bounds
```

**Decay rate adaptation:**
```
if correction_rate > 0.3:
    # High correction rate — stale facts, speed up decay
    decay_half_life_days -= 2 (min 7)
elif continuation_rate > 0.7:
    # High continuation rate — good facts, slow down decay
    decay_half_life_days += 2 (max 90)
clamp to bounds
```

**Adapted parameters (3):** relevance_floor, decay_half_life_days, maturation_t_half (moves ±12h toward the observed average maturity time, driven by `record_maturity()` on each maturation event).

**Non-adapted parameters:** pin_threshold (does not exist — no pinning), embedding_weight (the composite is multiplicative, so embedding is a mandatory factor rather than a weight — the parameter is fixed at 1.0 and not read).

#### High-trust facts

There is no exemption class for high-trust facts. A pinned/exempt mechanism produced 79% saturation (182 of 231 facts) with pin state surviving correction and regrowing within ~30h of being zeroed — an immortal class is incompatible with natural retention. High-trust, frequently-retrieved facts surface more often through the composite score and fade like anything else when they stop being used. The measured outcome is in the paper (Section 4.4.2).

### 3. Hippocampus — Write Path (Encoding + Interference + Surprise)

```python
def remember(content: str, user_id: str) -> dict:
    # Guardian case: detect if fact is about another user
    target_user = detect_subject_user(content, user_id)
    if target_user and target_user != user_id:
        user_id = target_user

    embedding = embedder.embed(content)

    # Interference check — find similar existing facts
    similar = storage.search(embedding=embedding, filters={"user_id": user_id}, limit=5)

    for existing in similar:
        similarity = existing["score"]  # cosine similarity (1 - distance)
        if similarity > 0.85:
            # Retroactive interference — new fact supersedes old
            storage.update(existing["fact_id"], trust_score=old_trust - interference_penalty)
        elif similarity > 0.70:
            # Near-duplicate — skip storage
            return {"status": "deduplicated", "fact_id": existing["fact_id"]}

    # Bayesian surprise — how novel is this fact?
    surprise = compute_surprise(embedding, user_id)
    initial_trust = 0.5
    if surprise > surprise_percentile:
        initial_trust = 0.5 + surprise_boost

    # Store
    fact_id = storage.store(content=content, user_id=user_id, embedding=embedding,
                           metadata={"trust_score": initial_trust,
                                     "retrieval_count": 0, "created_at": now()})
    # Entity extraction (hippocampal spatial map)
    if entity_extractor and entity_extractor.available:
        entities = entity_extractor.extract(content)
        if entities:
            storage.store_entities(fact_id, entities)
    return {"status": "stored", "fact_id": fact_id}
```

#### Bayesian surprise

```python
def _compute_surprise(self, embedding: list[float], user_id: str) -> float:
    # 1. Fetch all embeddings for this user from the database
    # 2. Compute centroid (element-wise average of all existing embeddings)
    # 3. Compute cosine distance from new embedding to centroid
    # 4. Rank that distance against the distribution of all existing distances
    # 5. Return percentile: 0.0 = expected, 1.0 = very surprising
    #
    # Centroid cached per user, recomputed every 100 writes.
    # Falls back to magnitude-based stub if < 2 existing facts or on error.
```

**Implementation:** The per-user centroid is cached in `_centroid_cache` and recomputed every 100 writes (tracked by `_centroid_write_count`). The distance distribution is cached in `_centroid_distances` for percentile ranking. Cosine distance: `1.0 - cosine_similarity` (0 = identical, 1.0 = orthogonal, 2.0 = opposite). Falls back to the magnitude-based stub when there are fewer than 2 existing facts (first facts have no reference distribution) or on any error.

### 4. Thalamus — Routing and Gating (Prefetch Pipeline)

```
INPUT: user_message, session_id, user_id

STEP 1: Evaluate trust signals from previous turn
  last_injected = session_buffer[session_id]
  for fact in last_injected:
    overlap = text_overlap(fact.content, user_message)
    is_correction = check_correction(user_message)
    if is_correction and overlap > 0.3:
      trust -= correction_penalty
      record_signal(was_correction=True)
    elif overlap > 0.3:
      trust += continuation_boost
      record_signal(was_continuation=True)
    persist

STEP 2: Hybrid search — vector + full-text, fused via RRF
  query_embedding = embedder.embed(user_message)

  # Vector search (over-fetch: limit * 5)
  vector_results = storage.search(
      embedding=query_embedding,
      filters={"user_id": user_id},  # or user_id_any for guardians
      limit=limit * 5                # over-fetch so activation gate has a larger pool
  )

  # Full-text search (BM25)
  fts_results = storage.search_fts(
      query_text=user_message,
      filters={"user_id": user_id},
      limit=limit * 5
  )

  # Reciprocal Rank Fusion — fuse rankings
  # RRF score = 1/(k + rank_vector) + 1/(k + rank_fts)
  if fts_results:
      raw_results = rrf_fuse(vector_results, fts_results, limit * 5)
  else:
      raw_results = vector_results

  # Entity graph traversal (hippocampal spatial map)
  if entity_extractor and entity_extractor.available:
      query_entities = entity_extractor.extract(user_message)
      if query_entities:
          entity_names = [e["name"] for e in query_entities]
          graph_results = storage.search_graph(entity_names, filters, limit * 5)
          if graph_results:
              existing_ids = {r.get("fact_id") for r in raw_results}
              for gr in graph_results:
                  if gr.get("fact_id") not in existing_ids:
                      gr["score"] = 0.3  # moderate score so graph results pass the floor
                      raw_results.append(gr)

STEP 3: Score, gate, and prime
  for fact in results:
    activation = sigmoid(hours_since_created, retrieval_count, t_half, k)

    if activation < 0.5:
      # Confidence rescue (0.6.1): a just-stored fact with activation < 0.5 but
      # cosine >= recall_confidence_floor IS the correct answer for this query —
      # it surfaces as active immediately. Without the rescue, stale matured
      # facts with mediocre cosines keep the active gate non-empty, and the true
      # best match starves in the silent pool (measured live: the best fact,
      # cosine 0.534, sat silent behind 0.28-0.32 junk).
      if emb_score >= recall_confidence_floor:
          pass  # surface as active; composite computed below
      else:
          # Silent — not injected. Silent facts simply mature; there is no
          # read-time channel from a silent fact to a mature one.
          # (fallback pool: cosine >= relevance_floor kept for empty-gate turns)
          if emb_score >= relevance_floor:
              silent_facts.append(fact)
          continue

    # Active fact — compute composite score
    # Temporal decay is applied as a WRITE operation (apply_decay) every turn
    # via tick(), persisting decayed trust_score to the database.
    # No fact is exempt: trust_score already has decay applied
    # (write-time via apply_decay), so there is no read-time override.
    effective_trust = trust_score

    composite = emb_score * (1.0 - trust_weight * (1.0 - effective_trust))

STEP 4: Gating — relevance floor + top-confidence guard
  gated = [f for f in scored if f.composite >= relevance_floor]

  # Top-confidence guard (0.6.1): if even the best fact's RELEVANCE (raw cosine,
  # not composite) is below recall_confidence_floor, the query is a near-miss —
  # inject nothing rather than bias the turn with tangential facts. Gated on
  # cosine, not composite: composite multiplies cosine by a trust damping factor
  # (up to -20%), conflating "how relevant was the match" with "how much do we
  # trust that fact" — natural queries about low-trust (new) facts would falsely
  # abstain (measured: 5/10 natural probes abstained via composite, 0/10 via cosine).
  top_relevance = max(f.relevance for f in scored) if scored else 0.0
  if top_relevance < recall_confidence_floor:
      return ""   # no injection this turn; per-fact relevance_floor still applies above

  # Fallback: if no active facts passed the gate, use high-relevance silent facts
  if not gated and silent_facts:
    return silent_facts[:context_budget]

  gated.sort(key=composite, reverse=True)
  injected = gated[:context_budget]

STEP 5: Update retrieval metadata
  for fact in injected:
    retrieval_count += 1
    last_retrieved = now()
  persist

STEP 6: Store in session buffer for next-turn evaluation
  session_buffer[session_id] = [f.id for f in injected]

STEP 7: Trust engine tick (includes temporal decay)
  apply_decay()     # write operation: decays trust_score for all facts
                   # based on Ebbinghaus curve since max(last_retrieved, created_at, updated_at)
# anchored on the latest so per-turn application cannot re-charge elapsed time
                   # decay clock only resets on actual retrieval, not on decay application
  trust.tick()     # adapts every 50 turns
    _scan_and_sweep_gated_facts()  # background scan for facts below atrophy threshold
  _sweep_atrophy()               # delete marked facts past grace period (if any)
  _recheck_contradictions()      # re-assert open corrections whose penalty was outvoted

STEP 8: Format context block with identity directive
  context = identity_directive + "\n" + fact_contents
  inject_into_system_prompt("<limbic-context>" + context + "</limbic-context>")
```

**Composite scoring formula (implementation):**

```
composite = emb_score * (1.0 - trust_weight * (1.0 - effective_trust))
```

Where:
- `emb_score` = cosine similarity from KNN (1.0 - distance)
- `effective_trust` = trust_score (decay already applied as write operation via `apply_decay()`; all facts decay equally)
- `trust_weight` = 0.20 — the maximum fraction by which low trust damps relevance

**Relevance is a precondition, not a term.** The formula is multiplicative, so zero similarity yields zero composite regardless of trust. Trust can only damp a relevant fact (at most 20% at trust 0); it can never lift an irrelevant one. `activation` is not a scoring term — it gates injection only (see §5).

**Silent-facts fallback:** If no active facts pass the relevance gate, high-relevance silent facts (activation < 0.5 but embedding_score >= floor) are returned sorted by embedding score. This prevents empty recall when all matching facts are still in the maturation period. The confidence rescue (STEP 3) narrows this fallback's job to the truly-marginal band: facts whose relevance clears the confidence floor never enter it.

**Confidence-gated recall (tool contract):** the explicit `recall()` tool reports `{"status": "no_confident_match", "top_score": S}` when the best relevance falls below the floor — read-side metacognition. `prefetch()` shares the same top-relevance guard but returns an empty context block instead of a status field (the turn simply runs without memory). See "What the Agent Sees" for the agent-facing contract.

**Over-fetch:** The search requests `limit * 5` results. Without this, a crowd of recent silent facts can dominate the top-K and starve recall of active facts that are further in vector distance but actually eligible for injection.

#### Performance budget

| Component | Time |
|---|---|
| Embedding generation (bundled ONNX model, CPU) | ~17-19ms steady-state (measured 2026-10-10, production store); one-time ~1s ONNX session load paid at first load — moved off the dispatch path via a warm-up thread at initialization (daemon, fire-and-forget); `_init_lock` double-checked locking makes concurrent warm-up + first-embed safe (regression-tested, 3-thread) |
| sqlite-vec KNN search (brute-force, linear in fact count) | ~1-2ms at ~1k facts |
| FTS5 full-text search (BM25) | ~1-2ms |
| Entity graph traversal (SQL JOIN on mentions) | ~0.5-1ms |
| RRF fusion + graph merge (Python) | ~0.2ms |
| Trust signal evaluation | ~0.5ms |
| NLI correction detection | ~2-4ms (write time only) |
| Composite scoring + gating | ~0.5ms |
| Retrieval metadata update | ~0.5ms |
| **Total (bundled model, hybrid)** | **~10-17ms** |
| **Total (endpoint, cached embedding)** | **~6-8ms** |

#### Scale projections

| Facts | KNN search | Total prefetch |
|---|---|---|
| 100 | <1ms | ~18-20ms |
| 1,000 | <2ms | ~19-21ms |
| 5,000 | ~10ms | ~27-29ms |
| 10,000 | ~20ms | ~37-39ms |
| 100,000 | ~200ms | ~217-219ms |

**KNN search is a linear scan**, so cost grows with fact count; the 5,879-fact measurement (10.2ms, Section 4.1.1) sits where the curve predicts. Embedding generation is constant at ~17-19ms steady-state (int8 Arctic L, measured 2026-10-10 on the production store; earlier ~5-10ms estimates under-measured the 1024-dim model). One-time ~1s ONNX session load is paid at initialization via warm-up thread, not per-turn. An ANN index is future work in sqlite-vec, and would change the shape of this table.

### 5. Maturation — Activation Sigmoid (Amygdala)

```python
def compute_activation(created_at, retrieval_count: int) -> float:
    hours = (now() - created_at).hours
    t_half = params.maturation_t_half  # 168 hours default
    k = params.maturation_k            # 48 default

    # Retrieval accelerates maturation — each retrieval shifts curve left
    acceleration = retrieval_count * (t_half * 0.1)
    adjusted_t_half = max(t_half - acceleration, t_half * 0.2)

    activation = 1 / (1 + exp(-(hours - adjusted_t_half) / k))
    return min(activation, 1.0)
```

- Silent (A~0.03): just stored, primes but doesn't surface
- Threshold (A=0.5): starts being eligible for injection
- Mature (A>0.9): fully available

Below 0.5: silent — does not inject directly, and there is no channel from a silent fact to a mature one.

**Datetime handling:** Both `compute_activation` and `apply_decay` handle both ISO-format strings and datetime objects. The code checks `isinstance(created_at, datetime)` first, then falls back to `datetime.fromisoformat()` for string inputs.

### 6. Reconsolidation — Organic Correction (Hippocampus, write-time NLI)

#### Design principle

**Content is never modified after storage.** The system never blends, merges, appends, or replaces text in an existing fact. Corrections work through trust and activation dynamics — the same mechanisms that govern everything else.

#### How corrections propagate

```
1. remember("flat user, no domain separation") → stored as new silent fact (activation 0.03)
2. Silent fact matures over time and begins surfacing
3. is_correction() pattern matches on the user's message
4. For each mature fact that the silent fact primes:
     if is_correction AND vector_similarity > 0.6:
       mature_fact.trust_score -= correction_penalty
5. Mature fact's trust drops below relevance_floor → stops surfacing
6. Correction fact matures over time → begins surfacing
7. System has self-corrected. No content was touched.
```

#### Correction detection

Regex patterns (in trust engine): "no", "wrong", "actually", "that's not", "incorrect", "revised", "updated", "changed", "now", "not anymore", "don't", "isn't", "wasn't", "used to", "previously"

If pattern matches AND the new fact primes a mature fact (vector similarity > 0.6) → correction detected. The mature fact's trust is penalised through the existing `correction_penalty` parameter.

Embeddings are not used for correction *detection* — only for the priming/similarity gate above. Embeddings capture topic similarity, not semantic opposition: "I eat meat" and "I am vegan" have high cosine similarity (both about diet), making corrections indistinguishable from continuations on embeddings alone. NLI (write-time) and regex patterns (read-time) are the correction detection mechanisms.

### 7. Identity Interface

```python
class LimbicMemoryProvider:
    def initialize(self, session_id, user_id=None, user_profile=None, **kwargs):
        # Resolve canonical user_id from identity system
        canonical_user_id, guardian_of, display_name, scope = self._resolve_identity(user_id, kwargs)

        # Map session → canonical user
        self._session_user_id[session_id] = canonical_user_id
        self._session_guardian_of[session_id] = guardian_of
        self._session_display_name[session_id] = display_name
        self._session_scope[session_id] = scope

        # Baseline seeding (genetic predisposition)
        if user_profile and canonical_user_id != "general":
            self._maybe_seed(canonical_user_id, user_profile)
```

Identity resolution chain:
1. **User manifest** — users.yaml (canonical name, scope, guardian_of) resolved via `manifest.resolve(platform_user_id, platform)`
2. **Matrix fallback** — strip homeserver suffix from `@user:server.tld`, remove trailing digits, lowercase
3. **Raw fallback** — use platform_user_id as-is, lowercased

Concurrent sessions: session-keyed user_id cache with threading.Lock.

### 8. Plugin Architecture

```yaml
# plugin.yaml
name: limbic
version: 0.5.1
description: "Self-organising memory system. Opaque, adaptive, self-contained. Modelled on the biological limbic system."
author: "D Yiapanis"
kind: exclusive
pip_dependencies:
  - sqlite-vec
  - onnxruntime
  - tokenizers
  - pyyaml
  - spacy
```

Registration is via the memory-provider contract: `register(ctx)` calls `ctx.register_memory_provider(provider)`. The host MemoryManager dispatches the lifecycle methods (`on_pre_llm_call`, `on_post_tool_call`, `on_session_start`, `on_session_end`, `on_pre_compress`, `on_memory_write`) and serves the tools (`remember`, `recall`) — plugin.yaml declares no hooks or tools of its own.

Activation is config-driven (`memory.provider: limbic` in the host profile's `config.yaml`). Three points the host operator must know:

- **`provider: limbic` does not disable the built-in memory store.** The built-in MEMORY.md/USER.md store is gated by its own flags (`memory.memory_enabled`, `memory.user_profile_enabled`), which default to `true`. Set both to `false` — otherwise the built-in store runs alongside Limbic and the agent maintains two competing memory systems.
- **Limbic mirrors built-in writes as a safety net** (`on_memory_write`): if anything bypasses the disabled flags and writes to the built-in store, the fact is captured in Limbic too — nothing is silently lost during transition.
- The built-in `memory` tool schema remains in `agent.valid_tool_names` even when disabled; Limbic's `system_prompt_block()` steers the agent to `remember`/`recall` instead.

### 9. Trust Signal Evaluation

Last-injected buffer per session. On next turn's pre_llm_call:
- Text overlap between injected fact content and new user message → continuation/correction
- Trust adjustments persisted immediately
- Signals recorded for adaptation engine

### 10. Adaptive Parameter Storage

Key-value store with min/max bounds. Updated every N turns (default 50). Clamped to bounds. Defaults seeded on first run. Stored as JSON file alongside the database (`limbic_params.json`).

### 11. Database Triggers — Automated Trust Lifecycle

SQLite triggers fire atomically on write, moving write-time logic from Python polling into the database.

The only trigger governing trust lifecycle is `track_maturity` below.

**Track maturity trigger:** When retrieval_count crosses 5 and matured_at is not yet set:
```sql
CREATE TRIGGER track_maturity
AFTER UPDATE OF retrieval_count ON facts
FOR EACH ROW
WHEN OLD.retrieval_count < 5 AND NEW.retrieval_count >= 5 AND NEW.matured_at IS NULL
BEGIN
    UPDATE facts SET matured_at = datetime('now') WHERE fact_id = NEW.fact_id;
END
```

**What stays in Python:** Composite scoring, activation sigmoid, correction propagation, and retrieval. These are read-time or conditional logic that doesn't map to triggers.

### 12. Analytical Self-Evaluation

The `stats()` operator function queries the store for performance metrics:

```python
stats(user_id="alice")
# → {"user_id": "alice", "total_facts": 4912, "avg_trust": 0.535,
#    "matured": 0, "faded": 2}

stats()  # aggregate
# → {"total_facts": 4946, "users": [{"user_id": "bob", "facts": 14}, ...]}
```

Metrics: total facts, average trust, matured count, faded count (facts below 0.3 trust — likely corrected or irrelevant). This is the basis for richer adaptation in the future — the system can observe its own performance over time.

---

## Implementation Status

All mechanisms described in this spec are implemented — every declared interface member is overridden, and every adaptive parameter named here is read by the engine.
