# Anatomy Map — Biology to Implementation

Single source of truth for the biological mapping. `SPEC.md` §2 and the paper
(§1.3) reference this file rather than restating it.

**Structure.** Four regions. Every mechanism is owned by exactly one region.
Processes that are not regions (maturation, reconsolidation, atrophy) are
listed under the region that performs them — never as peers of a region.

**Truth rules.** A row exists only if the code exists. The `Code` column names
a real symbol; `verify_anatomy_map.py` fails if any symbol disappears. Nothing
in the mapping is load-bearing at runtime: the modules contain no anatomical
identifiers. The anatomy is a design *guide*, not an architecture.

---

## Hippocampus — Encoding

*Brain function:* converts experience into memory; place cells map spatial context.

*Implementation:* the write path. `remember()` in `provider.py`.

| Process | Code | Fires |
|---|---|---|
| Embedding + dedup (>0.70 cosine = near-duplicate, skip) | `provider.remember()` | write |
| Interference penalty (retroactive >0.85) | `provider.remember()` → `TrustEngine.interference_penalty_value()` | write |
| Bayesian surprise boost (centroid distance) | `provider._compute_surprise()` | write |
| Entity extraction → entity graph | `entities.EntityExtractor` | write |
| Initial trust 0.5 (0.6 if high surprise), activation ~0.03 | `provider.remember()` | write |
| **Reconsolidation** — NLI contradiction detection, trust penalty, and the continuing-correction link re-check | `provider.remember()` → `nli.LimbicNLI`; `provider._recheck_contradictions()` | write + per turn |

## Amygdala — Salience

*Brain function:* tags memories with weight; unreinforced memories fade.

*Implementation:* everything that moves trust after the write. `trust`.

| Process | Code | Fires |
|---|---|---|
| Implicit signals (continuation / correction / ignore) | `provider._evaluate_trust_signals()` → `TrustEngine.record_signal()` | per turn |
| Temporal decay (Ebbinghaus, clock resets on retrieval) | `TrustEngine.apply_decay()` | per turn |
| Adaptive parameters (decay rate, relevance floor, maturation curve) | `TrustEngine` + `provider.remember()` | per turn |
| Atrophy criterion — what makes a fact eligible for removal | `TrustEngine.is_atrophy_candidate()` | per turn |
| **Maturation** — activation sigmoid, silent → retrievable | `TrustEngine.compute_activation()` | per turn |

There is no exemption class — nothing in the table above or elsewhere skips decay or the correction penalty.

## Thalamus — Routing and Gating

*Brain function:* routes signals; gates what reaches cortex.

*Implementation:* retrieval, before every LLM call. `provider.prefetch()`.

| Process | Code | Fires |
|---|---|---|
| Three-path search (vector + FTS + entity graph) | `provider._search()`, `store_sqlite.search_graph()` | per turn |
| Reciprocal-rank fusion (preserves cosine as `vector_score`) | `provider._rrf_fuse()` | per turn |
| Relevance gate — `similarity × (1 − tw·(1−trust))`, floor 0.25 | `provider.prefetch()` → `TrustEngine.relevance_floor_value()` | per turn |
| Context assembly into `<limbic-context>` | `provider.prefetch()` | per turn |
| Bounded maintenance (trust tick, atrophy sweep) | `provider._sweep_atrophy()`, `TrustEngine.tick()` | periodic |

**Output, not a component:** injection into the system prompt is what the
thalamus *produces*. It is not an eighth mechanism and not a brain region.

## Entorhinal Cortex — Coordinates

*Brain function:* grid/place coordinates; the spatial index, not the content.

*Implementation:* the entity graph. `entities.py` + `store_sqlite.py`.

| Process | Code | Fires |
|---|---|---|
| Mention extraction → junction rows | `entities.EntityExtractor` | write |
| Coordinate traversal (facts linked to both entities) | `store_sqlite.search_graph()` | per turn |

## Synaptic Atrophy — Removal

*Brain function:* unused synapses are pruned; the tissue is reclaimed.

*Owner:* Amygdala decides (trust criterion above); storage executes.

| Process | Code | Fires |
|---|---|---|
| Mark-and-sweep, bounded batch | `provider._sweep_atrophy()` → `store_sqlite.sweep_atrophy()` | periodic |

---

**Count:** four regions, seven mechanisms. Prompt injection is what the thalamus produces — an output, not a peer component.
