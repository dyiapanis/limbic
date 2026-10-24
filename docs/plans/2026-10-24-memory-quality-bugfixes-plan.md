# Limbic Memory-Quality Bugfix Plan — fix existing code only

**Author:** Phoenix, 2026-10-24. **Doctrine (user directive):** optimise and weed out
bugs from existing code; NO new features, tools, or manual-cleanup rituals.
Every lane below is either (a) a fix to code that already exists and misbehaves,
or (b) a one-shot sanctioned data repair for damage the bugs already produced.

**Basis:** full-store audit 2026-10-23/24 (read-only forensics, all 6 profiles)
+ code-level root-cause pass over `provider.py` / `trust.py` / `users.yaml`.

## Audit evidence (compressed)

- F1: same person stored as two user_ids — phoenix: `demetri` 355 vs
  `@demetri1819:unredacted.org` 292 (fleet-wide pattern; herb 55/46, alfred 46/50).
  Junk ids: `user_id='6'` (4 facts), `general` (4).
- F2: most-retrieved facts at near-zero trust — #1294 r=231 t=0.029 (retrieved
  TODAY), #1158 r=67 t=0.037, #1118 r=50 t=0.021; 8 of 12 such outliers had
  `updated_at=2026-10-09`. Global trust↔retrieval r=+0.324, top-15 median 0.743 —
  a bimodal population, not a global inversion.
- F3: atrophy machinery: `matured_at` set on 512 facts (alive) but **0 atrophy
  marks ever**; 69/658 never retrieved; 12 at trust ≤ 0.0.
- F4: write-path pollution — 3 facts are literal session-frame wrappers
  ("[OUT-OF-BAND USER MESSAGE — …"), 17 chat-turn fragments stored as facts.
- Minor: 3 exact-dup groups (6 rows), 87/658 facts with no entity links,
  ~56 meta-fragment candidates, ~12 trust-0.0 stragglers.

## Root causes (code level, verified by read)

- RC1 (F1): `users.yaml` manifest lacks the `@demetri1819:unredacted.org →
  demetri` canonical mapping, so `_resolve_identity()` falls through to raw-MXID
  bucketing (provider.py:2090-2116). The `6`/`general` ids leak from callers that
  pass unresolved values past remember()'s `user_id or self._get_user_id(...)`
  (provider.py:292; the `general` default, provider.py:95).
- RC2 (F2): the interference penalty (NLI-neutral ∧ similarity>0.85 → −0.20 on
  the OLD fact, provider.py:430-444) is applied cumulatively per neighboring
  write, with no floor and no retrieval-based counter-pressure. A doctrine-class
  fact sitting in a dense semantic neighborhood takes one −0.20 dock per adjacent
  write and graves out within a handful of sessions. The correction penalty
  (−0.15, provider.py:395) has the same unbounded structure but is evidence-based
  (NLI contradiction), so it stays.
- RC3 (F3): atrophy candidacy is checked ONLY for facts present in the current
  recall's search `results` (provider.py:1017 `for fact in results:`). Facts
  that stop matching any query — all the dead weight — are unreachable by the
  marker and therefore never swept. The machinery works; it can't see its food.
- RC4 (F4): the fragment gate (provider.py:306-313) is length/word-count only;
  the host-frame wrapper text and any >60-char transcript turn pass it.

## Lanes

### L1 — Trust penalty floor + safe heal (B2 fix; trust.py, one code path)
Give the interference dock a floor so it demotes but cannot grave:
`new_trust = max(NEW_INTERFERENCE_FLOOR, old_trust - penalty)` with
`NEW_INTERFERENCE_FLOOR = 0.25` (configurable in DEFAULT_PARAMS like the
penalties; named `interference_floor`). The contradiction path keeps its full
−0.15 (evidence-based).
**Heal rule — reviewer-accepted, filtered form:** during the decay pass, a fact
is healed to `max(0.30, trust)` ONLY when `trust < 0.10 AND retrieval_count >=
10 AND` it has **no open contradiction link**: `SELECT 1 FROM contradictions
WHERE older_fact_id = :id AND resolved_at IS NULL` must return nothing (use the
existing helpers `open_supersessions`/`count_open`, store_sqlite.py:943/970).
Live counter-example why the filter is mandatory: fact 1302 (t≈0.08, r=180)
carries an OPEN link to fact 1628 (created 2026-10-23) — a blanket heal would
resurrect a fact that is still under active correction. Facts whose links are
all resolved are safe (the correction won).
**Acceptance:** #1294/#1158/#1118 — all fully-resolved links — recover to ≥
0.30 after one touch; fact 1302 stays untouched; a scripted 10-write
neighborhood-burst against a fresh store leaves the anchor fact ≥ 0.25; suite
green.
**Effort:** ~40 lines + params doc. **Rollback:** revert one commit.

### L2 — Sweeper reachability (B3 fix; provider.py, small)
Move atrophy marking into `_apply_temporal_decay`'s existing per-turn loop over
`decay_rows(user_id)` — that loop already visits every fact of the active user;
add the `is_atrophy_candidate` check there. **Single marking site (reviewer
amendment): delete BOTH the search-results marking (provider.py:1014-1028) AND
the second full scan `_scan_and_sweep_gated_facts` (provider.py:1278-1320) —
keeping both = redundant full scans per turn.** One query, one loop, one owner.
**Acceptance:** on the real store, the 12 trust-0.0 facts become marked within
the active user's next prefetch; after grace they sweep. Then stop: the sweeper
now works and the store self-cleans going forward.
**Effort:** ~15 lines, net −10. **Rollback:** revert one commit.

### L3 — Write-boundary junk guard (B4/RC4 fix; provider.py fragment gate)
Extend the existing fragment gate with two mechanical rejections:
(a) host-frame wrapper text — detect the session-frame wrapper pattern
("OUT-OF-BAND USER MESSAGE", "CONTEXT COMPACTION", "delivered once at this
position") — this is *our own mechanism's framing leaking into content*, not
classification, so a literal pattern list is honest here;
(b) unresolved-identity guard at store time: if the resolved uid is not in the
manifest and not a valid MXID-shaped id or the explicit `general` opt-in,
store under the resolved session user (log the attribution repair) instead of
storing the garbage id. Covers the `'6'` recurrence at source.
**Acceptance:** the audit's junk classes cannot be re-created through remember();
the existing fragment-rejection tests still pass; suite green.
**Effort:** ~20 lines. **Rollback:** revert one commit.

### L4 — One-shot data repair of past damage (sanctioned ops, not features)
Order matters; each step = provider/CLI verbs or documented provider-API
actions, never raw SQL:
1. **L4a — users.yaml canonical map:** add `@demetri1819:unredacted.org → demetri`
   (and confirm `@kirsch`/`usha` mappings exist or get them) so identity
   resolution stops forking.
2. **L4b — Alias merge on stored rows:** re-attributed via the provider's
   storage update path (user_id re-point to `demetri`), then the standard
   reindex. 45/45 split → single bucket. Junk ids (`6`, stray `general`)
   fold into `demetri` unless content-clearly-not-about-you → delete.
3. **L4c — Data repair order (reviewer amendment):** delete the audit-confirmed
   junk (3 wrapper facts, 17 chat-turn fragments — exact row list reviewed by the
   reviewer before execution), then **post-merge dedup KEEPING MAX-TRUST rows**
   (`content`-grouped duplicates after the alias re-point, anchor
   `store_sqlite.py:801,854`) — and only then the L1 heal rule, filtered per L1.
4. **L4d — Entity backfill:** re-run the existing entity-extraction pass over
   facts with no mentions (87 in phoenix) using the plugin's existing verb.
**Acceptance:** stats show single canonical user bucket; recall spot-checks
return the previously half-visible alias content; `hermes limbic stats` clean.
**Effort:** ~1h incl. verification. **Rollback:** pre-repair DB backup
(limbic.db.bak per profile) + reindex.

### L5 — Repo-doc repair (reviewer amendment: fix the anchor)
The stale pruning doctrine lives in **my skills tree**
(`phoenix profile skills/limbic-memory-plugin/references/pruning-stale-facts.md`),
not the limbic repo — the reviewer correctly grepped the repo and found nothing.
Fix target: that skills reference — it documents a schema that no longer exists
(`pinned` column) and prescribes direct-SQL DELETE/UPDATE, contradicting the
plugin's DATA ACCESS POLICY (CLI/provider-API only, "if the CLI lacks a verb —
request it"). Rewrite it to the sanctioned path (provider verbs + L2's working
sweeper), keep the surgical-prune mental model.
**Effort:** ~30 min. **Rollback:** git revert.

### L6 — Verification gate (existing eval harness)
Before L1/L2 land and after L4: run the existing `eval/` suite
(LoCoMo-style recall QA + the mechanism gates in `references/development-eval-gates.md`),
plus `hermes limbic stats` diff before/after. Gateways restarted by the user
afterwards (Phoenix cannot). Success = recall quality no worse on LoCoMo,
junk-classes unreproducible, trust outliers recovered.

## Execution order
L5 docs → L1 trust floor → L2 sweeper → L3 guard → L4 data repair (L4a..L4d) →
L6 verify. Every code lane = its own commit + revert path. The plugin's own
eval gates from `references/development-eval-gates.md` apply to each code lane.

## Explicitly out of scope (by user doctrine)
No salience/AI classifier gates, no supersede-rewrite of storage semantics, no
new CLI verbs beyond what exists, no recall-scoring rewrites, no schema changes
(the L1 floor + L2 sweeping need zero new columns).

## Open questions for the reviewer
1. interference_floor = 0.25 — right value? (audit: outlier cluster = 0.02-0.05;
   healthy band = 0.6-0.8)
2. delete or keep the search-results marking path in L2?
3. L4b: any facts under alias ids that should stay separate rather than merge?
4. L4c deletion list — reviewer reviews the exact 20-25 rows before we touch them.