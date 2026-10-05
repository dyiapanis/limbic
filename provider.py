"""Limbic memory provider — the brain.

Coordinates hippocampus (encoding), amygdala (trust), thalamus (prefetch),
maturation (activation), and reconsolidation (auto-update).

Implements the MemoryProvider ABC from Hermes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus

from .embeddings import create_embedder, LimbicEmbedder
from .nli import create_nli_classifier, LimbicNLI
from .storage import LimbicStorage
from .trust import TrustEngine

logger = logging.getLogger(__name__)


class LimbicMemoryProvider(MemoryProvider):
    """Self-organising memory provider. Opaque, adaptive, local-first."""

    @property
    def name(self) -> str:
        return "limbic"

    def is_available(self) -> bool:
        """Always available — graceful degradation handles failures."""
        return True

    # ── Lifecycle ──────────────────────────────────────────────────

    def __init__(self):
        self._config: dict = {}
        self._storage: Optional[LimbicStorage] = None
        self._embedder: Optional[LimbicEmbedder] = None
        self._nli: Optional[LimbicNLI] = None
        self._trust: Optional[TrustEngine] = None
        self._entity_extractor = None
        self._initialised = False

        # Session state
        self._session_user_id: dict[str, str] = {}  # session_id → user_id
        self._session_buffer: dict[str, list] = {}  # session_id → last injected fact refs
        self._session_guardian_of: dict[str, list[str]] = {}  # session_id → guarded users
        self._session_display_name: dict[str, str] = {}  # session_id → display name
        self._session_scope: dict[str, str] = {}  # session_id → scope (adult/child)
        self._session_lock = threading.Lock()
        self._current_user_id: str = "general"  # fallback for prefetch with empty session_id
        self._MAX_SESSIONS = 10_000  # cap on session-keyed maps (evict oldest)

        # Centroid cache for Bayesian surprise
        self._centroid_cache: dict[str, list[float]] = {}
        self._centroid_write_count: dict[str, int] = {}
        self._centroid_distances: dict[str, list[float]] = {}  # per-user sorted distance distribution

        # Telemetry counters (observability)
        self._prefetch_latencies: list[float] = []           # ms per prefetch call
        self._embed_cache_hits = 0
        self._embed_cache_misses = 0
        self._centroid_cache_hits = 0
        self._centroid_cache_misses = 0
        self._atrophy_marked_count = 0
        self._atrophy_swept_count = 0
        self._last_recall_count = 0  # for recall_status() UI indicator
        self._component_scores: dict[str, list[float]] = {   # embedding/trust/activation contributions
            "embedding": [], "trust": [], "activation": []
        }

        # In-process DB maintenance counters (replaces external cron optimize)
        self._prefetch_count = 0  # incremented every prefetch() call
        self._last_optimize_ts = 0.0  # epoch seconds of last PRAGMA optimize
        self._last_vacuum_ts = 0.0  # epoch seconds of last VACUUM

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called by Hermes at session start."""
        # Non-primary contexts (cron jobs, subagents) must not read or write
        # user memory: their transcripts may be delivered to other channels
        # and their writes would be attributed to a stale current-user.
        # Upstream ABC contract: agent/memory_provider.py.
        if kwargs.get("agent_context", "primary") != "primary":
            self._initialised = False
            return
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))  # default profile's home is the root, not profiles/default/
        user_id = kwargs.get("user_id", "general")
        user_profile = kwargs.get("user_profile")
        agent_identity = kwargs.get("agent_identity", "default")

        # Load config
        self._load_config(hermes_home)

        # User manifest (users.yaml) — identity subset, roles ignored.
        # Multi-user mode when present; single-user mode (loud log) when not.
        from .identity import load_manifest
        self._manifest = load_manifest(self._config, hermes_home)
        self._seen_user_ids: set = set()

        # Collection name is per-agent: limbic_<agent_identity>
        collection_name = f"limbic_{agent_identity}"
        dims = self._config.get("embedding", {}).get("dims", 384)

        # Init storage — config-driven backend selection
        # SQLite is the only storage backend
        params_path = os.path.join(hermes_home, "limbic_params.json")
        if not self._storage or getattr(self._storage, "_collection", "") != collection_name:
            from .store_sqlite import SQLiteStorage
            db_path = os.path.join(hermes_home, "limbic.db")
            self._storage = SQLiteStorage(path=db_path, dims=dims, collection_name=collection_name)

        # Init embedder
        if not self._embedder:
            self._embedder = create_embedder(self._config)

        # Init NLI classifier (reconsolidation trigger — prediction error detection)
        if not self._nli:
            try:
                self._nli = create_nli_classifier(self._config)
                logger.info("limbic: NLI classifier initialised")
            except Exception as e:
                logger.warning("limbic: NLI classifier init failed — correction detection will use regex fallback only: %s", e)
                self._nli = None

        # Init entity extractor (NLP)
        if not self._entity_extractor:
            try:
                from .entities import EntityExtractor
                nlp_config = self._config.get("nlp", {})
                languages = nlp_config.get("languages", ["en"])
                # known_entities: "LABEL:pattern" strings from limbic.yaml
                patterns = []
                for entry in self._config.get("known_entities", []) or []:
                    label, _, pat = str(entry).partition(":")
                    if label and pat:
                        patterns.append({"label": label.strip(), "pattern": pat.strip()})
                self._entity_extractor = EntityExtractor(languages=languages, known_patterns=patterns)
            except Exception as e:
                logger.warning("limbic: entity extractor init failed: %s", e)
                self._entity_extractor = None

        # Init trust engine
        if not self._trust:
            self._trust = TrustEngine(params_path)

        # Resolve canonical user_id and identity from the host identity system
        # Limbic is system-agnostic: identity comes from its own user
        # manifest (users.yaml) or falls back to the raw user_id.
        canonical_user_id, guardian_of, display_name, scope = self._resolve_identity(user_id, kwargs)
        
        # Set session user (map both session_id and platform user_id)
        with self._session_lock:
            # ponytail: naive LRU-by-eviction — long-lived gateways rotate
            # session ids forever; cap the maps, drop oldest first.
            for m in (self._session_user_id, self._session_guardian_of,
                      self._session_display_name, self._session_scope):
                while len(m) >= self._MAX_SESSIONS:
                    m.pop(next(iter(m)))
            self._session_user_id[session_id] = canonical_user_id
            # Also map the raw platform user_id so prefetch can find it
            if user_id and user_id != "general":
                self._session_user_id[user_id] = canonical_user_id
            # Store a fallback for empty session_id (prefetch passes session_id="")
            self._current_user_id = canonical_user_id
            # Store identity context for this session
            self._session_guardian_of[session_id] = guardian_of
            self._session_display_name[session_id] = display_name
            self._session_scope[session_id] = scope

        # Baseline seeding (genetic predisposition)
        if user_profile and canonical_user_id != "general":
            self._maybe_seed(canonical_user_id, user_profile)

        self._initialised = True
        logger.info("limbic: initialised for session=%s user=%s (canonical=%s, guardian_of=%s)", 
                    session_id, user_id, canonical_user_id, guardian_of)

    def _load_config(self, hermes_home: str):
        """Load config from $HERMES_HOME/limbic.yaml or plugin defaults."""
        import yaml as _yaml

        config_path = os.path.join(hermes_home, "limbic.yaml")
        defaults_path = os.path.join(os.path.dirname(__file__), "limbic.yaml")

        config = {}
        if os.path.exists(defaults_path):
            with open(defaults_path) as f:
                config = _yaml.safe_load(f) or {}

        if os.path.exists(config_path):
            with open(config_path) as f:
                user_config = _yaml.safe_load(f) or {}
                config.update(user_config)

        self._config = config

    # ── Encoding (Hippocampus) ─────────────────────────────────────

    # ── Ephemeral date fact filtering ──────────────────────────────

    # Patterns that indicate a date fact is ephemeral (true only on a specific
    # day) and should not be stored as a durable memory. These cause chronic
    # day-of-week confusion when re-injected in future sessions.
    _EPHEMERAL_DATE_PATTERNS = [
        re.compile(r"^(today|tomorrow|yesterday)\s+(is|was|will\s+be)\s+\w+", re.IGNORECASE),
        re.compile(r"^(today|tomorrow|yesterday)\s*[:=]\s*\w+", re.IGNORECASE),
        re.compile(r"^what\s+(the\s+)?\w*\s*is\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", re.IGNORECASE),
    ]

    # Patterns that indicate a fact is a date-extraction artifact (from the
    # entity extractor) rather than a durable memory. These contain stale
    # date references embedded in context snippets and cause day-of-week
    # confusion when re-injected.
    _DATE_ARTIFACT_PATTERNS = [
        # Entity-extraction date fragments
        re.compile(r"^named_date:", re.IGNORECASE),
        re.compile(r"^iso_date:", re.IGNORECASE),
        # Embedded stale day-of-week references
        re.compile(r"today's\s+\w+day", re.IGNORECASE),
        re.compile(r"tomorrow(?:'s|\s+is)\s+\w+day", re.IGNORECASE),
    ]

    def _is_ephemeral_date_fact(self, content: str) -> bool:
        """Detect ephemeral date facts that are only true on a specific day.

        Examples rejected:
          "Today is Wednesday"
          "Tomorrow is Thursday"
          "Yesterday was Monday"
          "What the fuck is Thursday"
          "named_date: Jun 24 (context: ...today's Tue Jun 23...)"
          "today's Tue Jun 23, so tomorrow is Wed Jun 24"

        These accumulate as stale references and cause day-of-week confusion
        in future sessions. The system prompt already provides the current
        date, so storing these is always wrong.
        """
        stripped = content.strip().lower()
        if not stripped:
            return False
        # Short ephemeral assertions (≤60 chars)
        if len(stripped) <= 60:
            for pat in self._EPHEMERAL_DATE_PATTERNS:
                if pat.match(stripped):
                    return True
        # Date artifacts of any length (named_date: prefix, embedded "today's Xday")
        for pat in self._DATE_ARTIFACT_PATTERNS:
            if pat.search(stripped):
                return True
        return False

    def remember(self, content: str, user_id: str | None = None,
                 session_id: str | None = None) -> dict:
        """Store a fact. The hippocampus encodes experience into memory."""
        if not self._initialised:
            return {"error": "Limbic not initialised"}

        uid = user_id or self._get_user_id(session_id or "")
        if uid is None:
            return {"error": "No user_id resolved"}

        # Reject ephemeral date facts — they go stale and cause confusion
        if self._is_ephemeral_date_fact(content):
            logger.info("limbic: rejected ephemeral date fact (fact_id none, len=%d)", len(content))
            return {"status": "rejected_ephemeral_date", "reason": "Date facts go stale — the system prompt provides the current date."}

        # ponytail: fragment gate — both conditions must fail to reject.
        # words<3 catches the real junk (1-2 word steering/ack messages:
        # "yes", "ok", "proceed", "shut up"); dense short facts
        # ("Alice is vegan") pass and decay handles anything marginal.
        # Thresholds tunable in limbic.yaml.
        gate_chars = int(self._config.get("fragment_gate_chars", 60))
        gate_words = int(self._config.get("fragment_gate_words", 3))
        stripped = content.strip()
        if len(stripped) < gate_chars and len(stripped.split()) < gate_words:
            logger.info("limbic: rejected fragment (len=%d, words=%d)",
                        len(stripped), len(stripped.split()))
            return {"status": "rejected_fragment",
                    "reason": f"Too short to be a durable fact (<{gate_chars} chars and <{gate_words} words)."}

        # Guardian case: detect if the fact is about another user
        # "Bob is interested in Harry Potter" → user_id=bob, not alice
        target_user = self._detect_subject_user(content, uid)
        if target_user and target_user != uid:
            logger.info("limbic: guardian write — fact about %s stored under %s (told by %s)",
                        target_user, target_user, uid)
            uid = target_user

        # Generate embedding
        embedding = self._embedder.embed(content)
        if embedding is None:
            return {"error": "Embedding generation failed"}

        # Interference check — find similar existing facts
        similar = self._storage.search(
            embedding=embedding,
            filters={"user_id": uid},
            limit=5,
        )

        # ── NLI-aware interference and correction detection ────────
        #
        # The flow runs NLI at every similarity band above 0.60 and branches
        # on the label:
        #
        #   contradiction → store correction + penalise old fact's trust
        #   entailment    → consistent — no penalty (store new fact normally)
        #   neutral       → depends on similarity band:
        #     >0.85  → interference penalty (high similarity, not related = replacement)
        #     >0.70  → deduplicate (same topic, not related = redundant)
        #     >0.60  → no action (related but distinct — store normally)
        #
        # Contradictions are NEVER deduplicated — the correction must be stored
        # so it can mature and replace the stale fact through trust dynamics.
        #
        # This runs at WRITE TIME, so it works in all session types — not just
        # continuous gateway sessions.

        should_deduplicate = False
        dedup_fact_id = None
        # Older facts NLI flagged as contradicted by this write. The link cannot
        # be written here — this fact has no fact_id until store() below.
        contradicted_ids: list[int] = []

        for existing in similar:
            similarity = existing.get("score", 0)  # cosine similarity from KNN
            existing_content = existing.get("content", "")
            old_trust = existing.get("trust_score", 0.5)

            if similarity <= 0.60:
                continue  # too dissimilar — no action

            # Run NLI for all bands above 0.60. The NLI model is noisy on
            # unrelated pairs (probed: benign pairs mislabelled contradiction),
            # so a contradiction label only counts inside a similarity band
            # where the texts actually discuss the same thing.
            nli_label = "neutral"
            if self._nli and existing_content and similarity > 0.70:
                nli_label = self._nli.classify(existing_content, content)

            if nli_label == "contradiction":
                # ── Correction detected ────────────────────────────────
                # Penalise the old fact's trust so it fades. Store the new fact
                # (don't deduplicate) so the correction can mature and replace it.
                penalty = self._trust.correction_penalty_value()
                new_trust = max(0.0, old_trust - penalty)

                self._storage.update(existing["fact_id"], trust_score=new_trust)
                logger.debug(
                    "limbic: NLI contradiction — fact %d trust %.2f → %.2f (similarity %.2f)",
                    existing["fact_id"], old_trust, new_trust, similarity,
                )

                self._check_organic_atrophy(
                    existing["fact_id"], new_trust,
                    existing.get("last_retrieved"),
                    created_at=existing.get("created_at"),
                )

                contradicted_ids.append(existing["fact_id"])

            elif nli_label == "entailment":
                # ── Consistent fact ────────────────────────────────────
                # The new fact is consistent with or elaborates the existing one.
                # No penalty. At high similarity, this is effectively a duplicate
                # — skip storage to avoid redundancy. At lower similarity, store
                # the new fact as it adds distinct information.
                if similarity > 0.70:
                    should_deduplicate = True
                    dedup_fact_id = existing["fact_id"]
                    logger.debug(
                        "limbic: NLI entailment — fact %d entails new content (similarity %.2f), deduplicating",
                        existing["fact_id"], similarity,
                    )
                # else: store normally — the new fact adds information

            else:
                # ── Neutral (unrelated despite topic overlap) ──────────
                # Fall back to similarity-band logic:
                if similarity > 0.85:
                    # High similarity but not contradictory or entailing —
                    # the new fact is probably a newer version of the same thing.
                    # Apply interference penalty.
                    new_trust = max(0.0, old_trust - self._trust.interference_penalty_value())
                    self._storage.update(existing["fact_id"], trust_score=new_trust)
                    self._check_organic_atrophy(
                        existing["fact_id"], new_trust,
                        existing.get("last_retrieved"),
                        created_at=existing.get("created_at"),
                    )
                    logger.debug(
                        "limbic: neutral interference — fact %d trust %.2f → %.2f (similarity %.2f)",
                        existing["fact_id"], old_trust, new_trust, similarity,
                    )
                elif similarity > 0.70:
                    # Same topic, not related — redundant. Skip storage.
                    should_deduplicate = True
                    dedup_fact_id = existing["fact_id"]
                # 0.60-0.70 neutral: store normally — related but distinct

        # Check deduplication result
        if should_deduplicate:
            return {"status": "deduplicated", "fact_id": dedup_fact_id}

        # Bayesian surprise — how novel is this fact?
        surprise = self._compute_surprise(embedding, uid)
        initial_trust = 0.5
        if surprise > self._trust.surprise_percentile_value():
            initial_trust = 0.5 + self._trust.surprise_boost_value()

        # Store
        fact_id = self._storage.store(
            content=content,
            user_id=uid,
            embedding=embedding,
            metadata={
                "trust_score": initial_trust,
                "retrieval_count": 0,
                "created_at": _now_iso(),
            },
        )

        logger.debug("limbic: stored fact_id=%d user=%s trust=%.2f surprise=%.2f",
                     fact_id, uid, initial_trust, surprise)

        # Continuing correction: link this fact to each older fact it contradicts.
        # The write-time penalty above is one-shot; the link is what lets the tick
        # re-assert the correction if the older fact's trust recovers later.
        for older_id in contradicted_ids:
            try:
                self._storage.link_contradiction(fact_id, older_id)
            except Exception as e:
                logger.warning("limbic: could not link contradiction %d→%d: %s",
                               fact_id, older_id, e)

        # Extract entities and create graph edges (NLP)
        if self._entity_extractor and self._entity_extractor.available:
            entities = self._entity_extractor.extract(content)
            if entities:
                self._storage.store_entities(fact_id, entities)
                logger.debug("limbic: extracted %d entities from fact %d",
                             len(entities), fact_id)

        return {"status": "stored", "fact_id": fact_id}

    def _compute_surprise(self, embedding: list[float], user_id: str) -> float:
        """Bayesian surprise — how novel is this fact relative to the user's existing facts?

        Computes the cosine distance from the new embedding to the per-user centroid
        (average of all existing fact embeddings), then ranks that distance against
        the distribution of distances from each existing fact to the centroid.

        Returns a percentile: 0.0 = completely expected, 1.0 = very surprising.

        The centroid is cached per user and recomputed every 100 writes.
        Falls back to the MVP magnitude-based stub if no existing facts or on error.
        """
        try:
            # Check if we need to recompute the centroid
            write_count = self._centroid_write_count.get(user_id, 0)
            if write_count == 0 or write_count >= 100 or user_id not in self._centroid_cache:
                self._centroid_cache_misses += 1
                # Fetch all embeddings for this user
                embeddings = self._storage.get_all_embeddings(user_id)
                if not embeddings or len(embeddings) < 2:
                    # Not enough facts to compute a meaningful centroid
                    self._centroid_write_count[user_id] = 0
                    return self._compute_surprise_stub(embedding)

                # Compute centroid (element-wise average)
                dims = len(embeddings[0])
                centroid = [0.0] * dims
                for emb in embeddings:
                    for i in range(dims):
                        centroid[i] += emb[i]
                centroid = [c / len(embeddings) for c in centroid]
                self._centroid_cache[user_id] = centroid
                self._centroid_write_count[user_id] = 0

                # Cache the distance distribution for percentile ranking
                distances = [self._cosine_distance(emb, centroid) for emb in embeddings]
                distances.sort()
                self._centroid_distances[user_id] = distances
            else:
                self._centroid_cache_hits += 1
                centroid = self._centroid_cache.get(user_id)
                distances = self._centroid_distances.get(user_id, [])

            if not centroid or not distances:
                return self._compute_surprise_stub(embedding)

            # Compute distance from new embedding to centroid
            new_distance = self._cosine_distance(embedding, centroid)

            # Rank against the distribution (percentile)
            count = len(distances)
            rank = sum(1 for d in distances if d < new_distance)
            percentile = rank / count

            # Increment write count for cache invalidation
            self._centroid_write_count[user_id] = self._centroid_write_count.get(user_id, 0) + 1

            return percentile

        except Exception as e:
            logger.debug("limbic: surprise computation failed, using stub: %s", e)
            return self._compute_surprise_stub(embedding)

    def _compute_surprise_stub(self, embedding: list[float]) -> float:
        """Fallback: embedding magnitude as proxy for novelty."""
        magnitude = math.sqrt(sum(x * x for x in embedding))
        return min(magnitude / 10.0, 1.0)

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Cosine distance: 0 = identical, 2 = opposite. Returns 1 - cosine_similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    @staticmethod
    def _is_newer(ts_a, ts_b) -> bool:
        """Returns True if timestamp a is more recent than timestamp b.
        Handles both ISO-format strings and datetime objects."""
        try:
            if isinstance(ts_a, datetime):
                a = ts_a if ts_a.tzinfo else ts_a.replace(tzinfo=timezone.utc)
            else:
                a = datetime.fromisoformat(str(ts_a).replace("Z", "+00:00"))
            if isinstance(ts_b, datetime):
                b = ts_b if ts_b.tzinfo else ts_b.replace(tzinfo=timezone.utc)
            else:
                b = datetime.fromisoformat(str(ts_b).replace("Z", "+00:00"))
            return a > b
        except Exception:
            return False

    @staticmethod
    def _rrf_fuse(vector_results: list[dict], fts_results: list[dict],
                  limit: int = 20, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion — fuse vector and full-text search results.

        RRF score = 1/(k + rank_in_vector) + 1/(k + rank_in_fts)
        A fact that appears in both lists gets both contributions.
        A fact in only one list gets only that contribution.
        """
        scores = {}
        facts_by_id = {}

        for rank, r in enumerate(vector_results):
            fid = r.get("fact_id", 0)
            scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
            if fid not in facts_by_id:
                facts_by_id[fid] = r

        for rank, r in enumerate(fts_results):
            fid = r.get("fact_id", 0)
            scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
            if fid not in facts_by_id:
                facts_by_id[fid] = r

        # Sort by fused score, return top limit
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        result = []
        for fid, rrf_score in sorted_ids:
            fact = facts_by_id.get(fid)
            if fact:
                # Preserve the raw cosine similarity before RRF overwrites
                # `score`. RRF is a RANK score (~0.016-0.033) on a completely
                # different scale to cosine (~0.0-1.0); the relevance gate
                # needs the cosine, the ordering needs the RRF score.
                # FTS-only facts never had a cosine — leave vector_score unset.
                if rrf_score is not None:
                    fact.setdefault("vector_score", fact.get("score"))
                fact["score"] = rrf_score  # ranking score from RRF
                result.append(fact)

        return result

    # ── Retrieval (Thalamus) ───────────────────────────────────────

    def recall(self, query: str, session_id: str | None = None) -> dict:
        """Search memory. The thalamus routes and gates."""
        if not self._initialised:
            return {"error": "Limbic not initialised", "results": [], "count": 0}

        uid = self._get_user_id(session_id or "")
        if uid is None:
            return {"error": "No user_id resolved", "results": [], "count": 0}

        results = self._search(query, uid, session_id=session_id or "")
        return {
            "results": [{"content": r["content"]} for r in results],
            "count": len(results),
        }

    def _search(self, query: str, user_id: str, limit: int = 20, session_id: str = "",
                user_message: str = "") -> list[dict]:
        """Internal search — returns full fact dicts with scores.

        Silent facts (activation < 0.5) are not surfaced; they are kept only as
        a fallback for turns where nothing active clears the relevance floor.
        instead of boosted. This is organic reconsolidation: the old fact
        fades, the new fact grows, no content is ever modified.

        The KNN search over-fetches (limit * 5) so the activation gate has
        a larger pool to filter — without this, a crowd of recent silent
        facts can dominate the top-K and starve recall of active facts.
        """
        embedding = self._embedder.embed(query, is_query=True)
        if embedding is None:
            return []

        # Access control — guardian sees own + guarded, user sees own
        guardian_of = self._session_guardian_of.get(session_id, [])
        if guardian_of:
            search_filters = {"user_id_any": [user_id] + list(guardian_of)}
        else:
            search_filters = {"user_id": user_id}

        raw_results = self._storage.search(
            embedding=embedding,
            filters=search_filters,
            limit=limit * 5,  # over-fetch so the activation gate has a larger pool
        )

        # Full-text search — BM25 exact word matching
        fts_results = self._storage.search_fts(
            query_text=query,
            filters=search_filters,
            limit=limit * 5,
        )

        # Hybrid fusion — RRF combines vector + full-text rankings
        if fts_results:
            raw_results = self._rrf_fuse(raw_results, fts_results, limit=limit * 5)
        # If FTS returned nothing (no exact word matches), vector results stand alone

        # Entity graph search — extract entities from query, traverse graph
        if self._entity_extractor and self._entity_extractor.available:
            query_entities = self._entity_extractor.extract(query)
            if query_entities:
                entity_names = [e["name"] for e in query_entities]
                graph_results = self._storage.search_graph(entity_names, search_filters, limit * 5)
                if graph_results:
                    # Merge graph results with existing results, deduplicate by fact_id
                    existing_ids = {r.get("fact_id") for r in raw_results}
                    for gr in graph_results:
                        if gr.get("fact_id") not in existing_ids:
                            # Give graph results a moderate score so they pass the floor
                            gr["score"] = 0.3
                            raw_results.append(gr)

        is_correction = self._trust.is_correction(user_message) if user_message else False

        # Score and gate
        floor = self._trust.get("relevance_floor")
        scored = []
        silent_facts = []  # track high-relevance silent facts as fallback
        for fact in raw_results:
            # Skip facts already marked for atrophy — they're in grace period
            # and should not surface in context
            if fact.get("atrophy_marked_at"):
                continue
            # Activation maturation
            activation = self._trust.compute_activation(
                fact.get("created_at", _now_iso()),
                fact.get("retrieval_count", 0),
            )

            # Relevance is the COSINE similarity, not `score` — when FTS
            # matches, _rrf_fuse overwrites `score` with a rank score (~0.016-
            # 0.033, a completely different scale). Gating on `score` silently
            # dropped every highly-relevant fact on any query with keyword
            # hits. `vector_score` carries the cosine; FTS-only facts have no
            # cosine and fall back to the rank score.
            emb_score = fact.get("vector_score")
            if emb_score is None:
                emb_score = fact.get("score", 0)
            emb_score = float(emb_score or 0.0)

            # FTS-only facts (keyword hit, no vector hit) arrive with no
            # vector_score and a rank-scale `score`, which would zero them out
            # and silently drop exact keyword matches on any store big enough
            # that the vector over-fetch misses them. Compute the real cosine
            # against the query embedding so relevance stays on one scale.
            if fact.get("vector_score") is None and embedding is not None:
                stored = self._storage.get_embedding(fact.get("fact_id"))
                if stored:
                    emb_score = 1.0 - self._cosine_distance(embedding, stored)
                    emb_score = float(emb_score or 0.0)

            if activation < 0.5:
                # Silent — not surfaced normally, kept as a fallback for when
                # nothing active gates. Uses the cosine `emb_score` hoisted
                # above, NOT `score` (rank scale, ~0.016-0.033 — never reaches
                # a floor of 0.25, so this fallback could never fire).
                if emb_score >= floor:
                    fact["composite_score"] = emb_score
                    fact["activation"] = activation
                    silent_facts.append(fact)
                continue

            # Composite score — relevance is MANDATORY, trust/activation only
            # modulate it. Trust is never an independent additive term: with
            # the old additive form (emb*w + trust*w + act*w) a saturated fact
            # cleared a 0.13 floor with embedding similarity of ZERO, so stale
            # facts surfaced on any topic.
            #
            trust_weight = self._trust.get("trust_weight")

            trust_score = fact.get("trust_score", 0.5)
            # Trust modulates relevance by at most `trust_weight` (0.20):
            # full trust yields raw relevance, zero trust dampens it to 0.8x.
            composite = emb_score * (1.0 - trust_weight * (1.0 - trust_score))

            fact["composite_score"] = composite
            fact["relevance"] = emb_score  # observability
            fact["effective_trust"] = trust_score  # kept for observability/logging
            fact["activation"] = activation

            # Dominance + observability. With relevance mandatory, embedding
            # always dominates what surfaces; trust is reported as the damping
            # it actually applies, so the adaptation signal stays honest.
            emb_contribution = emb_score * (1.0 - trust_weight * (1.0 - trust_score))
            trust_contribution = emb_score * trust_weight * trust_score
            act_contribution = 0.0

            # Record component contribution for observability (rolling window of 500)
            self._component_scores["embedding"].append(emb_contribution)
            self._component_scores["trust"].append(trust_contribution)
            self._component_scores["activation"].append(act_contribution)
            for key in self._component_scores:
                if len(self._component_scores[key]) > 500:
                    self._component_scores[key] = self._component_scores[key][-500:]

            scored.append(fact)

        # Relevance floor gating
        gated = [f for f in scored if f["composite_score"] >= floor]

        # Fallback: if no active facts passed the gate, use high-relevance silent facts
        # so recall isn't empty when all matching facts are still in the maturation period.
        if not gated and silent_facts:
            silent_facts.sort(key=lambda f: f["composite_score"], reverse=True)
            budget = self._config.get("context_budget", 20)
            return silent_facts[:budget]

        # Sort by composite, take top context_budget
        budget = self._config.get("context_budget", 20)
        gated.sort(key=lambda f: f["composite_score"], reverse=True)

        return gated[:budget]

    # ── Prefetch (Thalamus → Prefrontal Cortex) ───────────────────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Called before each LLM call. Returns context block or empty string."""
        import time
        start_ts = time.perf_counter()

        if not self._initialised:
            logger.warning("limbic: prefetch skipped — not initialised")
            self._prefetch_latencies.append((time.perf_counter() - start_ts) * 1000)
            return ""

        uid = self._get_user_id(session_id)
        if uid is None:
            logger.warning("limbic: prefetch skipped — no user_id (session=%s)", session_id)
            self._prefetch_latencies.append((time.perf_counter() - start_ts) * 1000)
            return ""

        # Step 1: Evaluate trust signals from previous turn
        self._evaluate_trust_signals(query, session_id)

        # Step 2: Search
        results = self._search(query, uid, session_id=session_id, user_message=query)
        logger.info("limbic: prefetch user=%s results=%d", uid, len(results))

        # Step 3: Update retrieval metadata. NO trust boost here —
        # prefetch injection is not retrieval. Rewarding injection lets any
        # fact that surfaces once keep surfacing (rich-get-richer loop:
        # trust lifts rank → rank keeps it injected → more trust). Trust
        # moves only on real signals in _evaluate_trust_signals()
        # (continuation +0.03 / correction −0.05). retrieval_count and
        # last_retrieved still update: maturation needs the former, the
        # decay clock resets only on genuine surfacing.
        now_iso = _now_iso()

        injected_ids = []
        for fact in results:
            # Detect maturation crossing for timing recording (adaptation).
            # The matured_at stamp itself is handled by the SQLite track_maturity
            # event, but we still need to detect the crossing here to record
            # hours-to-maturity for the maturation curve adaptation engine.
            activation = fact.get("activation", 0.0)
            new_retrieval_count = fact.get("retrieval_count", 0) + 1
            if activation >= 0.5 and not fact.get("matured_at") and new_retrieval_count >= 5:
                hours_to_mature = 0.0
                created_at = fact.get("created_at")
                if created_at:
                    try:
                        if isinstance(created_at, datetime):
                            created = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
                        else:
                            created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                        now_dt = datetime.now(timezone.utc)
                        hours_to_mature = (now_dt - created).total_seconds() / 3600
                        self._trust.record_maturity(hours_to_mature)
                    except Exception:
                        pass
                logger.debug("limbic: fact %d maturing (activation=%.2f, hours=%.1f)", fact["fact_id"], activation, hours_to_mature)

            self._storage.update(
                fact["fact_id"],
                retrieval_count=new_retrieval_count,
                last_retrieved=now_iso,
            )
            injected_ids.append(fact["fact_id"])

        # Step 4: Store in session buffer for next-turn evaluation
        with self._session_lock:
            if len(self._session_buffer) >= self._MAX_SESSIONS:
                self._session_buffer.pop(next(iter(self._session_buffer)))
            self._session_buffer[session_id] = list(injected_ids)

        # Step 4b: Mark atrophy candidates — facts with trust below threshold
        # These are NOT injected facts; we check all scored facts from the search
        if self._storage:
            for fact in results:
                trust_score = fact.get("trust_score", 0.5)
                if self._trust.is_atrophy_candidate(
                    trust_score=trust_score,
                    last_retrieved=fact.get("last_retrieved"),
                ):
                    # Only mark if not already marked
                    if not fact.get("atrophy_marked_at"):
                        self._storage.mark_atrophy(fact["fact_id"])
                        self._atrophy_marked_count += 1
                        logger.debug("limbic: marked fact %d for atrophy (trust=%.3f)",
                                     fact["fact_id"], trust_score)

        # Step 5: Trust engine tick (includes atrophy sweep)
        self._trust.tick()
        self._apply_temporal_decay(uid)
        self._scan_and_sweep_gated_facts(uid)
        self._sweep_atrophy(uid)
        self._recheck_contradictions(uid)

        # Step 5b: Periodic in-process DB maintenance (replaces external cron)
        self._maybe_optimize()

        # Step 6: Format context block with identity directive
        display_name = self._session_display_name.get(session_id, uid)
        scope = self._session_scope.get(session_id, "adult")
        
        context_parts = [
            f"You're chatting with {display_name} ({scope}). Recalled user preferences shape your responses.",
            f"User-separation rule: You are talking to {display_name}. Do not reference, mention, or apply "
            f"other users' personal data (training, health, nutrition, schedule, preferences). "
            f"If recalled context or conversation history mentions another user, do not surface it "
            f"to {display_name}.",
        ]

        if results:
            for fact in results:
                # Suppress ephemeral date facts at retrieval time (defense-in-depth)
                # in case any were stored before the write-time filter was added
                if self._is_ephemeral_date_fact(fact.get("content", "")):
                    logger.info("limbic: suppressed stale date fact at retrieval (fact_id=%s)",
                                fact.get("fact_id"))
                    continue
                context_parts.append(fact["content"])

        context = "\n".join(context_parts)

        # Record latency before returning
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        self._prefetch_latencies.append(elapsed_ms)
        # Keep only last 1000 measurements (rolling window ~1 day at 50 turns)
        if len(self._prefetch_latencies) > 1000:
            self._prefetch_latencies = self._prefetch_latencies[-1000:]

        # Stash injected fact count for recall_status() UI indicator
        self._last_recall_count = max(0, len(context_parts) - 2)

        return f"<limbic-context>\n{context}\n</limbic-context>"

    def recall_status(self) -> Optional[RecallStatus]:
        """Number of facts injected in the last prefetch, for the UI indicator.

        Returns None when nothing was injected, so the indicator stays hidden
        rather than rendering "0 memories" on a turn that legitimately had none.
        """
        n = self._last_recall_count
        if n <= 0:
            return None
        return RecallStatus(provider_label="limbic", count=n)

    # ── Trust signals (Amygdala feedback) ──────────────────────────

    def _evaluate_trust_signals(self, user_message: str, session_id: str):
        """Evaluate continuation/correction signals on previously injected facts."""
        with self._session_lock:
            last_injected = self._session_buffer.get(session_id, [])

        if not last_injected:
            return

        is_correction = self._trust.is_correction(user_message)

        for entry in last_injected:
            # Handle both old format (int) and new format (dict)
            fact_id = entry.get("fact_id") if isinstance(entry, dict) else entry

            fact = self._storage.get(fact_id)
            if not fact:
                continue

            overlap = self._trust.text_overlap(fact.get("content", ""), user_message)

            if is_correction and overlap > 0.3:
                # Regex-based correction — devalue trust
                new_trust = max(0.0, fact.get("trust_score", 0.5) -
                                self._trust.correction_penalty_value())
                self._storage.update(fact_id, trust_score=new_trust)
                self._trust.record_signal(fact_id, was_continuation=False, was_correction=True)
                # Organic atrophy: evaluate the corrected fact against existing criteria
                self._check_organic_atrophy(
                    fact_id, new_trust,
                    fact.get("last_retrieved"),
                    created_at=fact.get("created_at"),
                )
            elif overlap > 0.3:
                # Continuation — reinforce trust
                new_trust = min(1.0, fact.get("trust_score", 0.5) +
                               self._trust.continuation_boost_value())
                self._storage.update(fact_id, trust_score=new_trust)
                self._trust.record_signal(fact_id, was_continuation=True, was_correction=False)

    # ── Reconsolidation (write-time NLI, in remember()) ────────────────
    #
    # Correction is handled at write time: a new fact contradicting an existing
    # one penalises the existing fact's trust. There is no read-time channel.
    # facts via vector similarity. If the user message is a correction, the mature
    # fact's trust is penalised — it fades naturally. No content is ever modified.
    # The _check_reconsolidation() and _blend_content() methods were removed.

    # ── Atrophy evaluation after trust penalties ────────────────────

    def _check_organic_atrophy(self, fact_id: int, trust_score: float,
                                last_retrieved, created_at=None):
        """Evaluate whether a fact qualifies for atrophy after a trust penalty.

        Uses the same is_atrophy_candidate() criteria and mark_atrophy() path
        as the search-results path. trust_score already includes temporal decay.
        """
        try:
            if self._trust.is_atrophy_candidate(
                trust_score=trust_score,
                last_retrieved=last_retrieved,
                created_at=created_at,
            ):
                self._storage.mark_atrophy(fact_id)
                self._atrophy_marked_count += 1
                logger.debug("limbic: organic atrophy mark for fact %d (trust=%.3f)",
                             fact_id, trust_score)
        except Exception as e:
            logger.debug("limbic: organic atrophy check failed: %s", e)

    # ── Continuing correction ──────────────────────────────────────

    def _recheck_contradictions(self, user_id: str):
        """Re-evaluate open contradiction links, bounded per turn.

        Write-time NLI is one-shot: it penalises the older fact once and stops.
        If that trust later recovers through continuations, the correction is
        undone with nothing to notice. This pass gives the correction a channel
        that persists after the write.

        Two terminal conditions, either of which resolves the link:
          - the correction landed: older fact's trust has fallen below the floor
          - the correction is stale: NLI no longer agrees the pair contradicts
        One non-terminal case re-asserts the penalty: the older fact recovered
        above the floor AND NLI still confirms the contradiction.
        """
        if not self._storage or not self._trust:
            return

        max_per_turn = int(self._trust.get("max_corrections_per_turn"))
        ttl_days = int(self._trust.get("correction_link_ttl_days"))
        floor = self._trust.get("relevance_floor")

        try:
            # Expire stale links first so the bounded query below spends its
            # budget on live pairs.
            cutoff = (datetime.now(timezone.utc) - timedelta(days=ttl_days)).isoformat()
            self._storage.delete_expired_contradictions(cutoff)

            links = self._storage.unresolved_contradictions(user_id, limit=max_per_turn)
            if not links:
                return

            reasserted = resolved = 0
            for link in links:
                older_id = link["older_fact_id"]
                older_trust = link.get("older_trust", 0.5)

                # Landed — the corrected fact has faded.
                if older_trust < floor:
                    self._storage.resolve_contradiction(link["newer_fact_id"], older_id)
                    resolved += 1
                    continue

                # Still contradictory? Only pay for NLI when the older fact has
                # recovered enough that the penalty is being outvoted.
                nli_label = "neutral"
                if self._nli and link.get("newer_content") and link.get("older_content"):
                    nli_label = self._nli.classify(link["older_content"], link["newer_content"])

                if nli_label != "contradiction":
                    # No longer true (the older fact was itself corrected, or the
                    # newer fact is stale). Stop tracking it.
                    self._storage.resolve_contradiction(link["newer_fact_id"], older_id)
                    resolved += 1
                    continue

                # Contradiction persists and the penalty has been outvoted —
                # re-assert it. Bounded: one trust write per link.
                penalty = self._trust.correction_penalty_value()
                new_trust = max(0.0, older_trust - penalty)
                self._storage.update(older_id, trust_score=new_trust)
                self._check_organic_atrophy(
                    older_id, new_trust,
                    None,
                    created_at=link.get("older_created_at"),
                )
                reasserted += 1
                logger.info(
                    "limbic: continuing correction — re-asserted fact %d "
                    "(trust %.2f → %.2f, contradicted by fact %d)",
                    older_id, older_trust, new_trust, link["newer_fact_id"],
                )

            if reasserted or resolved:
                logger.debug("limbic: contradiction recheck — %d re-asserted, %d resolved",
                             reasserted, resolved)
        except Exception as e:
            logger.warning("limbic: contradiction recheck failed: %s", e)

    # ── Atrophy sweep (gradual organic removal) ────────────────────

    def _apply_temporal_decay(self, user_id: str):
        """Apply Ebbinghaus temporal decay to all facts.

        Writes the decayed trust_score directly to the database. This runs
        every turn before the background scan, so trust_score always reflects
        both interaction history AND time-based forgetting.
        """
        if not self._storage or not self._trust:
            return
        try:
            # Fetch all facts with their timestamps
            rows = self._storage.decay_rows(user_id)

            decayed = 0
            for row in rows:
                old_trust = row.get("trust_score", 0.5)
                last_retrieved = row.get("last_retrieved")
                created_at = row.get("created_at")

                # decayed_at (the row's updated_at) is the previous decay
                # pass. Without it the anchor never advances and every turn
                # re-bills the whole elapsed interval — see apply_decay.
                new_trust = self._trust.apply_decay(
                    old_trust, last_retrieved, created_at=created_at,
                    decayed_at=row.get("updated_at"),
                )

                # Only write if the value actually changed (avoid unnecessary DB writes)
                # Do NOT stamp last_retrieved here — decay is a metabolic process,
                # not a retrieval. The decay clock should only reset when a fact
                # actually surfaces in search results (real retrieval = reactivation).
                if abs(new_trust - old_trust) > 0.001:
                    self._storage.update(
                        row["fact_id"],
                        trust_score=new_trust,
                    )
                    decayed += 1

            if decayed:
                logger.debug("limbic: applied temporal decay to %d facts (user=%s)", decayed, user_id)
        except Exception as e:
            logger.debug("limbic: temporal decay failed: %s", e)

    def _scan_and_sweep_gated_facts(self, user_id: str):
        """Background scan: find facts below the trust threshold, mark them
        for atrophy, and let the sweep handle deletion. trust_score already
        includes temporal decay (applied by _apply_temporal_decay), so this
        query catches both correction-driven and time-driven decay.
        """
        if not self._storage or not self._trust:
            return
        try:
            threshold = self._trust.atrophy_threshold()
            max_per_turn = self._trust.max_atrophy_per_turn()

            # Facts below threshold, not yet marked
            rows = self._storage.atrophy_scan_rows(
                user_id, threshold, max_per_turn
            )

            marked = 0
            for row in rows:
                trust = row.get("trust_score", 0.5)
                # NOTE: created_at must be passed by KEYWORD only. Passing it
                # positionally as well raises TypeError, which the except arm
                # below swallowed at DEBUG level — so the background scan
                # silently never marked anything. The two foreground paths
                # (search results, post-penalty) always used keywords.
                if self._trust.is_atrophy_candidate(
                    trust_score=trust,
                    last_retrieved=row.get("last_retrieved"),
                    created_at=row.get("created_at"),
                ):
                    self._storage.mark_atrophy(row["fact_id"])
                    marked += 1
                    logger.debug(
                        "limbic: background scan marked fact %d for atrophy (trust=%.3f)",
                        row["fact_id"], trust
                    )

            if marked:
                self._atrophy_marked_count += marked
                logger.info("limbic: background scan marked %d facts for atrophy (user=%s)", marked, user_id)
        except Exception as e:
            logger.debug("limbic: background atrophy scan failed: %s", e)

    def _sweep_atrophy(self, user_id: str):
        """Sweep facts marked for atrophy whose grace period has elapsed.

        Runs every turn (via tick() → prefetch). Bounded by max_atrophy_per_turn.
        Facts are marked during search when their effective trust decays below
        threshold and their age exceeds the horizon. After a grace period,
        they are physically deleted from the database.
        """
        if not self._storage:
            return
        try:
            grace_days = self._trust.atrophy_grace_days()
            max_per_turn = self._trust.max_atrophy_per_turn()

            # Compute cutoff ISO: now - grace_days
            from datetime import datetime, timedelta, timezone
            cutoff = datetime.now(timezone.utc) - timedelta(days=grace_days)
            cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

            deleted = self._storage.sweep_atrophy(cutoff_iso, limit=max_per_turn, user_id=user_id)
            if deleted:
                self._atrophy_swept_count += deleted
                logger.info("limbic: atrophied %d facts for user=%s", deleted, user_id)
        except Exception as e:
            logger.debug("limbic: atrophy sweep failed: %s", e)

    # ── In-process DB maintenance ──────────────────────────────────

    _OPTIMIZE_INTERVAL_TURNS = 200   # PRAGMA optimize every ~200 prefetch calls
    _VACUUM_INTERVAL_S = 86400       # VACUUM at most once per 24h
    _OPTIMIZE_INTERVAL_S = 3600      # PRAGMA optimize at most once per 1h (turn gate is the primary)

    def _maybe_optimize(self) -> None:
        """Periodic SQLite maintenance — PRAGMA optimize + VACUUM.

        Runs inside prefetch() after the atrophy sweep, so it fires when
        the database has just been modified (facts deleted by atrophy).
        Replaces the external cron job (optimize_limbic.py).

        PRAGMA optimize: updates query planner stats. Fast, no exclusive
        lock needed. Gated by a turn counter (~every 200 turns).

        VACUUM: reclaims free pages from atrophied fact deletions. Needs
        a brief exclusive write lock. Gated by a 24h timestamp. If the
        gateway is mid-write, VACUUM is skipped (caught OperationalError).

        Both are no-ops if the storage backend isn't SQLite or is unset.
        """
        if not self._storage or not hasattr(self._storage, '_db'):
            return
        now = time.time()
        self._prefetch_count += 1

        # PRAGMA optimize — turn-gated (every ~200 turns) + time-gated (1h min)
        if (
            self._prefetch_count % self._OPTIMIZE_INTERVAL_TURNS == 0
            and (now - self._last_optimize_ts) >= self._OPTIMIZE_INTERVAL_S
        ):
            try:
                self._storage._db.execute("PRAGMA optimize")
                self._last_optimize_ts = now
                logger.debug("limbic: PRAGMA optimize (turn %d)", self._prefetch_count)
            except Exception as e:
                logger.debug("limbic: PRAGMA optimize failed: %s", e)

        # VACUUM — time-gated (once per 24h). Only runs if at least one
        # atrophy sweep has deleted facts since the last VACUUM (checked
        # via freelist_count — if 0 free pages, there's nothing to reclaim).
        if (now - self._last_vacuum_ts) >= self._VACUUM_INTERVAL_S:
            try:
                # Check if there are free pages to reclaim
                freelist = self._storage._db.execute(
                    "PRAGMA freelist_count"
                ).fetchone()[0]
                if freelist > 0:
                    self._storage._db.execute("VACUUM")
                    self._last_vacuum_ts = now
                    logger.info(
                        "limbic: VACUUM reclaimed %d free pages (turn %d)",
                        freelist, self._prefetch_count,
                    )
                else:
                    # No free pages — stamp the time so we don't check every turn
                    self._last_vacuum_ts = now
            except Exception as e:
                # VACUUM needs an exclusive lock; if the gateway is mid-write,
                # this will fail. Not an error — we'll try again next window.
                logger.debug("limbic: VACUUM skipped: %s", e)
                self._last_vacuum_ts = now  # don't retry every turn

    # ── Baseline seeding ───────────────────────────────────────────

    def _maybe_seed(self, user_id: str, user_profile: dict):
        """Seed baseline identity facts if none exist for this user."""
        existing = self._storage.count(user_id)
        if existing > 0:
            return  # already has data

        seed_fields = [
            ("canonical_name", "User canonical name: {value}"),
            ("display_name", "User display name: {value}"),
            ("scope", "User scope: {value}"),
            ("role", "User role: {value}"),
            ("email", "User email: {value}"),
        ]

        for field, template in seed_fields:
            value = user_profile.get(field)
            if value:
                content = template.format(value=value)
                self.remember(content, user_id=user_id)

        logger.info("limbic: seeded baseline identity for user=%s (%d facts)", user_id, len(seed_fields))

    # ── Operator functions ─────────────────────────────────────────

    def forget(self, user_id: str) -> dict:
        """Right to be forgotten. Removes all data for a user."""
        if not self._storage:
            return {"error": "Not initialised"}
        removed = self._storage.delete_user(user_id)
        return {"removed": removed, "status": "complete"}

    def forget_fact(self, fact_id: int, user_id: str | None = None) -> dict:
        """Remove a single fact by ID. user_id is REQUIRED (CLI resolves it).

        None user_id is only legal for operator CLI runs (root), never from
        an agent session — a factless delete-all path must not exist.
        """
        if not self._storage:
            return {"error": "Not initialised"}
        if user_id is None:
            # Trust boundary: refuse unscoped delete from non-CLI paths.
            # The CLI passes an explicit user; agent code never calls this.
            logger.warning("limbic: forget_fact called without user_id — refusing")
            return {"error": "user_id required for forget_fact"}
        fact = self._storage.get(fact_id)
        if not fact or fact.get("user_id") != user_id:
            return {"error": "fact not found for user", "fact_id": fact_id}
        ok = self._storage.delete_fact(fact_id)
        return {"removed": ok, "fact_id": fact_id}

    def export(self, user_id: str) -> dict:
        """Data export. Returns all facts for a user."""
        if not self._storage:
            return {"error": "Not initialised"}
        return self._storage.export_user(user_id)

    def seed(self, user_id: str, content: str) -> dict:
        """Operator-initiated seeding without conversation."""
        return self.remember(content, user_id=user_id)

    def stats(self, user_id: str | None = None) -> dict:
        """Analytical self-evaluation — query the store for performance metrics.
        
        Returns per-user statistics: fact count, average trust,
        matured count, faded count (low trust). If user_id is None, returns
        aggregate stats across all users.
        """
        if not self._storage:
            return {"error": "Not initialised"}

        try:
            if user_id:
                # Per-user stats — use separate queries for reliability
                total = self._storage.count(user_id)
                
                avg_trust = self._storage.avg_trust(user_id)
                matured = self._storage.count_matured(user_id)
                faded = self._storage.count_faded(user_id, 0.3)

                # Atrophy metrics
                atrophy_marked = self._storage.count_atrophy_candidates()
                
                return {
                    "user_id": user_id,
                    "total_facts": total,
                    "avg_trust": round(avg_trust, 3),
                    "matured": matured,
                    "faded": faded,
                    "atrophy_marked": atrophy_marked,
                    "atrophy_swept": self._atrophy_swept_count,
                }
            else:
                # Aggregate stats across all users
                users = self._storage.facts_per_user()
                per_user = []
                for u in users or []:
                    uid = u.get("user_id", "unknown")
                    count = u.get("facts", 0)
                    per_user.append({"user_id": uid, "facts": count})

                total = self._storage.count()

                # Telemetry summary
                latencies = self._prefetch_latencies
                prefetch_avg = round(sum(latencies) / len(latencies), 2) if latencies else 0
                prefetch_max = round(max(latencies), 2) if latencies else 0
                prefetch_count = len(latencies)

                embed_total = self._embed_cache_hits + self._embed_cache_misses
                embed_hit_rate = round(self._embed_cache_hits / embed_total, 3) if embed_total > 0 else 0

                centroid_total = self._centroid_cache_hits + self._centroid_cache_misses
                centroid_hit_rate = round(self._centroid_cache_hits / centroid_total, 3) if centroid_total > 0 else 0

                # Component dominance (last 500 scores)
                def _avg(l):
                    return round(sum(l) / len(l), 4) if l else 0
                component_dominance = {
                    k: _avg(v) for k, v in self._component_scores.items()
                }

                return {
                    "total_facts": total,
                    "users": per_user,
                    "telemetry": {
                        "prefetch_calls": prefetch_count,
                        "prefetch_avg_ms": prefetch_avg,
                        "prefetch_max_ms": prefetch_max,
                        "embed_cache_hit_rate": embed_hit_rate,
                        "embed_cache_hits": self._embed_cache_hits,
                        "embed_cache_misses": self._embed_cache_misses,
                        "centroid_cache_hit_rate": centroid_hit_rate,
                        "centroid_cache_hits": self._centroid_cache_hits,
                        "centroid_cache_misses": self._centroid_cache_misses,
                        "atrophy_marked_count": self._atrophy_marked_count,
                        "atrophy_swept_count": self._atrophy_swept_count,
                        "component_dominance": component_dominance,
                    }
                }
        except Exception as e:
            logger.debug("limbic: stats query failed: %s", e)
            return {"error": str(e)}

    # ── Hook handlers ──────────────────────────────────────────────

    def on_pre_llm_call(self, event, **kwargs):
        """Prefetch — inject context before LLM call."""
        session_id = kwargs.get("session_id", "")
        user_message = ""

        # Extract user message from event
        if hasattr(event, "messages") and event.messages:
            for msg in reversed(event.messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
        elif isinstance(event, dict):
            messages = event.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break

        if not user_message:
            return

        context = self.prefetch(user_message, session_id=session_id)
        if context:
            # Inject into system prompt or context
            if hasattr(event, "system_prompt"):
                event.system_prompt = (event.system_prompt or "") + "\n\n" + context
            elif isinstance(event, dict) and "system_prompt" in event:
                event["system_prompt"] = (event.get("system_prompt") or "") + "\n\n" + context

    def on_post_tool_call(self, event, **kwargs):
        """Track tool calls for trust signals."""
        pass  # Trust signals evaluated on next turn's prefetch

    def on_session_start(self, event, **kwargs):
        """Session start — already handled by initialize()."""
        pass

    def on_session_end(self, messages: list | None = None) -> None:
        """Session end — cleanup. No-op for Limbic."""
        pass

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Extract salient facts before context compression discards them.

        Scans messages about to be compressed for content worth remembering.
        Auto-stores facts via remember() so they survive the compression.
        Returns empty string — does not contribute to compression summary.
        """
        if not self._initialised or not messages:
            return ""

        uid = self._get_user_id("")
        if not uid or uid == "general":
            return ""

        # Extract user and assistant utterances from messages about to be compressed
        extracted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content or len(content) < 20:
                continue
            if role == "user":
                # Skip commands, URLs, one-word responses
                if content.startswith(("/", "http", "!")) or len(content.split()) < 3:
                    continue
                extracted.append(content)
            elif role == "assistant" and msg.get("tool_calls"):
                # Assistant tool results may contain valuable findings
                for tc in msg["tool_calls"]:
                    result = tc.get("result", "")
                    if result and len(result) > 50:
                        extracted.append(result[:500])  # cap length

        for item in extracted[:3]:  # limit auto-extraction to 3 per compression
            # Skip if too similar to existing facts (light dedup)
            try:
                existing = self._storage.search(
                    embedding=self._embedder.embed(item),
                    filters={"user_id": uid},
                    limit=1,
                )
                if existing and existing[0].get("score", 0) > 0.80:
                    continue  # near-duplicate, skip
            except Exception:
                pass
            self.remember(item, user_id=uid)

        return ""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror built-in memory writes to Limbic.

        Safety net: if anything bypasses the disabled flag and writes to
        built-in memory, we capture it in Limbic too.
        """
        if not self._initialised or not content:
            return

        uid = self._get_user_id("")
        if not uid:
            uid = metadata.get("user_id", "general") if metadata else "general"

        if action in ("add", "replace"):
            self.remember(content, user_id=uid)
        elif action == "remove":
            # Best-effort delete — was a no-op before 2026-08-29, which is how
            # failed cleanups degraded into "Correction: ..." garbage facts.
            if self._storage:
                deleted = self._storage.delete_fact_by_content(content, uid)
                if deleted:
                    logger.info("limbic: memory-write mirror removed one fact for user=%s", uid)

    # ── Tool dispatch ──────────────────────────────────────────────

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        """Called when agent invokes remember or recall."""
        from .tools import handle_tool_call as dispatch
        session_id = kwargs.get("session_id", "")
        return dispatch(self, tool_name, args, session_id=session_id)

    def get_tool_schemas(self) -> list[dict]:
        """Return tool schemas for the agent."""
        from .tools import REMEMBER_SCHEMA, RECALL_SCHEMA
        return [REMEMBER_SCHEMA, RECALL_SCHEMA]

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: list | None = None) -> None:
        """Called after each turn. Non-blocking — no-op for Limbic.
        Trust signals are evaluated on the NEXT turn's prefetch."""
        pass

    def system_prompt_block(self) -> str:
        """Static provider info for system prompt.

        This block replaces the built-in MEMORY_GUIDANCE (which references the
        disabled ``memory`` tool).  It mirrors the same guidance content but
        points the agent at Limbic's ``remember``/``recall`` tools instead.

        Note: the built-in MEMORY_GUIDANCE is still injected by core
        (``system_prompt.py`` line 228) because ``memory_enabled: false`` does
        not suppress the ``memory`` tool from ``agent.valid_tool_names``.  Both
        blocks appear in the system prompt — this one in volatile_parts (later)
        and the built-in in stable_parts (earlier).  By matching the built-in
        guidance content here and explicitly naming ``remember()``, the agent
        is steered toward the correct tool.  A future upstream fix should skip
        MEMORY_GUIDANCE when a memory provider is registered.
        """
        return (
            "You have persistent memory via Limbic (SQLite-backed). "
            "Save durable facts using the `remember(content)` tool. "
            "Search prior context using the `recall(query)` tool. "
            "Memory is injected into every turn, so keep it compact and focused on facts that "
            "will still matter later.\n"
            "Prioritize what reduces future user steering — the most valuable memory is one "
            "that prevents the user from having to correct or remind you again. "
            "User preferences and recurring corrections matter more than procedural task details.\n"
            "Do NOT save task progress, session outcomes, completed-work logs, or temporary TODO "
            "state to memory; use session_search to recall those from past transcripts. "
            "Specifically: do not record PR numbers, issue numbers, commit SHAs, 'fixed bug X', "
            "'submitted PR Y', 'Phase N done', file counts, or any artifact that will be stale "
            "in 7 days. If a fact will be stale in a week, it does not belong in memory. "
            "If you've discovered a new way to do something, solved a problem that could be "
            "necessary later, save it as a skill with the skill tool.\n"
            "Write memories as declarative facts, not instructions to yourself. "
            "'User prefers concise responses' ✓ — 'Always respond concisely' ✗. "
            "'Project uses pytest with xdist' ✓ — 'Run tests with pytest -n 4' ✗. "
            "Imperative phrasing gets re-read as a directive in later sessions and can "
            "cause repeated work or override the user's current request. Procedures and "
            "workflows belong in skills, not memory."
        )

    def get_config_schema(self) -> list[dict[str, Any]]:
        """Return config fields for `hermes memory setup`.

        HONEST CONTRACT: embedding provider/model are NOT configurable —
        the bundled Arctic Embed 2.0 L ONNX model is the only embedder
        (embeddings.py: create_embedder has no endpoint option). Advertise
        only what the code actually reads.
        """
        return [
            {
                "key": "db_path",
                "description": "Path to SQLite database file (supports $HERMES_HOME). Leave default for auto.",
                "default": "$HERMES_HOME/limbic.db",
            },
            {
                "key": "embedding_dims",
                "description": "Embedding dimensionality. Must match the bundled model (1024).",
                "default": 1024,
            },
            {
                "key": "context_budget",
                "description": "Max facts injected per turn (ceiling, not target).",
                "default": 20,
            },
            {
                "key": "nlp_languages",
                "description": "spaCy language models for entity extraction (comma-separated, e.g. en,fr).",
                "default": "en",
            },
            {
                "key": "known_users",
                "description": "Canonical user names this deployment recognises (comma-separated). Deployments configure; plugin default is empty.",
                "default": "",
            },
            {
                "key": "known_entities",
                "description": "Extra entity patterns as 'LABEL:pattern' strings, comma-separated (e.g. 'PERSON:Alice,ORG:Acme Corp'). Deployments configure; plugin default is empty.",
                "default": "",
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """Write non-secret config to $HERMES_HOME/limbic.yaml."""
        import yaml as _yaml

        config_path = os.path.join(hermes_home, "limbic.yaml")
        config: dict[str, Any] = {}

        # Map flat keys back to nested structure
        if "db_path" in values:
            config["db_path"] = values["db_path"]
        if any(k in values for k in ("embedding_provider", "embedding_model", "embedding_dims")):
            config["embedding"] = {}
            if "embedding_provider" in values:
                config["embedding"]["provider"] = values["embedding_provider"]
            if "embedding_model" in values:
                config["embedding"]["model"] = values["embedding_model"]
            if "embedding_dims" in values:
                config["embedding"]["dims"] = values["embedding_dims"]
        if "context_budget" in values:
            config["context_budget"] = values["context_budget"]
        if "known_users" in values:
            config["known_users"] = [u.strip() for u in str(values["known_users"]).split(",") if u.strip()]
        if "known_entities" in values:
            config["known_entities"] = [e.strip() for e in str(values["known_entities"]).split(",") if e.strip()]
        if "nlp_languages" in values:
            langs = [l.strip() for l in str(values["nlp_languages"]).split(",") if l.strip()]
            config["nlp"] = {"languages": langs}

        # Merge with existing config so we don't clobber untouched fields
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    existing = _yaml.safe_load(f) or {}
                if existing:
                    # Deep-merge: nested dicts get merged, top-level keys overwritten
                    def _deep_merge(base: dict, overlay: dict) -> dict:
                        for k, v in overlay.items():
                            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                                base[k] = _deep_merge(base[k], v)
                            else:
                                base[k] = v
                        return base
                    config = _deep_merge(existing, config)
            except Exception:
                pass

        with open(config_path, "w") as f:
            _yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("limbic: saved config to %s", config_path)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue background recall. No-op — Limbic does synchronous prefetch."""
        pass

    def backup_paths(self) -> list[str]:
        """Paths to include in hermes backup. Delegates to the storage layer, which
        snapshots via VACUUM INTO (the live .db + WAL sidecars must never be copied
        file-by-file while running — torn backups)."""
        if self._storage and hasattr(self._storage, 'backup_paths'):
            try:
                return list(self._storage.backup_paths())
            except Exception as e:
                logger.warning("limbic: backup snapshot failed: %s", e)
                return []
        return []

    # ── Helpers ────────────────────────────────────────────────────

    def _get_user_id(self, session_id: str) -> str | None:
        # 2026-09-04 user-scoping audit: the gateway sets a per-message contextvar
        # holding the CURRENT turn's speaking user. Cached agents and shared chats
        # made the instance map + _current_user_id stale (user B's recall resolving
        # to user A). The contextvar is authoritative when present; the map is the
        # fallback for non-gateway contexts (CLI, cron, tests).
        try:
            from gateway.session_context import get_session_env
            raw = get_session_env("HERMES_SESSION_USER_ID", "").strip()
        except Exception:
            raw = ""
        if raw:
            canonical, _, _, _ = self._resolve_identity(raw, {})
            if canonical:
                with self._session_lock:
                    self._current_user_id = canonical
                return canonical
        with self._session_lock:
            # Try session_id first, then fallback to current_user_id
            # (prefetch may pass session_id="" — use the last initialised user)
            uid = self._session_user_id.get(session_id)
            if uid:
                return uid
            return getattr(self, "_current_user_id", "general")

    def _detect_subject_user(self, content: str, speaker_id: str) -> str | None:
        """Detect if a fact is ABOUT another user, not about the speaker.
        
        Guardian case: "Bob is interested in Harry Potter" 
        → speaker is alice, but fact is about bob → store under bob.
        
        Heuristic: if the content starts with or prominently mentions 
        another known user's name as the subject, attribute to them.
        """
        if not content or speaker_id == "general":
            return None
        
        # Load known users — manifest first, storage scan as fallback
        known_users = self._get_known_users()
        if not known_users:
            return None
        
        content_lower = content.lower().strip()
        
        # 2026-09-04 user-scoping audit: only a GUARDIAN may write facts into
        # another user's scope. Without this, any speaker re-attributes by
        # naming a subject ("Alice is..." pollutes alice's store).
        speaker_guardian_of: list[str] = []
        manifest = getattr(self, "_manifest", None)
        if manifest is not None:
            ident = manifest.resolve_by_canonical(speaker_id)
            if ident:
                speaker_guardian_of = ident.guardian_of
        # else: no manifest → single-user mode, no guardian re-attribution
        
        # Check if content starts with another user's name as subject
        # "Bob is interested in..." → subject is Bob
        # "Carol prefers..." → subject is Carol
        for user_name in known_users:
            if user_name == speaker_id:
                continue
            # Guardian gate: skip subjects the speaker cannot access
            if user_name.lower() not in speaker_guardian_of:
                continue
            
            # Check patterns: "<Name> is/has/likes/prefers/follows/exhibits/struggles/requires..."
            # or "Bob's <something>"
            name_lower = user_name.lower()
            patterns = [
                f"{name_lower} is ",
                f"{name_lower}'s ",
                f"{name_lower} has ",
                f"{name_lower} likes ",
                f"{name_lower} prefers ",
                f"{name_lower} follows ",
                f"{name_lower} exhibits ",
                f"{name_lower} struggles ",
                f"{name_lower} requires ",
                f"{name_lower} needs ",
                f"{name_lower} was born",
                f"{name_lower} identified as",
            ]
            
            for pattern in patterns:
                if content_lower.startswith(pattern):
                    return user_name
        
        return None

    def _get_known_users(self) -> list[str]:
        """Get list of known canonical user names — manifest or storage scan."""
        # Manifest first (users.yaml — Limbic's own config)
        manifest = getattr(self, "_manifest", None)
        if manifest is not None:
            return manifest.canonical_names
        
        # Fallback: scan storage for existing user_ids
        try:
            if self._storage:
                rows = self._storage.list_user_ids()
                if rows:
                    return [u.lower() for u in rows]
        except Exception:
            pass
        
        return []

    def _resolve_identity(self, platform_user_id: str, kwargs: dict) -> tuple:
        """Resolve user identity from the host identity system.

        Resolution order:
          1. built-in user manifest (users.yaml — Limbic multi-user mode)
          2. Matrix homeserver-suffix heuristic (raw ID fallback)
          3. raw platform_user_id as-is

        Returns: (canonical_user_id, guardian_of, display_name, scope)
        """
        if not platform_user_id or platform_user_id == "general":
            return "general", [], "general", "adult"

        # 1. Built-in manifest (users.yaml) — Limbic is the identity source
        manifest = getattr(self, "_manifest", None)
        if manifest is not None:
            platform = kwargs.get("platform", "")
            ident = manifest.resolve(platform_user_id, platform)
            if ident:
                return ident.canonical, ident.guardian_of, ident.display_name, ident.scope
            # Known manifest, unknown person — fall through with a warning
            logger.warning(
                "limbic: user %s not in users.yaml (%d known users) — "
                "attributing to raw ID", platform_user_id, manifest.user_count)

        # 2. Single-user-mode guard: second distinct ID without a manifest
        seen = getattr(self, "_seen_user_ids", None)
        if seen is not None:
            seen.add(platform_user_id)
            if len(seen) > 1:
                logger.warning(
                    "limbic: MULTIPLE user IDs seen (%d) with no users.yaml — "
                    "single-user mode is misconfigured; scopes and guardian "
                    "semantics inactive; define users.yaml for multi-user mode",
                    len(seen))

        # 3. Matrix identity fallback — full lowercased MXID as the bucket.
        # Never strip the homeserver or trailing digits: @alice:hs1 and @alice42:evil.org
        # are DIFFERENT people and must not share a memory store.
        if platform_user_id.startswith("@") and ":" in platform_user_id:
            mxid = platform_user_id.lower()
            local_part = mxid.split(":")[0].lstrip("@")
            display = local_part  # display hint from the local part only (cosmetic)
            return mxid, [], display.capitalize(), "adult"

        # Last resort — use as-is
        return platform_user_id.lower(), [], platform_user_id, "adult"

    def shutdown(self) -> None:
        if self._storage:
            self._storage.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()