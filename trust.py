"""Limbic trust engine — the amygdala.

Tags memories with salience. Self-tuning through implicit interaction signals.

Trust is not just "is this fact true" — it's "does this fact matter."
The amygdala modulates through:
  - Retrieval boost (fact was useful, reinforce)
  - Continuation signal (user stayed on topic, confirming relevance)
  - Correction signal (user said "no", devalue)
  - Temporal decay (hasn't been retrieved, let it fade)

All parameters are adaptive — they drift from defaults toward values
that produce better recall quality. Nobody configures retrieval_boost.
The system discovers what works.
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default parameters — seeded on first run, stored in params file
DEFAULT_PARAMS = {
    "trust_weight":         {"value": 0.20, "min": 0.05, "max": 0.40},
    # Continuing correction — bounded re-check of open contradiction links.
    "max_corrections_per_turn": {"value": 3,  "min": 1,  "max": 10},
    "correction_link_ttl_days": {"value": 30, "min": 7,  "max": 90},
    "decay_half_life_days": {"value": 30,   "min": 7,    "max": 90},   # Ebbinghaus forgetting curve — half-life in days
    "retrieval_boost":      {"value": 0.02, "min": 0.005, "max": 0.05},
    "continuation_boost":   {"value": 0.03, "min": 0.01, "max": 0.08},
    "correction_penalty":   {"value": 0.15, "min": 0.01, "max": 0.20},
    "maturation_t_half":    {"value": 168,  "min": 24,    "max": 336},  # hours
    "maturation_k":         {"value": 48,   "min": 12,    "max": 96},
    "relevance_floor":      {"value": 0.25, "min": 0.15, "max": 0.60},
    # Recall-confidence gate (no_confident_match abstention): if the best fact's
    # relevance (top cosine) is below this, recall() reports an abstention
    # instead of surfacing a near-miss. Calibrated 2026-10-10 on the live
    # phoenix store (429 facts): answerable top-cosine 0.63-0.95,
    # unanswerable 0.15-0.30 — 0.40 sits in a 0.33-wide gap.
    "recall_confidence_floor": {"value": 0.40, "min": 0.30, "max": 0.60},
    "interference_penalty": {"value": 0.20, "min": 0.02, "max": 0.25},
    "surprise_boost":       {"value": 0.10, "min": 0.02, "max": 0.20},
    "surprise_percentile":  {"value": 0.90, "min": 0.80, "max": 0.99},
    # Atrophy — facts are swept only when trust has fully decayed to zero
    "atrophy_threshold":           {"value": 0.01, "min": 0.001, "max": 0.20},
    "atrophy_horizon_multiplier":  {"value": 0,    "min": 0,    "max": 10},  # 0 = no horizon check, trust-only
    "atrophy_grace_days":          {"value": 0,    "min": 0,    "max": 30},  # 0 = no grace, lifetime was the grace
    "max_atrophy_per_turn":        {"value": 10,   "min": 1,    "max": 20},
}

# Correction patterns — regex for detecting user corrections
CORRECTION_PATTERNS = [
    r'\b(no|wrong|actually|that\'s not|incorrect|don\'t|isn\'t|wasn\'t)\b',
    r'\b(revised|updated|changed|now)\b',
    r'\b(not anymore|used to|previously)\b',
]


class TrustEngine:
    """Adaptive trust scoring engine. All parameters self-tune."""

    def __init__(self, params_path: str):
        self._params_path = params_path
        self._params: dict[str, dict] = {}
        self._turn_count = 0
        self._adaptation_interval = 50
        # Signal tracking for adaptation
        self._signals: list[dict] = []
        # Maturity timing tracking for maturation curve adaptation
        self._maturity_times: list[float] = []
        self._load_params()

    def _load_params(self):
        """Load params from file, or seed defaults on first run.

        Values are coerced to float and clamped — a hand-edited or torn
        params file must never break the engine downstream (a str value
        would TypeError in the relevance-floor compare for every recall).
        Corrupt entries fall back to their default rather than reseeding
        the whole file (a torn write must not destroy all learned state).
        """
        loaded: dict[str, dict] = {}
        if os.path.exists(self._params_path):
            try:
                with open(self._params_path) as f:
                    loaded = json.load(f)
            except Exception as e:
                logger.warning("limbic: params file unreadable (%s) — keeping defaults where unknown: %s",
                               self._params_path, e)
        params: dict[str, dict] = {}
        for key, default in DEFAULT_PARAMS.items():
            entry = loaded.get(key)
            value = default["value"]
            if isinstance(entry, dict):
                try:
                    value = float(entry.get("value", default["value"]))
                except (TypeError, ValueError):
                    logger.warning("limbic: param %s has non-numeric value %r — using default", key, entry.get("value"))
                    value = default["value"]
            # Clamp against the entry's own envelope when it supplies one,
            # else the default. decay_half_life_days=0 is the documented
            # kill switch — a file that persists 0 must keep it (the default
            # min of 7 would silently re-enable decay).
            min_b = default["min"]
            max_b = default["max"]
            if isinstance(entry, dict):
                try:
                    min_b = float(entry.get("min", default["min"]))
                    max_b = float(entry.get("max", default["max"]))
                except (TypeError, ValueError):
                    pass
            clamped = max(min_b, min(max_b, value))
            params[key] = {**default, "value": clamped, "min": min_b, "max": max_b}
        self._params = params
        if params != loaded:
            self._save_params()
        if not loaded:
            logger.info("limbic: seeded default adaptive parameters at %s", self._params_path)

    def _save_params(self):
        """Atomic write — a torn write would silently destroy all learned adaptation."""
        os.makedirs(os.path.dirname(self._params_path) or ".", exist_ok=True)
        tmp = self._params_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._params, f, indent=2)
        os.replace(tmp, self._params_path)

    def get(self, key: str) -> float:
        entry = self._params.get(key)
        if entry is None:
            entry = DEFAULT_PARAMS.get(key, {"value": 0.0, "min": 0.0, "max": 1.0})
        return entry.get("value", 0.0)

    def _set(self, key: str, value: float):
        entry = self._params.get(key, DEFAULT_PARAMS.get(key, {}))
        min_b = entry.get("min", 0.0)
        max_b = entry.get("max", 1.0)
        clamped = max(min_b, min(max_b, value))
        self._params[key] = {**entry, "value": clamped}
        self._save_params()

    # ── Trust adjustments ──────────────────────────────────────────

    def retrieval_boost_value(self) -> float:
        return self.get("retrieval_boost")

    def continuation_boost_value(self) -> float:
        return self.get("continuation_boost")

    def correction_penalty_value(self) -> float:
        return self.get("correction_penalty")

    def interference_penalty_value(self) -> float:
        return self.get("interference_penalty")

    def surprise_boost_value(self) -> float:
        return self.get("surprise_boost")

    def surprise_percentile_value(self) -> float:
        return self.get("surprise_percentile")

    # ── Atrophy helpers ────────────────────────────────────────────

    def atrophy_threshold(self) -> float:
        return self.get("atrophy_threshold")

    def atrophy_horizon_multiplier(self) -> int:
        return int(self.get("atrophy_horizon_multiplier"))

    def atrophy_grace_days(self) -> int:
        return int(self.get("atrophy_grace_days"))

    def max_atrophy_per_turn(self) -> int:
        return int(self.get("max_atrophy_per_turn"))

    def is_atrophy_candidate(self, trust_score: float, last_retrieved,
                             created_at=None) -> bool:
        """Determine if a fact should be marked for atrophy.

        A fact is a candidate when trust_score is below atrophy_threshold.

        With atrophy_horizon_multiplier = 0 (default), trust alone determines
        candidacy — no time gate. trust_score already includes temporal decay
        (applied as a write operation), so it reflects both interaction history
        AND time-based forgetting.
        """
        if trust_score >= self.atrophy_threshold():
            return False

        horizon_mult = self.atrophy_horizon_multiplier()
        if horizon_mult <= 0:
            return True  # trust-only, no horizon check

        half_life = self.get("decay_half_life_days")
        horizon = horizon_mult * half_life
        if horizon <= 0:
            return True

        try:
            ref = last_retrieved
            if ref is None and created_at is not None:
                ref = created_at
            if isinstance(ref, datetime):
                last = ref if ref.tzinfo else ref.replace(tzinfo=timezone.utc)
            elif isinstance(ref, str):
                last = datetime.fromisoformat(ref.replace("Z", "+00:00"))
            else:
                return True
            days_since = (datetime.now(timezone.utc) - last).days
            return days_since >= horizon
        except Exception:
            return True

    # ── Temporal decay (write operation, not read-time computation) ──────

    def apply_decay(self, trust_score: float, last_retrieved,
                    created_at=None, decayed_at=None) -> float:
        """Apply Ebbinghaus temporal decay to trust_score and return the new value.

        This is a WRITE operation — the caller should persist the returned value
        to trust_score in the database.

        The anchor is the MOST RECENT of last_retrieved (real reinforcement),
        created_at (orphans), and decayed_at (the previous decay pass). Using
        the latest of these is what makes the operation idempotent: a second
        call at the same instant finds the anchor already advanced and charges
        zero further decay.

        This matters because decay runs every turn. Anchoring solely on
        last_retrieved re-billed the ENTIRE elapsed interval on every turn —
        the same 27 days of real time were charged ~5 times, driving a healthy
        0.69 to 0.036 (10x the correct decay) and pushing facts past the
        atrophy threshold in a single session. Callers MUST pass decayed_at
        (the row's updated_at) or decay will compound again.

        Returns the decayed trust_score. If no valid timestamp is available,
        returns the original trust_score unchanged.
        """
        half_life = self.get("decay_half_life_days")
        if half_life <= 0:
            return trust_score

        candidates = []
        for value in (last_retrieved, created_at, decayed_at):
            if not value:
                continue
            try:
                if isinstance(value, datetime):
                    dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
                elif isinstance(value, str):
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                else:
                    continue
                candidates.append(dt)
            except Exception:
                continue

        if not candidates:
            return trust_score

        try:
            last = max(candidates)
            now = datetime.now(timezone.utc)
            days_since = (now - last).days
            if days_since <= 0:
                return trust_score

            decay_factor = 0.5 ** (days_since / half_life)
            return min(trust_score * decay_factor, 1.0)
        except Exception:
            return trust_score

    # ── Activation sigmoid ─────────────────────────────────────────

    def compute_activation(self, created_at, retrieval_count: int) -> float:
        """Sigmoid activation — silent → active through time + retrieval.

        Handles both ISO-format strings and datetime objects (database returns
        datetime objects directly; a bare str.replace on a datetime raises
        TypeError, which previously silently gated every fact as 'silent').
        """
        hours = 0.0
        try:
            if isinstance(created_at, datetime):
                created = created_at if created_at.tzinfo else created_at.replace(tzinfo=timezone.utc)
            elif isinstance(created_at, str):
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            else:
                created = None
            if created is not None:
                now = datetime.now(timezone.utc)
                hours = (now - created).total_seconds() / 3600
        except Exception:
            hours = 0.0

        t_half = self.get("maturation_t_half")
        k = self.get("maturation_k")

        # Retrieval accelerates maturation — each retrieval shifts curve left
        acceleration = retrieval_count * (t_half * 0.1)
        adjusted_t_half = max(t_half - acceleration, t_half * 0.2)

        try:
            activation = 1.0 / (1.0 + math.exp(-(hours - adjusted_t_half) / k))
        except OverflowError:
            activation = 0.0 if hours < adjusted_t_half else 1.0

        return min(activation, 1.0)

    # ── Correction detection ───────────────────────────────────────

    def is_correction(self, user_message: str) -> bool:
        """Regex-based correction detection. High precision, low recall.
        
        Catches explicit corrections: 'no', 'wrong', 'actually', 'that's not',
        'incorrect', 'revised', 'updated', 'changed', 'not anymore'.
        
        Embedding-based correction detection was explored and rejected —
        e5-large embeddings capture topic similarity, not semantic
        opposition. 'I eat meat' and 'I am vegan' have high cosine
        similarity (both about diet) making them indistinguishable from
        continuations. The false positive rate is too high for production.
        """
        import re
        for pattern in CORRECTION_PATTERNS:
            if re.search(pattern, user_message, re.IGNORECASE):
                return True
        return False

    def text_overlap(self, content: str, user_message: str) -> float:
        """Simple term overlap score between fact content and user message."""
        content_terms = set(content.lower().split())
        user_terms = set(user_message.lower().split())
        if not content_terms:
            return 0.0
        return len(content_terms & user_terms) / len(content_terms)

    # ── Signal tracking for adaptation ─────────────────────────────

    def record_signal(self, fact_id: int, was_continuation: bool, was_correction: bool):
        self._signals.append({
            "fact_id": fact_id,
            "continuation": was_continuation,
            "correction": was_correction,
        })

    def record_maturity(self, hours_to_mature: float):
        """Record time-to-maturity for a fact (hours from creation to activation >= 0.5).
        Called from prefetch when a fact crosses the maturation threshold."""
        self._maturity_times.append(hours_to_mature)

    def tick(self):
        """Called every turn. Triggers adaptation every N turns."""
        self._turn_count += 1
        if self._turn_count % self._adaptation_interval == 0:
            self._adapt()

    def _adapt(self):
        """Adapt parameters based on accumulated signals."""
        # Signal-based adaptations (require continuation/correction data)
        if self._signals:
            continuations = sum(1 for s in self._signals if s["continuation"])
            corrections = sum(1 for s in self._signals if s["correction"])
            total = len(self._signals)
            continuation_rate = continuations / total if total else 0
            correction_rate = corrections / total if total else 0

            # Relevance floor adaptation
            ignore_rate = 1.0 - continuation_rate - correction_rate
            if ignore_rate > 0.5:
                self._set("relevance_floor", self.get("relevance_floor") + 0.02)
            if continuation_rate > 0.7 and total < 3 * self._adaptation_interval:
                self._set("relevance_floor", self.get("relevance_floor") - 0.01)

            # Decay rate adaptation — if corrections are frequent, shorten
            # half-life so stale facts fade faster. If continuations dominate,
            # lengthen half-life so good facts persist longer.
            if correction_rate > 0.3:
                current = self.get("decay_half_life_days")
                if current > 7:
                    self._set("decay_half_life_days", current - 2)
            elif continuation_rate > 0.7:
                current = self.get("decay_half_life_days")
                if current < 90:
                    self._set("decay_half_life_days", current + 2)

            logger.info(
                "limbic: signal adaptation (turns=%d, cont=%.2f, corr=%.2f, floor=%.3f, half_life=%.0f)",
                self._turn_count, continuation_rate, correction_rate,
                self.get("relevance_floor"), self.get("decay_half_life_days"),
            )
            self._signals.clear()
        else:
            continuation_rate = 0
            correction_rate = 0

        # Maturation curve adaptation — adjust t_half toward observed average maturity time
        if self._maturity_times:
            avg_hours = sum(self._maturity_times) / len(self._maturity_times)
            current_t_half = self.get("maturation_t_half")
            # Adjust t_half toward the observed average (convert hours to the same unit)
            # t_half is in hours; if facts mature faster than t_half, lower it; slower, raise it
            if avg_hours < current_t_half * 0.5:
                # Facts maturing much faster than expected — lower t_half
                self._set("maturation_t_half", current_t_half - 12)
            elif avg_hours > current_t_half * 2:
                # Facts maturing much slower than expected — raise t_half
                self._set("maturation_t_half", current_t_half + 12)

            self._maturity_times.clear()

        logger.info(
            "limbic: adapted parameters (turns=%d, cont=%.2f, corr=%.2f, floor=%.3f)",
            self._turn_count, continuation_rate, correction_rate,
            self.get("relevance_floor"),
        )

        # Clear signals for next cycle
        self._signals.clear()
        self._save_params()