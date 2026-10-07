"""
LimbicStorage — abstract storage backend interface for Limbic.

SQLite is the only backend implementation (store_sqlite.py).
"""
from __future__ import annotations


class LimbicStorage:
    """Storage backend interface."""

    def store(self, content: str, user_id: str, embedding: list[float],
              metadata: dict) -> int:
        raise NotImplementedError

    def search(self, embedding: list[float], filters: dict, limit: int) -> list[dict]:
        raise NotImplementedError

    def search_fts(self, query_text: str, filters: dict, limit: int) -> list[dict]:
        """Full-text search using BM25. Override in implementations that support FTS."""
        return []  # default: no FTS support

    def store_entities(self, fact_id: int, entities: list[dict]):
        """Store entities and create graph edges. Override in implementations that support graphs."""
        pass  # default: no graph support

    def search_graph(self, entity_names: list[str], filters: dict, limit: int) -> list[dict]:
        """Graph traversal search. Override in implementations that support graphs."""
        return []  # default: no graph support

    def get(self, fact_id: int) -> dict | None:
        raise NotImplementedError

    def update(self, fact_id: int, **fields) -> None:
        raise NotImplementedError

    def delete_user(self, user_id: str) -> int:
        raise NotImplementedError

    def export_user(self, user_id: str) -> dict:
        raise NotImplementedError

    def count(self, user_id: str | None = None) -> int:
        raise NotImplementedError

    def list_user_ids(self) -> list[str]:
        """Distinct user_ids present in the store."""
        raise NotImplementedError

    # ── Typed analytical / maintenance methods ────────────────────────
    # Backends implement these natively. Default implementations return
    # zeroed data so callers degrade gracefully (same contract the old
    # untranslatable-query -> [] path had, but explicit and debuggable).

    def get_all_embeddings(self, user_id: str) -> list[list[float]]:
        """All stored embeddings for a user (centroid cache)."""
        return []

    def count_user_facts(self, user_id: str) -> int:
        return 0

    def decay_rows(self, user_id: str) -> list[dict]:
        """Rows the temporal decay pass updates: fact_id/trust_score/last_retrieved/created_at."""
        return []

    def atrophy_scan_rows(self, user_id: str, threshold: float,
                          limit: int) -> list[dict]:
        """Unmarked facts below the atrophy threshold."""
        return []

    def avg_trust(self, user_id: str) -> float:
        return 0.0

    def count_matured(self, user_id: str) -> int:
        return 0

    def count_faded(self, user_id: str, threshold: float = 0.3) -> int:
        return 0

    def check_embedder_stamp(self, stamp: str | None, dims: int) -> bool:
        """Abstract default: no stamping support (never stale)."""
        return False

    def facts_per_user(self) -> list[dict]:
        """Aggregate: [{user_id, facts}, ...]."""
        return []

    def delete_fact(self, fact_id: int) -> bool:
        """Surgically remove one fact. Returns True if it existed."""
        return False

    def delete_fact_by_content(self, content: str, user_id: str) -> bool:
        """Best-effort remove one fact matching exact content+user. Returns True if deleted."""
        return False

    # ── Contradiction links (continuing correction) ────────────────

    def link_contradiction(self, newer_fact_id: int, older_fact_id: int,
                           label: str = "contradiction") -> None:
        """Record an NLI-detected contradiction between a fact pair. Idempotent."""
        pass

    def unresolved_contradictions(self, user_id: str, limit: int) -> list[dict]:
        """Open links for a user, oldest first, bounded."""
        return []

    def resolve_contradiction(self, newer_fact_id: int, older_fact_id: int) -> None:
        """Close a link — the correction has landed, or no longer holds."""
        pass

    def count_open_contradictions(self, user_id: str) -> int:
        return 0

    def delete_expired_contradictions(self, cutoff_iso: str) -> int:
        """Drop open links older than the TTL."""
        return 0

    def close(self) -> None:
        pass