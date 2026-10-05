"""Limbic storage — SQLite backend.

SQLite storage backend using sqlite-vec + FTS5.
Single-file DB (limbic.db) with WAL mode for durability — no clog/manifest
directory, no bloat, survives ungraceful restarts.

SQLite gives us:
- WAL mode (durable, survives kill -9, no checkpoint-on-close bug)
- sqlite-vec (brute-force KNN over vec0 — linear scan, no ANN index)
- FTS5 (full-text search, BM25)
- CREATE TRIGGER (track_maturity, FTS sync)
- Junction tables (entity graph)
- Single-file backup (VACUUM INTO — consistent snapshot, see backup())
- ACID transactions
"""
from __future__ import annotations

try:  # POSIX only — Windows has no fcntl; initialize degrades to the fail-open path instead of ImportError
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None
import json
import logging
import os
import sqlite3
import struct
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import sqlite_vec

from .storage import LimbicStorage

# ponytail: cross-process writer serialization (advisory lock file).
# Gap: in-process writes are serialized by self._lock (RLock), but OTHER
# processes (hermes limbic CLI verbs, reindex.py) open this same file and
# write with only the SQLite busy_timeout between them and the gateway.
# SQLite serializes write transactions, not multi-statement logical
# operations — interleaved multi-writer sessions were the plausible cause
# of the 2026-09-08 page corruption (see the single-writer invariant block
# in __init__). The old convention "run CLI write verbs only while the
# profile is stopped" was paper-only; the lock file makes it mechanical.
# Mechanism: exclusive flock on <db>.writer.lock, taken by every write
# transaction (the `with self._lock, self._db:` blocks), held only for the
# transaction — a CLI writer waits at most one gateway transaction, reads
# never queue; WAL snapshot readers coexist with writers by design.
# Same-process reentrancy: the fd is cached per (pid, abspath) and SHARED.
# flock exclusivity is per open file description, so a second
# SQLiteStorage instance on the same file must NOT open its own fd or it
# would deadlock against the first (provider config hot-swap overlaps two
# instances; tests double-open). Refcounted release in close().
# Fail-open on flock errors — advisory hardening, never a gateway-killer.

logger = logging.getLogger(__name__)

# ── Cross-process writer lock (<db>.writer.lock) ─────────────────────
# One exclusive flock serializes every write transaction across processes.
_writer_fds: dict = {}  # (pid, abspath) -> [fd, refcount]
_writer_mu = threading.Lock()


def _writer_lock_path(db_path: str) -> str:
    return db_path + ".writer.lock"


def _writer_acquire(db_path: str):
    """Acquire the exclusive writer flock; context-manager style.

    Returns None when locking is unavailable — callers treat that as
    fail-open (advisory mechanism only).
    """
    key = (os.getpid(), os.path.abspath(db_path))
    with _writer_mu:
        entry = _writer_fds.get(key)
        if entry is None:
            if fcntl is None:
                logger.warning("limbic: fcntl unavailable (non-POSIX) — "
                               "proceeding fail-open")
                return None
            try:
                fd = os.open(_writer_lock_path(db_path),
                             os.O_RDWR | os.O_CREAT, 0o600)
            except OSError as e:
                logger.warning("limbic: writer lock unavailable (%s) — "
                               "proceeding fail-open", e)
                return None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as e:
                os.close(fd)
                logger.warning("limbic: flock failed (%s) — proceeding "
                               "fail-open", e)
                return None
            entry = [fd, 0]
            _writer_fds[key] = entry
        entry[1] += 1
        return key


def _writer_release(key) -> None:
    """Drop one claim; the fd closes when the last instance closes."""
    if key is None:
        return
    with _writer_mu:
        entry = _writer_fds.get(key)
        if entry is None:
            return
        entry[1] -= 1
        if entry[1] <= 0:
            fd = entry[0]
            del _writer_fds[key]
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(fd)

# Whitelist of updatable facts columns (fact_id excluded — updating the PK
# would desync vec_index rowid linkage).
_UPDATABLE_COLUMNS = frozenset({
    "content", "user_id", "trust_score", "retrieval_count",
    "last_retrieved", "matured_at", "atrophy_marked_at", "created_at",
    "updated_at",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_vec(vec: list[float]) -> str:
    """Serialize a vector for sqlite-vec (JSON array string)."""
    return json.dumps(vec)


def _deserialize_vec(blob) -> list[float]:
    """Deserialize a vector from sqlite-vec storage.

    vec0 stores embeddings as raw float32 bytes (4 bytes per float).
    We also handle JSON strings (our insert format) and plain lists.
    """
    if isinstance(blob, str):
        return json.loads(blob)
    if isinstance(blob, (bytes, bytearray)):
        # vec0 stores as float32 (4 bytes per float), NOT float64
        n = len(blob) // 4
        if n == 0:
            return []
        return list(struct.unpack(f"{n}f", blob))
    if isinstance(blob, list):
        return blob
    return []


class SQLiteStorage(LimbicStorage):
    """SQLite embedded storage. One .db file per agent profile."""

    def __init__(self, path: str, dims: int = 384, collection_name: str = "limbic"):
        # The provider passes the path for the SQLite database.
        if path.endswith("/"):
            path = os.path.join(path, "limbic.db")
        elif not path.endswith(".db"):
            path = os.path.join(path, "limbic.db")

        self._db_path = path
        self._dims = dims
        self._collection = collection_name
        self._lock = threading.RLock()

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)  # load only what sqlite_vec needs; keep disabled
        self._degraded = False  # set True below on pragma/schema failure

        # WAL mode for durability — survives ungraceful restarts
        try:
            self._db.execute("PRAGMA journal_mode=WAL")
            # synchronous=NORMAL: WAL commits are not fsynced per-commit — an OS /
            # power loss may lose the most recent transactions but cannot corrupt
            # the database image. FULL would fsync every commit at a real latency
            # cost per remember() call; NORMAL is a deliberate tradeoff. Change
            # only with operator sign-off (pragma behavior change, not a one-liner).
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            # 2026-09-04: remember() hit "database is locked" under gateway write load.
            # WAL had ballooned to 11MB — stalled checkpoints make writers wait, and 5s
            # wasn't enough. Raise the wait, checkpoint little and often, cap the WAL.
            self._db.execute("PRAGMA busy_timeout=15000")
            self._db.execute("PRAGMA wal_autocheckpoint=256")
            self._db.execute("PRAGMA journal_size_limit=8388608")
        except sqlite3.DatabaseError as e:
            self._degraded = True
            logger.critical(
                "limbic: pragma setup failed on possibly-corrupt %s: %s — "
                "opening DEGRADED (quick_check result logged separately)",
                path, e,
            )

        # ── Single-writer invariant ────────────────────────────────────
        # This connection is meant to be the ONLY writer to this DB file.
        # The gateway holds one long-lived connection per profile; CLI verbs
        # (hermes limbic ...) and scripts (reindex.py) open SEPARATE
        # connections to the same file while the gateway keeps writing.
        # Multi-process writers + WAL + synchronous=NORMAL is the plausible
        # recipe behind the 2026-09-08 page corruption (duplicate rowids,
        # out-of-order cells). SQLite serializes write *transactions*, it
        # does not protect against interleaved multi-process writer bugs.
        # HARDENED 2026-10-02: every write transaction runs under an
        # exclusive flock on <db>.writer.lock (_write_txn) — CLI verbs and
        # reindex.py now serialize against the gateway instead of relying
        # on the old stop-the-profile convention.
        # Corrupt-image policy: pragmas AND schema init raise DatabaseError
        # on a damaged file. Both are wrapped — a corrupt DB must open
        # degraded-but-visible (healthy=False), never crash the gateway.
        # quick_check below adds the forensic detail to the log.
        self._startup_quick_check()
        try:
            self._init_schema()
        except sqlite3.DatabaseError as e:
            self._degraded = True
            logger.critical(
                "limbic: schema init failed on possibly-corrupt %s: %s — "
                "opening DEGRADED; write/read calls will surface DatabaseError",
                path, e,
            )

        logger.info(
            "limbic: SQLite at %s (dims=%d, collection=%s, facts=%s, degraded=%s)",
            path, dims, collection_name,
            "n/a" if self._degraded else self.count(),
            self._degraded,
        )

    def _startup_quick_check(self):
        """One-time integrity probe at open. Logs loudly; does not quarantine
        (a corrupt DB must still be openable for read/export/repair — the
        operator decides via `healthy` / backup / reindex)."""
        try:
            row = self._db.execute("PRAGMA quick_check").fetchone()
            result = row[0] if row else "unknown"
        except sqlite3.DatabaseError as e:
            result = f"quick_check itself failed: {e}"
        if result != "ok":
            logger.error(
                "limbic: SQLite quick_check FAILED at %s: %s — DB image may be "
                "corrupt; run `hermes limbic fts-rebuild` / reindex or restore "
                "from backup()", self._db_path, result,
            )
        else:
            logger.debug("limbic: startup quick_check ok at %s", self._db_path)

    def _write_txn(self, body: Callable[[], Any]) -> Any:
        """Run a write transaction under the cross-process writer flock.

        The in-process lock guards thread access to the shared connection;
        the flock covers the multi-statement transaction so other processes
        (CLI verbs, reindex) serialize against it instead of interleaving.
        Commit happens on body success via `with self._db:`; rollback on
        exception; flock released in finally, after commit.
        """
        key = _writer_acquire(self._db_path)
        try:
            with self._lock, self._db:
                return body()
        finally:
            _writer_release(key)

    @property
    def healthy(self) -> bool:
        """Read-only integrity probe: True iff PRAGMA quick_check passes."""
        try:
            with self._lock:
                row = self._db.execute("PRAGMA quick_check").fetchone()
            return bool(row) and row[0] == "ok"
        except sqlite3.DatabaseError as e:
            logger.error("limbic: health probe failed (possible corruption): %s", e)
            return False

    def _init_schema(self):
        """Create all tables, indexes, triggers if they don't exist."""
        db = self._db

        # Main facts table
        db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                user_id TEXT NOT NULL,
                trust_score REAL DEFAULT 0.5,
                retrieval_count INTEGER DEFAULT 0,
                last_retrieved TEXT,
                matured_at TEXT,
                atrophy_marked_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Indexes for common queries
        db.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_facts_atrophy ON facts(atrophy_marked_at)")

        # FTS5 virtual table for full-text search (external content table)
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
                content,
                content='facts',
                content_rowid='fact_id',
                tokenize='porter unicode61'
            )
        """)

        # sqlite-vec virtual table for KNN vector search
        # distance_metric=cosine for KNN similarity search
        # (distance 0 = identical, 1 = orthogonal, 2 = opposite)
        # Without this, vec0 defaults to L2 (euclidean) which produces
        # incorrect scores when the provider does score = 1.0 - distance
        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_index USING vec0(embedding float[{self._dims}] distance_metric=cosine)"
        )

        # Entity graph tables
        db.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                type TEXT DEFAULT 'UNKNOWN'
            )
        """)
        # Contradiction links — one row per NLI-detected contradiction between a
        # newer fact and an older one. Records what write-time NLI already decided;
        # does NOT re-detect. Re-evaluated on tick until resolved (or TTL expiry).
        db.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                newer_fact_id  INTEGER NOT NULL,
                older_fact_id  INTEGER NOT NULL,
                created_at     TEXT NOT NULL,
                nli_label      TEXT NOT NULL,
                resolved_at    TEXT,
                PRIMARY KEY (newer_fact_id, older_fact_id),
                FOREIGN KEY (newer_fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY (older_fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_contradictions_open ON contradictions(resolved_at)")

        db.execute("""
            CREATE TABLE IF NOT EXISTS mentions (
                fact_id INTEGER NOT NULL,
                entity_id TEXT NOT NULL,
                PRIMARY KEY (fact_id, entity_id),
                FOREIGN KEY (fact_id) REFERENCES facts(fact_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
            )
        """)

        # Triggers — track_maturity and FTS sync

        # Auto-stamp maturity: when retrieval_count crosses 5, record matured_at
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS track_maturity
            AFTER UPDATE OF retrieval_count ON facts
            FOR EACH ROW
            WHEN NEW.retrieval_count >= 5 AND NEW.matured_at IS NULL
            BEGIN
                UPDATE facts SET matured_at = datetime('now') WHERE fact_id = NEW.fact_id;
            END
        """)

        # FTS sync triggers
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content) VALUES (new.fact_id, new.content);
            END
        """)
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.fact_id, old.content);
            END
        """)
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content) VALUES('delete', old.fact_id, old.content);
                INSERT INTO facts_fts(rowid, content) VALUES (new.fact_id, new.content);
            END
        """)

        self._db.commit()

    # ── Core CRUD ──────────────────────────────────────────────────

    def store(self, content: str, user_id: str, embedding: list[float],
              metadata: dict) -> int:
        now_iso = _now_iso()
        params = (
            content,
            user_id,
            metadata.get("trust_score", 0.5),
            metadata.get("retrieval_count", 0),
            metadata.get("last_retrieved"),
            metadata.get("matured_at"),
            None,  # atrophy_marked_at
            metadata.get("created_at", now_iso),
            metadata.get("updated_at", now_iso),
        )
        def _txn():
            # AUTOINCREMENT assigns fact_id; lastrowid links the vec row.
            cur = self._db.execute(
                """INSERT INTO facts (content, user_id, trust_score,
                   retrieval_count, last_retrieved,
                   matured_at, atrophy_marked_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                params
            )
            fact_id = int(cur.lastrowid or 0)

            # Insert embedding into vec_index. A failure here (e.g. dim
            # mismatch) rolls the facts row back too — no half-stored fact,
            # no pending transaction left open.
            self._db.execute(
                "INSERT INTO vec_index(rowid, embedding) VALUES (?, ?)",
                (fact_id, _serialize_vec(embedding))
            )
            return fact_id
        return self._write_txn(_txn)

    def search(self, embedding: list[float], filters: dict, limit: int) -> list[dict]:
        with self._lock:
            query_vec = _serialize_vec(embedding)
            limit = max(1, int(limit))  # guard: limit<=0/non-int reaches sqlite-vec as a corruption-class error

            # For filtered KNN, fetch a larger candidate set from vec_index
            # in a subquery (which produces the distance), then join with
            # facts for filtering and limit in the outer query.
            knn_limit = limit * 10 if filters else limit

            # KNN subquery produces rowid + distance; outer query joins + filters
            sql = """
                SELECT knn.rowid AS fact_id, knn.distance,
                       f.content, f.user_id, f.trust_score,
                       f.retrieval_count, f.last_retrieved,
                       f.matured_at, f.atrophy_marked_at,
                       f.created_at, f.updated_at
                FROM (SELECT rowid, distance FROM vec_index WHERE embedding MATCH ? ORDER BY distance LIMIT ?) AS knn
                JOIN facts f ON knn.rowid = f.fact_id
                WHERE 1=1
            """
            params: list = [query_vec, knn_limit]

            # Add user filters
            where_extra = []
            if "user_id" in filters:
                where_extra.append("f.user_id = ?")
                params.append(filters["user_id"])
            elif "user_id_any" in filters:
                placeholders = ",".join("?" * len(filters["user_id_any"]))
                where_extra.append(f"f.user_id IN ({placeholders})")
                params.extend(filters["user_id_any"])

            if where_extra:
                sql += " AND " + " AND ".join(where_extra)

            sql += f" ORDER BY knn.distance LIMIT {limit}"

            try:
                results = self._db.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as e:
                logger.error("limbic: SQLite KNN search failed (possible corruption): %s", e)
                return []
            except Exception as e:
                logger.warning("limbic: SQLite KNN search failed: %s", e)
                return []

            parsed = []
            for r in results:
                distance = r["distance"]
                score = 1.0 - distance
                parsed.append({
                    "fact_id": r["fact_id"],
                    "content": r["content"],
                    "user_id": r["user_id"],
                    "trust_score": r["trust_score"],
                    "retrieval_count": r["retrieval_count"],
                    "last_retrieved": r["last_retrieved"],
                    "matured_at": r["matured_at"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "embedding": None,
                    "score": score,
                    "atrophy_marked_at": r["atrophy_marked_at"],
                })

            return parsed

    def get(self, fact_id: int) -> dict | None:
        with self._lock:
            try:
                row = self._db.execute(
                    "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
                ).fetchone()
                if row:
                    return {
                        "fact_id": row["fact_id"],
                        "content": row["content"],
                        "user_id": row["user_id"],
                        "trust_score": row["trust_score"],
                        "retrieval_count": row["retrieval_count"],
                        "last_retrieved": row["last_retrieved"],
                        "matured_at": row["matured_at"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
            except sqlite3.DatabaseError as e:
                logger.error("limbic: get fact %d failed (possible corruption): %s", fact_id, e)
            except Exception as e:
                logger.debug("limbic: get fact %d failed: %s", fact_id, e)
            return None

    def update(self, fact_id: int, **fields) -> None:
        if not fields:
            return

        # Whitelist against real facts columns — the SET clause used to be
        # built from raw f-strings, so any **fields key reached the SQL text.
        unknown = set(fields) - _UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(
                f"limbic: update() rejected unknown/forbidden column(s): {sorted(unknown)}"
            )

        fields["updated_at"] = _now_iso()

        set_parts = []
        params = []
        for key, value in fields.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        params.append(fact_id)
        sql = f"UPDATE facts SET {', '.join(set_parts)} WHERE fact_id = ?"
        def _txn():
            self._db.execute(sql, params)
        return self._write_txn(_txn)

    def delete_user(self, user_id: str) -> int:
        def _txn():
            count = self.count(user_id)
            rows = self._db.execute(
                "SELECT fact_id FROM facts WHERE user_id = ?", (user_id,)
            ).fetchall()
            fact_ids = [r["fact_id"] for r in rows]

            # Delete from facts (cascade triggers handle FTS, mentions via FK cascade)
            self._db.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))

            # Delete from vec_index (no FK, manual cleanup)
            self._db.executemany(
                "DELETE FROM vec_index WHERE rowid = ?", [(fid,) for fid in fact_ids]
            )
            return count
        return self._write_txn(_txn)

    def export_user(self, user_id: str) -> dict:
        with self._lock:
            rows = self._db.execute(
                """SELECT fact_id, content, user_id, trust_score,
                      retrieval_count, last_retrieved, matured_at,
                      atrophy_marked_at, created_at, updated_at
                   FROM facts WHERE user_id = ? ORDER BY fact_id""",
                (user_id,)
            ).fetchall()

            facts = []
            for r in rows:
                facts.append({
                    "fact_id": r["fact_id"],
                    "content": r["content"],
                    "user_id": r["user_id"],
                    "trust_score": r["trust_score"],
                    "retrieval_count": r["retrieval_count"],
                    "last_retrieved": r["last_retrieved"],
                    "matured_at": r["matured_at"],
                    "atrophy_marked_at": r["atrophy_marked_at"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                })

            return {"user_id": user_id, "fact_count": len(facts), "facts": facts}

    def count(self, user_id: str | None = None) -> int:
        with self._lock:
            if user_id:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM facts WHERE user_id = ?", (user_id,)
                ).fetchone()
            else:
                row = self._db.execute("SELECT COUNT(*) FROM facts").fetchone()
            return row[0] if row else 0

    # ── Typed analytical / maintenance methods ────────────────────

    def get_all_embeddings(self, user_id: str) -> list[list[float]]:
        """All stored embeddings for a user (centroid cache)."""
        with self._lock:
            try:
                rows = self._db.execute(
                    """SELECT v.embedding FROM vec_index v
                       JOIN facts f ON v.rowid = f.fact_id
                       WHERE f.user_id = ?""",
                    (user_id,)
                ).fetchall()
                return [_deserialize_vec(r["embedding"]) for r in rows]
            except sqlite3.DatabaseError as e:
                logger.error("limbic: get_all_embeddings failed (possible corruption): %s", e)
                return []
            except Exception as e:
                logger.debug("limbic: get_all_embeddings failed: %s", e)
                return []

    def count_user_facts(self, user_id: str) -> int:
        return self.count(user_id)

    def list_user_ids(self) -> list[str]:
        """Distinct user_ids present in the store (for identity fallback)."""
        with self._lock:
            try:
                rows = self._db.execute(
                    "SELECT DISTINCT user_id FROM facts LIMIT 50"
                ).fetchall()
                return [r["user_id"] for r in rows if r["user_id"]]
            except sqlite3.DatabaseError as e:
                logger.error("limbic: list_user_ids failed (corruption?): %s", e)
                return []

    def decay_rows(self, user_id: str) -> list[dict]:
        """Rows the temporal decay pass must update.

        Historical note: this query path silently returned [] under the old
        SQL-translation layer, so temporal decay never ran on SQLite and
        stale facts never faded. Typed method makes the failure explicit.
        """
        with self._lock:
            rows = self._db.execute(
                """SELECT fact_id, trust_score, last_retrieved, created_at, updated_at
                   FROM facts WHERE user_id = ?""",
                (user_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def atrophy_scan_rows(self, user_id: str, threshold: float,
                          limit: int) -> list[dict]:
        """Unmarked facts below the atrophy threshold."""
        with self._lock:
            rows = self._db.execute(
                """SELECT fact_id, trust_score, last_retrieved, created_at
                   FROM facts WHERE user_id = ?
                       AND atrophy_marked_at IS NULL
                   AND trust_score < ?
                   LIMIT ?""",
                (user_id, threshold, limit)
            ).fetchall()
            return [dict(r) for r in rows]

    def avg_trust(self, user_id: str) -> float:
        with self._lock:
            row = self._db.execute(
                "SELECT AVG(trust_score) FROM facts WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            return row[0] if row and row[0] is not None else 0.0

    def count_matured(self, user_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM facts WHERE user_id = ? AND matured_at IS NOT NULL",
                (user_id,)
            ).fetchone()
            return row[0] if row else 0

    def count_faded(self, user_id: str, threshold: float = 0.3) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM facts WHERE user_id = ? AND trust_score < ?",
                (user_id, threshold)
            ).fetchone()
            return row[0] if row else 0

    def facts_per_user(self) -> list[dict]:
        """Aggregate fact counts: [{user_id, facts}, ...]."""
        with self._lock:
            rows = self._db.execute(
                "SELECT user_id, COUNT(*) AS c FROM facts GROUP BY user_id"
            ).fetchall()
            return [{"user_id": r["user_id"], "facts": r["c"]} for r in rows]

    def delete_fact(self, fact_id: int) -> bool:
        """Surgically remove one fact. Returns True if it existed.

        FTS row cleanup is handled by the facts_ad trigger; vec_index has
        no FK so it is cleaned manually (same pattern as sweep_atrophy).
        One transaction: an FTS trigger abort rolls the whole delete back
        and raises instead of committing a half-deleted fact.
        """
        def _txn():
            cur = self._db.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._db.execute("DELETE FROM vec_index WHERE rowid = ?", (fact_id,))
            return cur.rowcount > 0
        return self._write_txn(_txn)

    def delete_fact_by_content(self, content: str, user_id: str) -> bool:
        """Best-effort remove one fact matching exact content+user."""
        def _txn():
            row = self._db.execute(
                "SELECT fact_id FROM facts WHERE content = ? AND user_id = ? LIMIT 1",
                (content, user_id)
            ).fetchone()
            if not row:
                return False
            self._db.execute("DELETE FROM facts WHERE fact_id = ?", (row["fact_id"],))
            self._db.execute("DELETE FROM vec_index WHERE rowid = ?", (row["fact_id"],))
            return True
        return self._write_txn(_txn)

    # ── Contradiction links (continuing correction) ────────────────

    def link_contradiction(self, newer_fact_id: int, older_fact_id: int,
                           label: str = "contradiction") -> None:
        """Record an NLI-detected contradiction between a fact pair.

        Idempotent — INSERT OR IGNORE means a repeated write cannot resurrect a
        link that was already resolved.
        """
        def _txn():
            self._db.execute(
                """INSERT OR IGNORE INTO contradictions
                   (newer_fact_id, older_fact_id, created_at, nli_label, resolved_at)
                   VALUES (?, ?, ?, ?, NULL)""",
                (newer_fact_id, older_fact_id, _now_iso(), label),
            )
        return self._write_txn(_txn)

    def unresolved_contradictions(self, user_id: str, limit: int) -> list[dict]:
        """Open links for a user, oldest first, bounded.

        Joins both facts so the caller can check trust and activation without a
        second round-trip. Scoped by the OLDER fact's owner — the corrected fact
        is the one whose store the link protects.
        """
        with self._lock:
            rows = self._db.execute(
                """SELECT c.newer_fact_id, c.older_fact_id, c.created_at, c.nli_label,
                          n.content AS newer_content, n.trust_score AS newer_trust,
                          n.created_at AS newer_created_at,
                          n.retrieval_count AS newer_retrievals,
                          o.content AS older_content, o.trust_score AS older_trust,
                          o.created_at AS older_created_at
                   FROM contradictions c
                   JOIN facts n ON n.fact_id = c.newer_fact_id
                   JOIN facts o ON o.fact_id = c.older_fact_id
                   WHERE c.resolved_at IS NULL AND o.user_id = ?
                   ORDER BY c.created_at ASC
                   LIMIT ?""",
                (user_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve_contradiction(self, newer_fact_id: int, older_fact_id: int) -> None:
        """Close a link — the correction has landed, or no longer holds."""
        def _txn():
            self._db.execute(
                """UPDATE contradictions SET resolved_at = ?
                   WHERE newer_fact_id = ? AND older_fact_id = ? AND resolved_at IS NULL""",
                (_now_iso(), newer_fact_id, older_fact_id),
            )
        return self._write_txn(_txn)

    def count_open_contradictions(self, user_id: str) -> int:
        with self._lock:
            row = self._db.execute(
                """SELECT COUNT(*) FROM contradictions c
                   JOIN facts o ON o.fact_id = c.older_fact_id
                   WHERE c.resolved_at IS NULL AND o.user_id = ?""",
                (user_id,),
            ).fetchone()
            return row[0] if row else 0

    def delete_expired_contradictions(self, cutoff_iso: str) -> int:
        """Drop open links older than the TTL — no longer worth re-checking."""
        def _txn():
            cur = self._db.execute(
                "DELETE FROM contradictions WHERE created_at < ? AND resolved_at IS NULL",
                (cutoff_iso,),
            )
            return cur.rowcount or 0
        return self._write_txn(_txn)

# ── Full-text search ───────────────────────────────────────────

    def search_fts(self, query_text: str, filters: dict, limit: int) -> list[dict]:
        with self._lock:
            # FTS5 MATCH has its own query syntax — raw user text with
            # ?, *, quotes etc. raises fts5 syntax errors. Wrap each
            # token in quotes: kills syntax errors, preserves AND
            # semantics (verified identical row counts vs raw).
            tokens = query_text.replace('"', " ").split()
            if not tokens:
                return []  # empty/whitespace query — nothing to match; NOT a corruption event
            limit = max(1, int(limit))
            safe_text = " ".join(f'"{t}"' for t in tokens)
            where_parts = ["facts_fts MATCH ?"]
            params: list = [safe_text]

            if "user_id" in filters:
                where_parts.append("f.user_id = ?")
                params.append(filters["user_id"])
            elif "user_id_any" in filters:
                placeholders = ",".join("?" * len(filters["user_id_any"]))
                where_parts.append(f"f.user_id IN ({placeholders})")
                params.extend(filters["user_id_any"])

            where_clause = " AND ".join(where_parts)

            sql = f"""
                SELECT f.fact_id, f.content, f.user_id, f.trust_score,
                       f.retrieval_count,
                       f.last_retrieved, f.matured_at, f.atrophy_marked_at,
                       f.created_at, f.updated_at,
                       bm25(facts_fts) AS fts_score
                FROM facts_fts
                JOIN facts f ON facts_fts.rowid = f.fact_id
                WHERE {where_clause}
                ORDER BY fts_score
                LIMIT {limit}
            """

            try:
                rows = self._db.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as e:
                logger.error("limbic: SQLite FTS search failed (possible corruption): %s", e)
                return []
            except Exception as e:
                logger.warning("limbic: SQLite FTS search failed: %s", e)
                return []

            parsed = []
            for r in rows:
                parsed.append({
                    "fact_id": r["fact_id"],
                    "content": r["content"],
                    "user_id": r["user_id"],
                    "trust_score": r["trust_score"],
                    "retrieval_count": r["retrieval_count"],
                    "last_retrieved": r["last_retrieved"],
                    "matured_at": r["matured_at"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "embedding": None,
                    "score": r["fts_score"],
                    "atrophy_marked_at": r["atrophy_marked_at"],
                })

            return parsed

    def rebuild_fts(self) -> None:
        """Rebuild the FTS5 index from the facts table.

        Maintenance verb (hermes limbic fts-rebuild) — repairs drift or
        corruption between facts and the external-content FTS index by
        repopulating it from scratch ('rebuild' is FTS5's own command).
        """
        def _txn():
            self._db.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
        return self._write_txn(_txn)

    # ── Entity graph ───────────────────────────────────────────────

    def store_entities(self, fact_id: int, entities: list[dict]):
        def _txn():
            import hashlib
            for ent in entities:
                name = ent.get("name", "").strip()
                etype = ent.get("type", "UNKNOWN")
                if not name:
                    continue

                ent_id = hashlib.md5(name.lower().encode()).hexdigest()[:12]

                self._db.execute(
                    "INSERT OR IGNORE INTO entities (entity_id, name, type) VALUES (?, ?, ?)",
                    (ent_id, name, etype)
                )
                self._db.execute(
                    "INSERT OR IGNORE INTO mentions (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, ent_id)
                )
        return self._write_txn(_txn)

    def search_graph(self, entity_names: list[str], filters: dict, limit: int) -> list[dict]:
        with self._lock:
            import hashlib

            all_fact_ids = set()
            for name in entity_names:
                ent_id = hashlib.md5(name.lower().encode()).hexdigest()[:12]
                try:
                    rows = self._db.execute(
                        "SELECT fact_id FROM mentions WHERE entity_id = ?",
                        (ent_id,)
                    ).fetchall()
                    for r in rows:
                        all_fact_ids.add(r["fact_id"])
                except Exception:
                    pass

            if not all_fact_ids:
                return []

            fid_list = list(all_fact_ids)[:limit * 5]
            placeholders = ",".join("?" * len(fid_list))
            params: list = list(fid_list)

            where_extra = []
            if "user_id" in filters:
                where_extra.append("user_id = ?")
                params.append(filters["user_id"])
            elif "user_id_any" in filters:
                ph = ",".join("?" * len(filters["user_id_any"]))
                where_extra.append(f"user_id IN ({ph})")
                params.extend(filters["user_id_any"])

            where_clause = " AND ".join(where_extra) if where_extra else ""

            sql = f"""
                SELECT fact_id, content, user_id, trust_score,
                       retrieval_count, last_retrieved, matured_at,
                       atrophy_marked_at, created_at, updated_at
                FROM facts
                WHERE fact_id IN ({placeholders})
            """
            if where_clause:
                sql += f" AND {where_clause}"
            sql += f" LIMIT {max(1, int(limit))}"

            try:
                rows = self._db.execute(sql, params).fetchall()
            except sqlite3.DatabaseError as e:
                logger.error("limbic: SQLite graph search failed (possible corruption): %s", e)
                return []
            except Exception as e:
                logger.warning("limbic: SQLite graph search failed: %s", e)
                return []

            parsed = []
            for r in rows:
                parsed.append({
                    "fact_id": r["fact_id"],
                    "content": r["content"],
                    "user_id": r["user_id"],
                    "trust_score": r["trust_score"],
                    "retrieval_count": r["retrieval_count"],
                    "last_retrieved": r["last_retrieved"],
                    "matured_at": r["matured_at"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "embedding": None,
                    "score": 0.0,
                    "atrophy_marked_at": r["atrophy_marked_at"],
                })

            return parsed

    # ── Atrophy ────────────────────────────────────────────────────

    def mark_atrophy(self, fact_id: int) -> None:
        def _txn():
            self._db.execute(
                "UPDATE facts SET atrophy_marked_at = ? WHERE fact_id = ?",
                (_now_iso(), fact_id)
            )
        return self._write_txn(_txn)

    def sweep_atrophy(self, before_iso: str, limit: int = 3, user_id: str = "") -> int:
        """Delete facts whose atrophy mark is older than before_iso.

        Optional user_id scopes the sweep to that user; without it the
        sweep is global (legacy callers). Single transaction — all-or-nothing.
        """
        def _txn():
            if user_id:
                rows = self._db.execute(
                    """SELECT fact_id FROM facts
                       WHERE atrophy_marked_at IS NOT NULL
                       AND atrophy_marked_at < ?
                       AND user_id = ?
                       LIMIT ?""",
                    (before_iso, user_id, limit)
                ).fetchall()
            else:
                rows = self._db.execute(
                    """SELECT fact_id FROM facts
                       WHERE atrophy_marked_at IS NOT NULL
                       AND atrophy_marked_at < ?
                       LIMIT ?""",
                    (before_iso, limit)
                ).fetchall()

            deleted = 0
            for r in rows:
                fid = r["fact_id"]
                self._db.execute("DELETE FROM facts WHERE fact_id = ?", (fid,))
                self._db.execute("DELETE FROM vec_index WHERE rowid = ?", (fid,))
                deleted += 1

            return deleted
        return self._write_txn(_txn)

    def count_atrophy_candidates(self) -> int:
        with self._lock:
            try:
                row = self._db.execute(
                    "SELECT COUNT(*) FROM facts WHERE atrophy_marked_at IS NOT NULL"
                ).fetchone()
                return row[0] if row else 0
            except sqlite3.DatabaseError as e:
                logger.error("limbic: count_atrophy_candidates failed (possible corruption): %s", e)
                return 0
            except Exception as e:
                logger.warning("limbic: count_atrophy_candidates failed: %s", e)
                return 0

    # ── Reindex ────────────────────────────────────────────────────

    def reindex(self, new_embedder, new_dims: int) -> dict:
        """Re-embed all facts with a new embedder and atomically rebuild vec_index.

        Crash safety: all embeddings are computed BEFORE the index is touched;
        the DROP → CREATE → INSERT swap then runs in ONE explicit transaction
        (DDL is transactional in SQLite under BEGIN). A crash or failure before
        COMMIT rolls back to the old index — the old failure mode (DROP first,
        then embed → crash destroys all vectors, and a 2026-09-08 verified
        incident: reindex CREATE dropped distance_metric=cosine) cannot recur.
        """
        import time
        t0 = time.time()
        try:
            with self._lock:
                # Stream fact rows in batches: we hold only fact_id+content +
                # the new embeddings, not the full result set.
                cur = self._db.execute("SELECT fact_id, content FROM facts")
                embedded: list[tuple[int, list[float]]] = []
                total = 0
                errors = 0
                skipped_blank = 0
                while True:
                    rows = cur.fetchmany(500)
                    if not rows:
                        break
                    total += len(rows)
                    for r in rows:
                        content = r["content"]
                        if not content.strip():
                            skipped_blank += 1
                            continue
                        try:
                            emb = new_embedder.embed(content)
                        except Exception as e:
                            errors += 1
                            if errors <= 3:
                                logger.warning("limbic: reindex embed error: %s", e)
                            continue
                        if not emb or len(emb) != new_dims:
                            errors += 1
                            continue
                        embedded.append((r["fact_id"], emb))
                cur.close()  # release the read txn before BEGIN (a live cursor holds it)

                if total == 0:
                    return {"reindexed": 0, "status": "complete",
                            "note": "no facts to reindex"}

                # Never swap a good index for an empty one: if every embed
                # failed the embedder is broken — leave the old index alone.
                if not embedded and errors:
                    return {"error": f"all {errors} embeddings failed; "
                                     "old vec_index left untouched",
                            "status": "failed"}

                logger.info("limbic: reindex — %d facts read, %d embedded, %d errors; "
                            "swapping vec_index (%d-dim, cosine)",
                            total, len(embedded), errors, new_dims)

                try:
                    self._db.execute("BEGIN IMMEDIATE")
                    self._db.execute("DROP TABLE IF EXISTS vec_index")
                    self._db.execute(
                        f"CREATE VIRTUAL TABLE vec_index USING vec0(embedding float[{new_dims}] distance_metric=cosine)"
                    )
                    self._db.executemany(
                        "INSERT INTO vec_index(rowid, embedding) VALUES (?, ?)",
                        ((fid, _serialize_vec(emb)) for fid, emb in embedded),
                    )
                    self._db.commit()
                except sqlite3.DatabaseError:
                    self._db.rollback()
                    raise

                self._dims = new_dims
                elapsed = time.time() - t0
                logger.info("limbic: reindex — %d stored in %.1fs", len(embedded), elapsed)

                return {
                    "reindexed": len(embedded),
                    "errors": errors,
                    "skipped_blank": skipped_blank,
                    "elapsed_s": round(elapsed, 1),
                    "new_dims": new_dims,
                    "status": "complete"
                }

        except sqlite3.DatabaseError:
            raise
        except Exception as e:
            logger.error("limbic: reindex failed: %s", e)
            return {"error": str(e), "status": "failed"}

    # ── Lifecycle ──────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._db.commit()
                self._db.close()
                logger.debug("limbic: SQLite closed cleanly")
            except Exception as e:
                logger.debug("limbic: close error: %s", e)

    def backup(self, target_dir: str) -> str:
        """Consistent snapshot via `VACUUM INTO` into target_dir.

        The live .db + -wal/-shm sidecar pair must NOT be copied file-by-file
        while the gateway is running (that is how torn backups happen);
        VACUUM INTO produces a self-contained, fully committed .db snapshot.
        Returns the snapshot path; fails if the target already exists.
        """
        os.makedirs(target_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        target = os.path.join(target_dir, f"limbic-backup-{stamp}.db")
        def _txn():
            self._db.execute("VACUUM INTO ?", (target,))
        self._write_txn(_txn)
        logger.info("limbic: backed up %s -> %s", self._db_path, target)
        return target

    def backup_paths(self) -> list[str]:
        """Paths to include in a backup — a VACUUM INTO snapshot.

        The raw live .db is deliberately NOT returned: while WAL sidecars
        exist, copying the bare .db can yield a torn/inconsistent image.
        """
        return [self.backup(os.path.join(os.path.dirname(self._db_path), "backups"))]

    # ── Embedding retrieval (for centroid computation) ─────────────

    def get_embedding(self, fact_id: int):
        """One stored embedding, or None. Used by the provider's relevance
        gate for FTS-only facts that never came through the vector KNN path."""
        with self._lock:
            try:
                row = self._db.execute(
                    "SELECT embedding FROM vec_index WHERE rowid = ?",
                    (fact_id,)
                ).fetchone()
                return _deserialize_vec(row["embedding"]) if row else None
            except sqlite3.DatabaseError as e:
                logger.error("limbic: get_embedding failed (possible corruption): %s", e)
                return None
            except Exception as e:
                logger.debug("limbic: get_embedding failed: %s", e)
                return None

    def get_embeddings(self, user_id: str) -> list[list[float]]:
        """Get all embeddings for a user. Used by the provider's centroid cache."""
        with self._lock:
            try:
                rows = self._db.execute(
                    """SELECT v.embedding FROM vec_index v
                       JOIN facts f ON v.rowid = f.fact_id
                       WHERE f.user_id = ?""",
                    (user_id,)
                ).fetchall()
                return [_deserialize_vec(r["embedding"]) for r in rows]
            except sqlite3.DatabaseError as e:
                logger.error("limbic: get_embeddings failed (possible corruption): %s", e)
                return []
            except Exception as e:
                logger.debug("limbic: get_embeddings failed: %s", e)
                return []