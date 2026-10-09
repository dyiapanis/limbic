"""Limbic embeddings — bundled ONNX models. Self-contained, no external services.

The sensory input layer. Converts text to vectors.

Two ONNX models run in-process on CPU via onnxruntime:
  - Arctic Embed 2.0 L int8 (embedding) — XLM-RoBERTa, 1024 dims, 74 languages
  - MiniLMv2-L6-mnli-xnli (NLI) — see nli.py

No external embedding server. No endpoint configuration. No fastembed dependency.
Models download from HuggingFace on first use and cache locally.

Note: RAG deployments keep their own embeddings infrastructure independently;
Limbic is fully self-contained and does not
depend on it.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Model metadata ──────────────────────────────────────────────

EMBEDDING_MODEL_ID = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_MODEL_SIZE_GB = 0.57  # int8 single-file download (fp32 was 2.11GB)
EMBEDDING_MODEL_NAME = "Arctic Embed 2.0 L int8"
EMBEDDING_DIMS = 1024
EMBEDDING_MAX_SEQ_LENGTH = 512  # XLM-RoBERTa supports 8192 but 512 is sufficient for facts

# Arctic Embed 2.0 uses query/passage prefixes for retrieval tasks.
# For memory facts (documents/passages), no prefix is needed.
# For queries (thalamus prefetch), the "query: " prefix is recommended.


class LimbicEmbedder:
    """Embedding interface."""

    @property
    def dims(self) -> int:
        raise NotImplementedError

    def embed(self, content: str, is_query: bool = False) -> Optional[list[float]]:
        raise NotImplementedError

    @property
    def cache_hits(self) -> int:
        return 0

    @property
    def cache_misses(self) -> int:
        return 0

    def reset_cache_counters(self):
        pass


def _urlretrieve_atomic(url: str, dest: str, min_bytes: int = 1024,
                        expect_sha256: str | None = None):
    """Download to a temp file, verify, then rename into place.

    Guards the failure modes of a bare urlretrieve:
    - partial/torn writes leave no final file (temp is discarded)
    - a concurrent downloader never sees a half-written cache entry
    - a truncated-but-"complete" response (empty/tiny file) is rejected
    - content that doesn't match the pinned sha256 is rejected (when given)
    """
    import urllib.request
    # Rule-12 hardening (unattended runs fail cleanly, never hang): no global
    # socket timeout means a stalled HuggingFace connection hangs remember()
    # indefinitely under cron/gateway. 60s connect/read stall bound.
    import socket
    old_timeout = socket.getdefaulttimeout()
    if old_timeout is None:
        socket.setdefaulttimeout(60)
    # PID-unique temp name: two cold-start downloaders (gateway + cron worker)
    # sharing ONE .part name clobber each other — the loser's urlretrieve hits
    # ENOENT at rename time (seen live 2026-10-07 during the int8 rollout).
    tmp = f"{dest}.{os.getpid()}.part"
    try:
        urllib.request.urlretrieve(url, tmp)
        if os.path.exists(tmp) and os.path.getsize(tmp) < min_bytes:
            raise RuntimeError(f"download too small ({os.path.getsize(tmp)} bytes) — refusing to cache")
        if expect_sha256:
            import hashlib
            h = hashlib.sha256()
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            got = h.hexdigest()
            if got != expect_sha256:
                os.remove(tmp)
                raise RuntimeError(
                    f"sha256 mismatch for {url}: got {got}, want {expect_sha256} — refusing to cache")
        os.replace(tmp, dest)
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        logger.error("limbic: failed to download %s: %s", url, e)
        raise
    finally:
        if old_timeout is None:
            socket.setdefaulttimeout(None)


class OnnxEmbedder(LimbicEmbedder):
    """Bundled ONNX embedding model. Runs in-process on CPU.

    Downloads Arctic Embed 2.0 L (int8-quantized ONNX) from HuggingFace on
    first use (single ~570MB file). Quantization is Snowflake's own published
    artifact; retrieval equivalence to fp32 verified on this repo's eval suite
    (identical hit rates, 18/18 rank agreement) — see model_pins.py docstring.
    Mean pooling + L2 normalisation (XLM-RoBERTa base).
    """

    # Cache layout: int8 is a single self-contained file (no external data),
    # cached under its own name so an existing fp32 cache coexists harmlessly.
    _MODEL_FILE = "model_int8.onnx"

    def __init__(self, cache_dir: Optional[str] = None, dims: int = EMBEDDING_DIMS):
        self._session = None
        self._tokenizer = None
        self._dims = dims
        self._cache: dict[str, list[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_dir = cache_dir or self._default_cache_dir()

    @staticmethod
    def _default_cache_dir() -> str:
        """Default cache directory — ~/.cache/limbic/embedding/"""
        return os.path.join(os.path.expanduser("~"), ".cache", "limbic", "embedding")

    def _ensure_model(self):
        """Download and load the ONNX model on first use."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            logger.error(
                "limbic: onnxruntime and tokenizers are required. "
                "Install: pip install onnxruntime tokenizers"
            )
            raise

        model_path = os.path.join(self._cache_dir, self._MODEL_FILE)
        tokenizer_path = os.path.join(self._cache_dir, "tokenizer.json")

        if not os.path.exists(model_path):
            self._download_model()

        if not os.path.exists(model_path):
            raise RuntimeError(f"limbic: embedding model not found at {model_path}")

        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=EMBEDDING_MAX_SEQ_LENGTH)

        logger.info(
            "limbic: embedding model loaded (%s, %d dims)",
            EMBEDDING_MODEL_NAME,
            self._dims,
        )

    def _download_model(self):
        """Download ONNX model files from HuggingFace at a PINNED revision with sha256 checks.

        The int8 ONNX engine is a single self-contained file under onnx/
        (no external-data companion). Tokenizer and config files are at the
        repository root.
        """
        os.makedirs(self._cache_dir, exist_ok=True)

        from .model_pins import model_pin, file_hash, is_sha1_blob
        pin = model_pin(EMBEDDING_MODEL_ID)
        rev = pin["revision"]
        file_hashes = pin["files"]

        root_url = f"https://huggingface.co/{EMBEDDING_MODEL_ID}/resolve/{rev}"
        onnx_url = f"{root_url}/onnx"

        # ONNX engine — under onnx/ subpath (cache layout keeps the flat name)
        onnx_files = [
            self._MODEL_FILE,
        ]
        for fname in onnx_files:
            dest = os.path.join(self._cache_dir, fname)
            expect = file_hashes[f"onnx/{fname}"]
            if os.path.exists(dest):
                self._verify_cached(dest, expect, is_sha1_blob(expect), fname)
                if os.path.exists(dest):
                    continue
            url = f"{onnx_url}/{fname}"
            logger.info(
                "limbic: downloading %s (%.1fGB total model)...",
                fname, EMBEDDING_MODEL_SIZE_GB,
            )
            _urlretrieve_atomic(url, dest,
                                expect_sha256=None if is_sha1_blob(expect) else expect)

        # Tokenizer and config files — at repository root
        root_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "config.json",
        ]
        for fname in root_files:
            dest = os.path.join(self._cache_dir, fname)
            expect = file_hashes[fname]
            if os.path.exists(dest):
                self._verify_cached(dest, expect, is_sha1_blob(expect), fname)
                if os.path.exists(dest):
                    continue
            url = f"{root_url}/{fname}"
            logger.info("limbic: downloading %s...", fname)
            _urlretrieve_atomic(url, dest,
                                expect_sha256=None if is_sha1_blob(expect) else expect)

        logger.info("limbic: embedding model files verified at pinned revision %s", rev[:12])
        self._gc_cache()

    def _gc_cache(self):
        """Delete cache-dir files not in the pin manifest (stale orphans from a
        previous model/quantization — e.g. the fp32 model.onnx + model.onnx_data
        a 0.5.x→0.6.0 upgrader carries, ~2.27GB). Only ever touches the model
        cache dir; the DB and Limbic state live elsewhere."""
        from .model_pins import model_pin
        pin = model_pin(EMBEDDING_MODEL_ID)
        allowed = {k.rsplit("/", 1)[-1] for k in pin["files"]}
        self._gc_stale_files(self._cache_dir, allowed)

    @staticmethod
    def _gc_stale_files(cache_dir: str, allowed: set[str]):
        if not os.path.isdir(cache_dir):
            return
        for name in os.listdir(cache_dir):
            path = os.path.join(cache_dir, name)
            if os.path.isfile(path) and name not in allowed:
                size_mb = os.path.getsize(path) / (1 << 20)
                try:
                    os.remove(path)
                    logger.info("limbic: cache GC removed stale non-pinned file %s (%.0fMB)", name, size_mb)
                except OSError as e:
                    logger.warning("limbic: cache GC could not remove %s: %s", path, e)

    @staticmethod
    def _verify_cached(dest: str, expect: str, is_sha1: bool, fname: str):
        """Verify an existing cache file against its pin; delete on mismatch (self-heal
        re-downloads on next start)."""
        import hashlib
        if is_sha1:
            # HF pins non-LFS files by git BLOB oid = sha1(b"blob <size>\0" + content),
            # NOT plain sha1(content). Hash the blob form or every valid cache file
            # is rejected and re-downloaded on every start.
            h = hashlib.sha1()
            h.update(b"blob %d\x00" % os.path.getsize(dest))
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != expect.removeprefix("sha1:"):
                logger.warning("limbic: cached %s fails sha1 blob-oid pin — removing for re-download", fname)
                os.remove(dest)
            return
        h = hashlib.sha256()
        with open(dest, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != expect:
            logger.warning("limbic: cached %s fails sha256 pin (%s != %s) — removing",
                           fname, got[:12], expect[:12])
            os.remove(dest)

    def embed(self, content: str, is_query: bool = False) -> Optional[list[float]]:
        """Generate embedding for a text.

        Args:
            content: Text to embed.
            is_query: If True, prepend "query: " prefix (recommended for
                retrieval queries). If False, no prefix (for stored facts).

        Returns:
            Normalised embedding vector (L2 norm = 1.0), or None on error.
        """
        if not content or not content.strip():
            return None

        prefix = "query: " if is_query else ""
        cache_key = hashlib.md5(f"{prefix}{content}".encode()).hexdigest()
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        self._ensure_model()

        try:
            import numpy as np

            text = f"{prefix}{content}" if prefix else content
            encoding = self._tokenizer.encode(text)
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )

            # Mean pooling — weighted by attention mask
            token_embeddings = outputs[0]  # [1, seq_len, hidden_dim]
            mask_expanded = attention_mask[:, :, None].astype(np.float32)
            summed = (token_embeddings * mask_expanded).sum(axis=1)
            counts = mask_expanded.sum(axis=1)
            counts = np.maximum(counts, 1e-9)

            embedding = (summed / counts)[0]  # [hidden_dim]

            # L2 normalise
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            vec = embedding.tolist()
            # ponytail: naive cap, evict oldest — embed cache on a long-lived
            # gateway grows forever (each entry ~8KB at 1024 dims).
            if len(self._cache) >= 20_000:
                for k in list(self._cache)[:5000]:
                    del self._cache[k]
            self._cache[cache_key] = vec
            return vec

        except Exception as e:
            logger.warning("limbic: embedding generation failed: %s", e)
            return None

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def reset_cache_counters(self):
        self._cache_hits = 0
        self._cache_misses = 0


def create_embedder(config: dict) -> LimbicEmbedder:
    """Factory — creates the bundled ONNX embedder.

    Only one implementation: OnnxEmbedder (Arctic Embed 2.0 L int8).
    No endpoint option. No external services. Self-contained.
    """
    emb_config = config.get("embedding", {})
    cache_dir = emb_config.get("cache_dir")
    dims = emb_config.get("dims", EMBEDDING_DIMS)

    return OnnxEmbedder(cache_dir=cache_dir, dims=dims)