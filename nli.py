"""Limbic NLI classifier — the reconsolidation trigger.

Detects contradiction between a new fact and existing facts. This is the
prediction-error signal that opens the reconsolidation window — the brain
detects mismatch between what it predicted and what it observed.

Architecture: cross-encoder (bi-text in, label out). Unlike embeddings
(bi-encoder), the cross-encoder processes both texts together so it can
capture their interaction — does A entail B, contradict B, or is it neutral
toward B?

Model: MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli
  - XLM-RoBERTa base (same family as Arctic Embed 2.0)
  - 107M parameters, ~430MB ONNX
  - 100+ languages (Greek, English, French, etc.)
  - MIT license
  - Trained on XNLI (15 languages) + MNLI
  - 87% accuracy on MNLI mismatched

Output: one of "contradiction", "entailment", "neutral"

Usage:
    classifier = create_nli_classifier()
    label = classifier.classify("I eat meat", "I am vegan")
    # → "contradiction"

Why not embeddings for contradiction detection:
    Embeddings capture topic, not stance. "I eat meat" and "I am vegan"
    have high cosine similarity because both are about diet. The cross-encoder
    sees both together and learns that "meat" and "vegan" are in tension.
    See trust.py is_correction() docstring for the full analysis.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Model metadata — used for download cache path and first-use logging
MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
NLI_MODEL_ID = MODEL_ID
MODEL_ONNX_SUBPATH = "onnx/model.onnx"
MODEL_SIZE_GB = 0.43

# Label mapping — model outputs 3 logits, index → label
# The model's config.json defines id2label as {0: "entailment", 1: "neutral", 2: "contradiction"}
# We verify this at load time from config.json
LABELS = ["entailment", "neutral", "contradiction"]

# Max sequence length — XLM-RoBERTa supports 512 tokens
# Fact content is typically short (1-2 sentences), so 256 is generous and keeps inference fast
MAX_SEQ_LENGTH = 256


class LimbicNLI:
    """NLI cross-encoder. Loads ONNX model on first use, runs in-process on CPU."""

    def __init__(self, cache_dir: Optional[str] = None):
        self._session = None
        self._tokenizer = None
        self._label_map: dict[int, str] = {}
        self._cache: dict[str, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_dir = cache_dir or self._default_cache_dir()

    @staticmethod
    def _default_cache_dir() -> str:
        """Default cache directory — ~/.cache/limbic/nli/"""
        return os.path.join(os.path.expanduser("~"), ".cache", "limbic", "nli")

    def _ensure_model(self):
        """Download and load the ONNX model on first use."""
        if self._session is not None:
            return

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError:
            logger.error("limbic NLI: onnxruntime and tokenizers are required. Install: pip install onnxruntime tokenizers")
            raise

        model_path = os.path.join(self._cache_dir, "model.onnx")
        tokenizer_path = os.path.join(self._cache_dir, "tokenizer.json")

        if not os.path.exists(model_path):
            self._download_model()

        if not os.path.exists(model_path):
            raise RuntimeError(f"limbic NLI: model file not found at {model_path}")

        # Load ONNX session
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)

        # Load label mapping from config.json
        config_path = os.path.join(self._cache_dir, "config.json")
        self._label_map = self._load_label_map(config_path)

        logger.info("limbic NLI: model loaded (labels=%s)", self._label_map)

        # Cache GC — every start (not only download), same policy as the embedder:
        # drop files not in the pin manifest. Pin keys mix "onnx/model.onnx" and
        # root names; cache layout is flat.
        from .model_pins import model_pin
        allowed = {k.rsplit("/", 1)[-1] for k in model_pin(NLI_MODEL_ID)["files"]}
        OnnxEmbedder._gc_stale_files(self._cache_dir, allowed)

    def _download_model(self):
        """Download ONNX model files from HuggingFace at a PINNED revision with sha256 checks."""
        os.makedirs(self._cache_dir, exist_ok=True)

        from .embeddings import _urlretrieve_atomic, OnnxEmbedder
        from .model_pins import model_pin, is_sha1_blob
        pin = model_pin(NLI_MODEL_ID)
        rev = pin["revision"]
        file_hashes = pin["files"]

        root_url = f"https://huggingface.co/{NLI_MODEL_ID}/resolve/{rev}"
        base_url = f"{root_url}/onnx"
        files = [
            "model.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "config.json",
        ]

        for fname in files:
            dest = os.path.join(self._cache_dir, fname)
            # Pin keys are HF-tree paths: the ONNX engine is LFS-pinned at
            # onnx/model.onnx; the aux files download from the repo root.
            # Cache layout is flat.
            expect = file_hashes["onnx/model.onnx" if fname == "model.onnx" else fname]
            if os.path.exists(dest):
                OnnxEmbedder._verify_cached(dest, expect, is_sha1_blob(expect), f"NLI/{fname}")
                if os.path.exists(dest):
                    continue
            url = f"{base_url}/{fname}" if fname == "model.onnx" else f"{root_url}/{fname}"
            logger.info("limbic NLI: downloading %s (%.0fMB total model)...", fname, MODEL_SIZE_GB * 1024)
            _urlretrieve_atomic(url, dest,
                                expect_sha256=None if is_sha1_blob(expect) else expect)

        logger.info("limbic NLI: all model files verified at pinned revision %s", rev[:12])

    @staticmethod
    def _load_label_map(config_path: str) -> dict[int, str]:
        """Load id2label mapping from config.json."""
        import json
        try:
            with open(config_path) as f:
                config = json.load(f)
            id2label = config.get("id2label", {})
            if id2label:
                return {int(k): v.lower() for k, v in id2label.items()}
        except Exception as e:
            logger.warning("limbic NLI: could not load config.json, using default labels: %s", e)

        # Default — matches the model's documented output order
        return {0: "entailment", 1: "neutral", 2: "contradiction"}

    def classify(self, premise: str, hypothesis: str) -> str:
        """Classify the relationship between two texts.

        Args:
            premise: The existing fact content.
            hypothesis: The new fact content.

        Returns:
            One of "contradiction", "entailment", "neutral".

        "contradiction" → the new fact contradicts the existing fact.
            This is the reconsolidation trigger — the existing fact's trust
            should be penalised so it fades naturally.
        "entailment" → the new fact is consistent with or elaborates the existing.
            No penalty. The new fact may prime the existing one.
        "neutral" → the new fact is unrelated despite topic overlap.
            No change. The similarity was coincidental.
        """
        if not premise or not hypothesis:
            return "neutral"

        # Cache by content hash of the pair
        cache_key = hashlib.md5(f"{premise}|||{hypothesis}".encode()).hexdigest()
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        self._cache_misses += 1

        self._ensure_model()

        try:
            import numpy as np

            # Tokenize the pair — cross-encoder processes both texts together
            # The tokenizer handles the [CLS] premise [SEP] hypothesis [SEP] format
            encoding = self._tokenizer.encode(premise, hypothesis)
            input_ids = np.array([encoding.ids], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

            # Run inference
            outputs = self._session.run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )

            # Output is logits [1, 3] — softmax to get probabilities
            logits = outputs[0]
            # Softmax
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Argmax → label index
            pred_idx = int(np.argmax(probs[0]))
            label = self._label_map.get(pred_idx, LABELS[pred_idx])

            # Cache
            self._cache[cache_key] = label

            return label

        except Exception as e:
            logger.warning("limbic NLI: classification failed: %s", e)
            return "neutral"  # safe default — no correction on error

    @property
    def available(self) -> bool:
        """Check if the model can be loaded (without actually loading it)."""
        try:
            model_path = os.path.join(self._cache_dir, "model.onnx")
            return os.path.exists(model_path)
        except Exception:
            return False

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    def reset_cache_counters(self):
        self._cache_hits = 0
        self._cache_misses = 0


def create_nli_classifier(config: dict | None = None) -> LimbicNLI:
    """Factory — creates the NLI classifier.

    Config is optional. If provided, nli.cache_dir can override the
    default download location.
    """
    nli_config = (config or {}).get("nli", {})
    cache_dir = nli_config.get("cache_dir")

    return LimbicNLI(cache_dir=cache_dir)