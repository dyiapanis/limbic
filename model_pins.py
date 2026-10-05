"""Limbic HF model integrity pins.

Each pinned model: immutable commit SHA (every URL resolves through /resolve/<sha>/) and
per-file sha256 (HF LFS oids = sha256 of the file content). Non-LFS small files use the
git blob oid (sha1) — verified as 'sha1:<oid>'.

Update procedure: bump the commit sha here, fetch per-file oids from
`https://huggingface.co/api/models/<id>/tree/<sha>?recursive=true`, and refresh
_MODELS. A mismatching local cache file is rejected and re-downloaded (self-heal),
so a bad pin cannot poison existing deployments.
"""

# Model 1 — embeddings: Snowflake Arctic Embed 2.0 L
EMBEDDING_MODEL_ID = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_REVISION = "ac6544c8a46e00af67e330e85a9028c66b8cfd9a"

# Model 2 — NLI (reconsolidation trigger): MoritzLaurer multilingual-MiniLMv2-L6
NLI_MODEL_ID = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
NLI_REVISION = "0a71e92a985b6e1ad1828cf67ce9c459639c1dca"

# sha256 for LFS-backed files; "sha1:<oid>" for plain git blobs (small json files).
_MODELS: dict[str, dict] = {
    EMBEDDING_MODEL_ID: {
        "revision": EMBEDDING_REVISION,
        "files": {
            "onnx/model.onnx": "f74aa79745ccfb1e75daa7e8e6552a78402d4de193eb8ca67a931358d3e0a25e",
            "onnx/model.onnx_data": "fe7d75ff258fbda10a6bea63c5422df5579d625355b1aca69ba6923c0ba604a9",
            "tokenizer.json": "39feb9863a378165ab9c5c689047203d789422966c0c58721c5309fd039a8edc",
            "tokenizer_config.json": "sha1:bda426376d27170ae0a14e7d4cf4087efdc39eb5",
            "special_tokens_map.json": "sha1:b1879d702821e753ffe4245048eee415d54a9385",
            "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            "config.json": "sha1:91a1faf6a9239179d70a70347628b182fa6a0cfb",
        },
    },
    NLI_MODEL_ID: {
        "revision": NLI_REVISION,
        "files": {
            "onnx/model.onnx": "79f8cda2b1230585a95ea0514a6f1bd21c5c986ba0529bb3261213a3e195fa6e",
            "tokenizer.json": "098c131bb4423163db239755e309facaa6850059f850f9f3d88a78344a4b631c",
            "tokenizer_config.json": "sha1:9aae6d8fa4201fe47deabe3c13e47cc684b02258",
            "special_tokens_map.json": "sha1:d5698132694f4f1bcff08fa7d937b1701812598e",
            "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
            "config.json": "sha1:43a6b699feac31f6eda6b77d6d605e7b4ebdb86e",
        },
    },
}


def model_pin(model_id: str) -> dict:
    """Pin (revision + file→hash table) for a model id, or raise KeyError."""
    return _MODELS[model_id]


def file_hash(path: str) -> dict:
    """Stream a file and return sha256 hex + size."""
    import hashlib
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "size": size}


def is_sha1_blob(expected: str) -> bool:
    return expected.startswith("sha1:")