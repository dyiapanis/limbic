"""Regression tests for the teknium1 re-review round (PR #131827):

1. Pin keys must match what each downloader actually looks up: nli.py fetches
   the ONNX engine from onnx/ (LFS-pinned as "onnx/model.onnx") and the aux
   files from the repo root (bare keys); embeddings.py likewise — ONNX files
   with "onnx/" prefix, aux files bare. A missing key = KeyError at runtime.
2. _verify_cached sha1 branch must hash the GIT BLOB OID form
   (b"blob <size>\\0" + data), not plain sha1 — plain sha1 rejects every valid
   cached non-LFS file (all five json pins per model) and re-downloads each start.
3. After deleting a mismatching cached file, the SAME loop iteration must reach
   the download call (no early continue past a deleted file).
"""
import hashlib
import importlib.util
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, rel_path: str):
    """Import a submodule without triggering limbic/__init__ (which imports
    provider.py → agent.memory_provider, unavailable outside Hermes)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO / rel_path, submodule_search_locations=[str(REPO)]
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pin_keys_resolve_for_both_downloaders():
    mp = _load("model_pins", "model_pins.py")
    nli = _load("limbic.nli", "nli.py")
    emb = _load("limbic.embeddings", "embeddings.py")

    # nli.py: engine key is onnx/-prefixed, aux keys bare
    nli_pin = mp.model_pin(nli.NLI_MODEL_ID)
    assert "onnx/model.onnx" in nli_pin["files"]
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "sentencepiece.bpe.model", "config.json"]:
        assert fname in nli_pin["files"], f"nli aux lookup '{fname}' missing in pins"
        assert not fname.startswith("onnx/")

    # embeddings.py: engine keys onnx/-prefixed, aux keys bare
    emb_pin = mp.model_pin(emb.EMBEDDING_MODEL_ID)
    for key in ["onnx/model.onnx", "onnx/model.onnx_data"]:
        assert key in emb_pin["files"], f"embeddings engine lookup '{key}' missing"
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                  "sentencepiece.bpe.model", "config.json"]:
        assert fname in emb_pin["files"]


def test_verify_cached_sha1_is_blob_oid(tmp_path):
    emb = _load("limbic.embeddings", "embeddings.py")
    data = b'{"hello": "world"}\n'
    p = tmp_path / "config.json"
    p.write_bytes(data)
    plain = hashlib.sha1(data).hexdigest()
    blob = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()

    emb.OnnxEmbedder._verify_cached(str(p), "sha1:" + blob, True, "test/config.json")
    assert p.exists(), "blob-oid sha1 verify must ACCEPT a valid pinned git blob"

    emb.OnnxEmbedder._verify_cached(str(p), "sha1:" + plain, True, "test/config.json")
    assert not p.exists(), "plain-sha1 mismatch must DELETE for re-download"


def test_mismatch_delete_falls_through_to_download():
    """The exists-branch must re-check existence after verify so a file the
    verifier just deleted is downloaded in the same pass (no bare continue)."""
    for rel, tag in [("nli.py", "NLI"), ("embeddings.py", "embeddings")]:
        src = (REPO / rel).read_text()
        assert re.search(
            r"if os\.path\.exists\(dest\):\n\s+(?:.*_verify_cached\(.*\n)+?\s+if os\.path\.exists\(dest\):\n\s+continue",
            src,
        ), f"{tag}: verify must re-check existence before continue"


def test_pins_match_live_hf_tree():
    """Every LFS pin matches the lfs oid in the live tree; every sha1 pin matches
    the git blob oid of the file at the pinned revision (fetched + hashed)."""
    mp = _load("model_pins", "model_pins.py")
    import json
    import urllib.request

    for model_id, pin in mp._MODELS.items():
        url = f"https://huggingface.co/api/models/{model_id}/tree/{pin['revision']}?recursive=true"
        with urllib.request.urlopen(url, timeout=30) as r:
            tree = json.load(r)
        by_path = {f["path"]: f for f in tree}

        for key, expect in pin["files"].items():
            entry = by_path.get(key)
            assert entry, f"{model_id}:{key} not in the pinned tree"
            if expect.startswith("sha1:"):
                # non-LFS: fetch raw content, compute blob oid
                raw = f"https://huggingface.co/{model_id}/raw/{pin['revision']}/{key}"
                with urllib.request.urlopen(raw, timeout=60) as r:
                    data = r.read()
                got = blob_sha1(data)
                assert got == expect.removeprefix("sha1:"), (
                    f"{model_id}:{key} blob-oid drift ({expect[:20]} → {got[:20]})"
                )
            else:
                lfs = (entry.get("lfs") or {}).get("oid", "")
                assert lfs == expect, f"{model_id}:{key} LFS oid drift"


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()