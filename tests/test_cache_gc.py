
"""Cache GC — stale non-pinned files are removed at model-verification time."""
import os


def _make_iso_cache(tmp_path):
    """Isolated cache dir mimicking the layout: pinned aux files + planted orphans."""
    from limbic.embeddings import OnnxEmbedder
    iso = str(tmp_path / "embedding")
    os.makedirs(iso, exist_ok=True)
    # real pinned aux content does not matter for GC (it checks names, not hashes)
    for f in ["tokenizer_config.json", "special_tokens_map.json", "config.json"]:
        open(os.path.join(iso, f), "wb").write(b"{}")
    # planted orphans: legacy fp32 pair + random file
    open(os.path.join(iso, "model.onnx"), "wb").write(b"fp32-legacy")
    open(os.path.join(iso, "model.onnx_data"), "wb").write(b"fp32-legacy-data")
    open(os.path.join(iso, "orphan.bin"), "wb").write(b"x" * 2048)
    return OnnxEmbedder(cache_dir=iso), iso


def test_gc_removes_non_pinned_files(tmp_path):
    e, iso = _make_iso_cache(tmp_path)
    e._gc_cache()
    left = set(os.listdir(iso))
    assert "model.onnx" not in left
    assert "model.onnx_data" not in left
    assert "orphan.bin" not in left


def test_gc_keeps_pinned_files(tmp_path):
    e, iso = _make_iso_cache(tmp_path)
    e._gc_cache()
    left = set(os.listdir(iso))
    # the int8 engine + aux files survive (not deleted even though model_int8.onnx absent here)
    for kept in ["config.json", "tokenizer_config.json", "special_tokens_map.json"]:
        assert kept in left, f"pinned file {kept} wrongly deleted"


def test_gc_noop_on_missing_dir():
    from limbic.embeddings import OnnxEmbedder
    # must not raise when the cache dir does not exist yet (fresh install)
    OnnxEmbedder._gc_stale_files("/tmp/phoenix/gc_nonexistent_dir_zz", {"anything"})


def test_nli_gc_uses_shared_helper(tmp_path):
    """NLI side uses the same _gc_stale_files policy — planted orphan dies, pinned survives."""
    import sys
    sys.path.insert(0, "/home/phoenix/limbic/..")
    from limbic.nli import LimbicNLI
    from limbic.model_pins import model_pin, NLI_MODEL_ID
    iso = str(tmp_path / "nli")
    os.makedirs(iso, exist_ok=True)
    allowed = {k.rsplit("/", 1)[-1] for k in model_pin(NLI_MODEL_ID)["files"]}
    open(os.path.join(iso, "config.json"), "wb").write(b"{}")          # pinned
    open(os.path.join(iso, "stale_nli_fp32.bin"), "wb").write(b"x" * 128)  # orphan
    from limbic.embeddings import OnnxEmbedder
    OnnxEmbedder._gc_stale_files(iso, allowed)
    left = set(os.listdir(iso))
    assert "stale_nli_fp32.bin" not in left
    assert "config.json" in left
