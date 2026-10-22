"""Owner-initiated removal — the `forget` tool (two-step delete + receipts).

Model-free (fake embedder/NLI, real SQLiteStorage) — same fixture pattern
as test_provider_tools.py.

Covers:
  - query step lists candidates (ids + excerpts, nothing deleted)
  - fact_id step deletes user-scoped, returns removed content
  - receipt is hash-only: sha256 + chars, never the content itself, 0600
  - one user's forget can never remove another user's fact
  - tool dispatch: exactly-one-of query/fact_id; session threading
  - receipt-write failure never blocks the deletion (delete stands)
"""
import hashlib
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.tools import handle_tool_call  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8


class FakeEmbedder:
    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    @property
    def cache_hits(self) -> int:
        return 0


class FakeNLI:
    def classify(self, old, new):
        return "neutral", 0.0


@pytest.fixture()
def provider(tmp_path):
    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    p._initialised = True
    yield p
    p._storage.close()


def _receipt_path(provider):
    return provider._storage._db_path + ".deletions.jsonl"


def test_query_step_lists_candidates_nothing_deleted(provider):
    fid1 = provider.remember("The HP laptop costs 800 dollars", user_id="alice")["fact_id"]
    provider.remember("Alice likes running in the mornings", user_id="alice")
    out = provider.forget_memory(query="HP laptop", user_id="alice")
    assert out["status"] == "candidates"
    assert out["count"] >= 1
    ids = [c["fact_id"] for c in out["candidates"]]
    assert fid1 in ids
    for c in out["candidates"]:
        assert c["excerpt"]
        assert "content" not in c  # excerpt only, never full content in listing
    # Nothing was deleted by the listing step
    assert provider._storage.get(fid1) is not None


def test_fact_id_step_deletes_scoped_and_writes_hash_only_receipt(provider, tmp_path):
    fid = provider.remember("secret: the wifi password is hunter2", user_id="alice")["fact_id"]
    out = provider.forget_memory(fact_id=fid, user_id="alice")
    assert out["removed"] is True
    assert out["fact_id"] == fid
    assert "hunter2" in out["content"]  # confirmation shows what was removed

    # Gone from the store
    assert provider._storage.get(fid) is None

    # Receipt: hash-only, no content, 0600
    rp = _receipt_path(provider)
    assert os.path.exists(rp)
    if os.name == "posix":  # windows has no posix mode bits; chmod is a no-op there
        assert os.stat(rp).st_mode & 0o777 == 0o600
    raw = open(rp).read()
    assert "hunter2" not in raw  # the secret must not be retained
    line = json.loads(raw.strip().splitlines()[-1])
    assert line["user_id"] == "alice"
    assert line["fact_id"] == fid
    assert line["content_sha256"] == hashlib.sha256(
        b"secret: the wifi password is hunter2").hexdigest()
    assert line["content_chars"] == len("secret: the wifi password is hunter2")


def test_one_user_cannot_delete_anothers_fact(provider):
    fid = provider.remember("bob private fact about taxes", user_id="bob")["fact_id"]
    out = provider.forget_memory(fact_id=fid, user_id="alice")
    assert "error" in out
    assert provider._storage.get(fid) is not None  # untouched


def test_exactly_one_of_query_fact_id_required(provider):
    assert "error" in provider.forget_memory(query="x", fact_id=1, user_id="alice")
    assert "error" in provider.forget_memory(user_id="alice")
    assert "error" in provider.forget_memory(user_id=None)


def test_tool_dispatch_two_step(provider, monkeypatch):
    provider.remember("the eiffel tower fact for dispatch", user_id="alice")
    # Dispatch test, not identity: stub the resolver (the real manifest
    # canonicalises freely and is owned by identity tests).
    monkeypatch.setattr(provider, "_get_user_id", lambda session_id: "alice")
    r = json.loads(handle_tool_call(provider, "forget", {"query": "eiffel"},
                                    session_id="sess-1"))
    assert r["status"] == "candidates"
    fid = r["candidates"][0]["fact_id"]
    r2 = json.loads(handle_tool_call(provider, "forget", {"fact_id": fid},
                                     session_id="sess-1"))
    assert r2["removed"] is True
    # mutual exclusion at the dispatch layer
    r3 = json.loads(handle_tool_call(provider, "forget", {"query": "a", "fact_id": fid}))
    assert "error" in r3
    r4 = json.loads(handle_tool_call(provider, "forget", {}))
    assert "error" in r4


def test_receipt_failure_never_blocks_deletion(provider, tmp_path, monkeypatch):
    fid = provider.remember("fact whose receipt write will fail", user_id="alice")["fact_id"]
    monkeypatch.setattr(provider._storage, "_db_path", "")
    out = provider.forget_memory(fact_id=fid, user_id="alice")
    assert out["removed"] is True  # delete stands even if the receipt fails
    assert provider._storage.get(fid) is None


def test_fts_fallback_when_embedder_gone(provider, monkeypatch):
    # Embedder may be unavailable while old facts remain: candidates must
    # still surface via FTS alone (no silent empty result).
    fid = provider.remember("the lighthouse keeper stores lamp oil", user_id="alice")["fact_id"]
    monkeypatch.setattr(provider, "_embedder", None)
    out = provider.forget_memory(query="lighthouse keeper", user_id="alice")
    assert out["status"] == "candidates"
    assert fid in [c["fact_id"] for c in out["candidates"]]