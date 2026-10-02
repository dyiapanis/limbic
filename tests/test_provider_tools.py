"""Provider & tools tests — model-free (fake embedder/NLI, real SQLiteStorage).

Run:
    cd <hermes-plugins-parent> && \
    python -m pytest limbic/tests/test_provider_tools.py -v
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.tools import handle_tool_call, MAX_CONTENT  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402

DIMS = 8


class FakeEmbedder:
    def __init__(self, dims=DIMS):
        self.dims = dims
        self.calls = 0

    def embed(self, content, is_query=False):
        self.calls += 1
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


def test_tool_rejects_oversized_content(provider):
    r = json.loads(handle_tool_call(provider, "remember", {"content": "x" * (MAX_CONTENT + 1)}))
    assert "error" in r and "too long" in r["error"]


def test_tool_rejects_non_string(provider):
    r = json.loads(handle_tool_call(provider, "remember", {"content": 12345}))
    assert "error" in r


def test_tool_exception_becomes_error_json(provider):
    provider.remember = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    r = json.loads(handle_tool_call(provider, "remember", {"content": "hello world fact"}))
    assert "error" in r and "RuntimeError" in r["error"]


def test_tool_rejects_empty(provider):
    r = json.loads(handle_tool_call(provider, "remember", {"content": "   "}))
    assert "error" in r


def test_session_id_threaded_to_remember(provider):
    seen = {}

    def fake_remember(content, user_id=None, session_id=None):
        seen["session_id"] = session_id
        return {"status": "stored", "fact_id": 1}

    provider.remember = fake_remember
    handle_tool_call(provider, "remember", {"content": "session threading test"}, session_id="sess-42")
    assert seen["session_id"] == "sess-42"


def test_session_id_threaded_to_recall(provider):
    seen = {}

    def fake_recall(query, session_id=None):
        seen["session_id"] = session_id
        return {"results": []}

    provider.recall = fake_recall
    handle_tool_call(provider, "recall", {"query": "anything"}, session_id="sess-99")
    assert seen["session_id"] == "sess-99"


def test_forget_fact_requires_user(provider):
    r = provider.forget_fact(1, user_id=None)
    assert "error" in r


def test_forget_fact_ownership_mismatch(provider):
    fid = provider.remember("ownership test fact", user_id="alice")["fact_id"]
    r = provider.forget_fact(fid, user_id="bob")
    assert "error" in r
    assert provider._storage.get(fid) is not None


def test_forget_fact_owner_can_delete(provider):
    fid = provider.remember("ownership test fact", user_id="alice")["fact_id"]
    r = provider.forget_fact(fid, user_id="alice")
    assert r.get("removed") is True
    assert provider._storage.get(fid) is None


def test_no_content_in_logs(provider, caplog):
    import logging
    with caplog.at_level(logging.DEBUG, logger="limbic"):
        provider.remember("very unique secret phrase zqxjwv", user_id="alice")
        provider.recall("very unique secret phrase zqxjwv")
    for record in caplog.records:
        assert "zqxjwv" not in record.getMessage(), f"PII leak: {record.getMessage()}"


def test_config_schema_matches_implementation(provider):
    schema_keys = {f["key"] for f in provider.get_config_schema()}
    assert "embedding_provider" not in schema_keys
    assert "embedding_model" not in schema_keys
    assert "memory_horizon" not in schema_keys
    dims_field = [f for f in provider.get_config_schema() if f["key"] == "embedding_dims"][0]
    assert dims_field["default"] == 1024


def test_known_entities_from_config(provider):
    provider._config["known_entities"] = ["PERSON:Zqxjwv"]
    patterns = []
    for entry in provider._config.get("known_entities", []):
        label, _, pat = str(entry).partition(":")
        if label and pat:
            patterns.append({"label": label.strip(), "pattern": pat.strip()})
    assert {"label": "PERSON", "pattern": "Zqxjwv"} in patterns


def test_entities_ship_empty():
    import limbic.entities as entities
    assert entities.KNOWN_PATTERNS == []


def test_known_users_from_storage(provider):
    provider.remember("fact one for carol", user_id="carol")
    provider.remember("fact two for dave", user_id="dave")
    users = provider._get_known_users()
    assert "carol" in users and "dave" in users


def test_guardian_recall_sees_ward_facts(provider):
    """The guardian gate must widen scope when guardian_of is non-empty.

    Regression test: the condition was `if user_id in guardian_of` — always
    False (a guardian is never their own ward) — so guardians could never
    see ward facts. The gate must test `if guardian_of:`.
    """
    provider._session_user_id["sess-g"] = "alice"
    provider._session_guardian_of["sess-g"] = ["bob"]
    provider._current_user_id = "alice"
    provider.remember("bob likes trains choo choo", session_id="sess-g", user_id="bob")

    # alice's recall on her own session must surface bob's fact (ward scope)
    results = provider._search("trains", "alice", session_id="sess-g")
    contents = [r["content"] for r in results]
    assert any("trains" in c for c in contents), f"guardian recall missed ward fact: {contents}"

    # and a non-guardian session must NOT see bob's fact (scoping intact)
    results = provider._search("trains", "carol", session_id="sess-other")
    contents = [r["content"] for r in results]
    assert not any("trains" in c for c in contents), f"non-guardian saw ward fact: {contents}"
