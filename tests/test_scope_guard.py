"""User-scope boundary guard (sprint C8): room-shaped user_ids are refused
at every reachable write path.

A room/platform id in facts.user_id is wrong-principal scope by definition:
recall resolves principals, so such facts are unreachable storage. Real
offender class: 7 facts written under a 26-char opaque Mattermost user id
just before that id was registered in the user manifest (the registration
landed hours after the last of those writes) — the L3 session-fallback
minted the raw id as a bucket. The guard refuses the shape regardless of
caller; the store-level checks are the backstop for non-provider writers.
The fixture ids are synthetic same-shape stand-ins for the audited rows.

Boundary: "!"-prefixed (Matrix internal room id) OR pure alphanumeric ≥18
chars (room-id length convention). Canonical keys, MXIDs ("@…"), "general",
"6" and "uid:1000" stay outside the net deliberately — they are shorter or
"@"-prefixed and belong to manifest re-attribution, not this guard.

Run:
    cd <repo-checkout> && python -m pytest tests/test_scope_guard.py -v
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.store_sqlite import SQLiteStorage, room_shaped_user_id  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402
from limbic.identity import UserManifest  # noqa: E402

DIMS = 8

# The orphan-bucket shapes the guard exists for (card-pinned): a 26-char
# opaque Mattermost user id (synthetic stand-in, same shape as the audited
# rows) and a Matrix internal room id. The content is a plainly durable infra
# fact so only the scope guard can reject it.
ROOM_MM_ID = "q" * 26
ROOM_MX_INTERNAL = "!abc123:unredacted.org"
DURABLE = "SimpleX renders markdown tables in the client"


class FakeEmbedder:
    def __init__(self, dims=DIMS):
        self.dims = dims

    def embed(self, content, is_query=False):
        v = [float((ord(c) % 13) + 1) for c in (content or "x")[:DIMS]]
        v += [1.0] * (self.dims - len(v))
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


class FakeNLI:
    def classify(self, old, new):
        return "neutral", 0.0


MANIFEST_USERS = {
    "grace": {"display_name": "Grace", "scope": "adult",
              "platform_ids": {}, "guardians": []},
    "bob": {"display_name": "Bob", "scope": "adult",
            "platform_ids": {}, "guardians": []},
}


def _provider(tmp_path, with_manifest=True):
    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=DIMS)
    if with_manifest:
        p._manifest = UserManifest("users.yaml (test fixture)", MANIFEST_USERS)
    p._initialised = True
    return p


@pytest.fixture()
def provider(tmp_path):
    p = _provider(tmp_path, with_manifest=True)
    yield p
    p._storage.close()


@pytest.fixture()
def bare_provider(tmp_path):
    p = _provider(tmp_path, with_manifest=False)
    yield p
    p._storage.close()


def _assert_rejected(p, out):
    assert out.get("status") == "rejected_room_scoped", out
    assert out.get("reason"), out


def test_predicates():
    assert room_shaped_user_id(ROOM_MM_ID)
    assert room_shaped_user_id(ROOM_MX_INTERNAL)
    assert room_shaped_user_id("A" * 18)
    assert room_shaped_user_id("a1B2c3D4e5F6g7H8i9")  # 18 alnum, mixed case
    assert not room_shaped_user_id("grace")
    assert not room_shaped_user_id("general")
    assert not room_shaped_user_id("@grace:unredacted.org")
    assert not room_shaped_user_id("")
    assert not room_shaped_user_id(None)


@pytest.mark.parametrize("room_id", [ROOM_MM_ID, ROOM_MX_INTERNAL, "A" * 18])
def test_remember_rejects_room_scoped_ids_bare(bare_provider, room_id):
    """Boundary: single-store (no manifest) deployments minted the raw bucket
    straight through — the guard refuses the shape instead."""
    out = bare_provider.remember(DURABLE, user_id=room_id)
    _assert_rejected(bare_provider, out)
    assert bare_provider._storage.count(room_id) == 0


def test_remember_rejects_room_scoped_via_raw_session_fallback(provider):
    """Replay of the 2026-11-04 orphan-mint path: a registered-later platform
    id arrives unresolvable, the L3 fallback keeps the raw session id, and
    the guard refuses it before the bucket exists."""
    provider._current_user_id = ROOM_MM_ID
    out = provider.remember(DURABLE, user_id=ROOM_MM_ID)
    _assert_rejected(provider, out)
    assert provider._storage.count(ROOM_MM_ID) == 0


def test_boundary_is_18_chars(bare_provider):
    """17-char opaque ids stay outside the net (manifest re-attributes them);
    18 is the room-id net edge."""
    out = bare_provider.remember(DURABLE, user_id="a" * 17)
    assert out.get("status") == "stored", out
    out = bare_provider.remember(DURABLE, user_id="a" * 18)
    _assert_rejected(bare_provider, out)


def test_mxid_and_non_room_ids_pass(provider):
    """The guard's net is the room shape, not 'any unusual id': MXIDs and
    short opaque ids keep flowing to manifest re-attribution."""
    out = provider.remember(DURABLE, user_id="@grace:unredacted.org")
    assert out.get("status") == "stored", out
    out = provider.remember(DURABLE, user_id="6")
    assert out.get("status") == "stored", out  # 'general' fallback owns it


def test_store_boundary_rejects_direct_writes(bare_provider):
    """Backstop for non-provider writers: store()/store_split() raise before
    any row lands."""
    emb = bare_provider._embedder.embed(DURABLE)
    with pytest.raises(ValueError, match="room-shaped"):
        bare_provider._storage.store(DURABLE, ROOM_MM_ID, emb, {})
    with pytest.raises(ValueError, match="room-shaped"):
        bare_provider._storage.store_split(
            "parent", ROOM_MM_ID,
            [{"content": "child", "embedding": emb, "trust_score": 0.5}])
    assert bare_provider._storage.count(ROOM_MM_ID) == 0


def test_no_reachable_write_path_emits_room_ids(provider):
    """Sweep every reachable write entry point with the room-shaped id in the
    caller position; afterwards the store holds ZERO room-shaped buckets.
    This is the card's 'no reachable in-repo path' gate."""
    provider._current_user_id = ROOM_MM_ID
    out = provider.remember(DURABLE, user_id="grace")  # provider is live
    assert out.get("status") == "stored", out
    provider.on_memory_write("add", "", DURABLE, None)
    provider.on_memory_write("add", "", DURABLE, {"user_id": ROOM_MM_ID})
    provider.on_pre_compress([{"role": "user", "content": DURABLE + " — long enough"}])
    out = provider.handle_tool_call("remember", {"content": DURABLE})
    assert "rejected_room_scoped" in out, out
    out = provider.remember(DURABLE, user_id=None)  # session uid = room id
    _assert_rejected(provider, out)
    out = provider.seed(ROOM_MM_ID, DURABLE)
    _assert_rejected(provider, out)

    rows = provider._storage._db.execute(
        "SELECT DISTINCT user_id FROM facts").fetchall()
    uids = [r[0] for r in rows]
    assert uids == ["grace"], uids  # only the scope owner; nothing room-shaped
    for uid in uids:
        assert not room_shaped_user_id(uid), uids
