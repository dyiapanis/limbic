"""Fragment-gate anti-fact legs (flat band, verbatim question, quoted speech)
with the durable-marker escape — golden set built from real store offenders.

Reject fixtures pin the audit offender class; survive fixtures pin zero
wrongly-rejected durable facts on the reviewed sample (durable intents
phrased as questions SURVIVE — watch-intents; dated micro-facts; principal
micro-facts; provenance-prefixed quoted directives).

Store provenance (anonymized): offender rows sit in two six-store-backup
audit copies — a 27/29-char utterance with the highest injection count in
its store (written 2026-08-11, pre-gate store-migration era), and a 38-char
question written 2026-09-03, five days before the gate reached the deployed
tree. Legacy rows were never re-appraised; post-deployment chat speech
rides the pair leg's words exemption (the reject set below is the proof).

Run:
    cd <repo-checkout> && python -m pytest tests/test_fragment_gate.py -v
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.provider import LimbicMemoryProvider  # noqa: E402
from limbic.store_sqlite import SQLiteStorage  # noqa: E402
from limbic.trust import TrustEngine  # noqa: E402
from limbic.identity import UserManifest  # noqa: E402

DIMS = 8

# Audit offenders (verbatim, card-pinned fixtures):
OFFENDER_LEAVE_IT = "Leave it. \n\nI want to use it."
OFFENDER_LULU = "Has Lulu dropped their latest results?"
OFFENDER_QUOTE_C = "it every day: you use it every day"

# Additional reviewed-sample rejects (real store rows):
REJECTS = [
    OFFENDER_LEAVE_IT,          # sub-threshold utterance (29 chars)
    OFFENDER_LULU,              # verbatim question, no durable marker (38)
    OFFENDER_QUOTE_C,        # quoted speech, no provenance (34)
    "test seeding debug",       # seed-era debug note (18)
    "New shortlist candidates",  # deictic label, not a fact (24)
    "PDF is delivered",         # process status line (16)
    "yes.. add audit lane",     # one-shot command ack (20)
    "Are you using dev branch?",  # verbatim chat question (26)
    "Did it exceed expectations?",  # verbatim chat question (27)
    "to go through and thoroughly test your tools: I want to go through and thoroughly test your tools",  # verbatim self-repeat (98)
    "Go ahead from your end",   # one-shot command (22)
    "proceed with 1, 2 and 3",  # one-shot command (23)
    '"leave me alone, then"',   # quote-leading, no provenance (22)
]

# Zero-wrongly-rejected golden set (reviewed): durable facts that MUST store.
SURVIVES = [
    "Grace is vegan",             # dense micro-fact, known principal (14)
    "Bob is active",              # dense micro-fact, known principal (13)
    "Week starting 23/11 looks light",  # dated training note (31)
    "Should we watch Severance tonight at 8?",  # watch-intent question (39)
    "Could we watch the F1 highlights then?",   # watch-intent question (40)
    "Grace does not want gym sessions pushed to her smartwatch as training targets",
    'Grace directive (19 Oct 2026): "title every section with one emoji"',
    '"call me before you ship that" — Grace',  # quote-leading WITH provenance
    "Grace wants the weekly digest to stay short",
]


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
    "mia": {"display_name": "Mia", "scope": "child",
            "platform_ids": {}, "guardians": ["grace"]},
    "noah": {"display_name": "Noah", "scope": "adult",
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


@pytest.mark.parametrize("content", REJECTS)
def test_anti_facts_rejected(provider, content):
    out = provider.remember(content, user_id="grace")
    assert out.get("status") == "rejected_fragment", (content, out)
    assert "fact_id" not in out
    assert provider._storage.count("grace") == 0, content


@pytest.mark.parametrize("content", SURVIVES)
def test_durable_facts_survive(provider, content):
    out = provider.remember(content, user_id="grace")
    assert "error" not in out, (content, out)
    assert "fact_id" in out, (content, out)
    assert out["status"] == "stored", (content, out)


def test_watch_intent_question_survives(provider):
    """Golden: durable intents phrased as questions SURVIVE."""
    out = provider.remember("Should we watch Severance tonight at 8?",
                            user_id="grace")
    assert "fact_id" in out, out


def test_host_frame_process_dumps_rejected_pre_split(provider):
    """Ephemeral process/delegation dumps are host framing — rejected before
    the splitter runs, so they never become parent+children facts."""
    dump = ("[IMPORTANT: Background process proc_deadbeef completed normally "
            "(exit code 0).\nCommand: cd /tmp && make test")
    out = provider.remember(dump, user_id="grace")
    assert out.get("status") == "rejected_host_frame", out
    delegation = "[ASYNC DELEGATION BATCH COMPLETE — deleg_0001]\nA background unit finished."
    out = provider.remember(delegation, user_id="grace")
    assert out.get("status") == "rejected_host_frame", out


def test_pair_gate_leg_unchanged(provider):
    """The pair leg (len<60 AND words<3) keeps its shape."""
    out = provider.remember("shut up", user_id="grace")
    assert out.get("status") == "rejected_fragment", out


def test_question_leg_preexisting(provider):
    """Regression: the bare-question offender is rejected by the leg that
    already existed before the echo leg — no new logic needed for it."""
    out = provider.remember("Did it exceed expectations?", user_id="grace")
    assert out.get("status") == "rejected_fragment", out
    assert "durable fact" in out["reason"], out


def test_seed_rejects_self_echo(provider):
    """seed() rides the same admission gate as remember()."""
    out = provider.seed("grace",
                        "to go through and thoroughly test your tools: "
                        "I want to go through and thoroughly test your tools")
    assert out.get("status") == "rejected_fragment", out
    assert provider._storage.count("grace") == 0


def test_echo_leg_shapes():
    """Pure helper: normalization, overlap, and no-false-positive edges."""
    f = LimbicMemoryProvider._is_self_echo
    assert f("The deployment team rehearses the failover drill.   the "
             "deployment team rehearses the failover drill")  # case+ws
    assert f("a a a a a a a")                                       # overlap
    assert f("to go through and thoroughly test your tools: "
             "I want to go through and thoroughly test your tools")
    assert not f("Grace is vegan")                                  # < ngram words
    assert not f("Grace does not want gym sessions pushed to her "
                 "smartwatch as training targets")                  # distinct


def test_flat_band_requires_manifest(bare_provider):
    """Without a manifest there is no reliable entity signal — the flat band
    is skipped and the pair gate stays the sole short-string rule."""
    out = bare_provider.remember(OFFENDER_LEAVE_IT, user_id="default")
    assert "fact_id" in out, out


def test_rejection_status_is_stable_for_log_review(provider):
    """All rejection legs share one status token for the rejection-log grep."""
    for content in REJECTS:
        out = provider.remember(content, user_id="grace")
        assert out.get("status") == "rejected_fragment", (content, out)
        assert isinstance(out.get("reason"), str) and out["reason"], content
