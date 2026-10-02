"""Identity resolver tests — users.yaml manifest, modes, guardian maps.

Model-free. Run:
    cd <plugins parent> && python -m pytest limbic/tests/test_identity.py -v
"""
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limbic.identity import UserManifest, load_manifest  # noqa: E402

YAML = """
users:
  alice:
    display_name: Alice
    scope: adult
    platform_ids: {matrix: '@alice:hs.org', email: Alice@Example.com}
  bob:
    display_name: Bob
    scope: child
    guardians: [alice]
    platform_ids: {matrix: '@bob:hs.org', simplex: '7'}
"""


@pytest.fixture()
def manifest(tmp_path):
    import yaml
    p = tmp_path / "users.yaml"
    p.write_text(YAML)
    return UserManifest(str(p), yaml.safe_load(YAML)["users"])


def test_resolve_matrix_id(manifest):
    ident = manifest.resolve("@alice:hs.org", "matrix")
    assert ident.canonical == "alice"
    assert ident.scope == "adult"
    assert ident.display_name == "Alice"


def test_resolve_case_insensitive_email(manifest):
    ident = manifest.resolve("ALICE@example.com", "email")
    assert ident is not None and ident.canonical == "alice"


def test_email_without_platform_stated(manifest):
    # platform="" → ONLY homespaced Matrix IDs may cross-match; a bare email
    # must NOT resolve (cross-platform value match was the identity leak)
    assert manifest.resolve("alice@example.com", "") is None


def test_matrix_id_without_platform_stated(manifest):
    # platform="" with a homespaced Matrix ID still resolves (unambiguous)
    ident = manifest.resolve("@alice:hs.org", "")
    assert ident is not None and ident.canonical == "alice"


def test_unknown_user_returns_none(manifest):
    assert manifest.resolve("@nobody:hs.org", "matrix") is None


def test_guardian_reverse_map(manifest):
    a = manifest.resolve("@alice:hs.org", "matrix")
    b = manifest.resolve("@bob:hs.org", "matrix")
    assert b.guardians == ["alice"]
    assert a.guardian_of == ["bob"]  # alice is guardian of bob


def test_child_scope_preserved(manifest):
    b = manifest.resolve("@bob:hs.org", "matrix")
    assert b.scope == "child"


def test_load_manifest_multi_user(tmp_path):
    import yaml
    p = tmp_path / "users.yaml"
    p.write_text(YAML)
    m = load_manifest({"users_manifest": str(p)})
    assert m is not None
    assert m.user_count == 2


def test_load_manifest_plugin_dir_default(tmp_path, monkeypatch):
    import yaml as _y
    # point hermes_home at an empty dir, plugin dir holds the manifest
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "users.yaml").write_text(YAML)
    import limbic.identity as ident
    monkeypatch.setattr(ident, "__file__", str(plugin_dir / "identity.py"))
    m = load_manifest({}, hermes_home=str(tmp_path / "home"))
    assert m is not None and m.user_count == 2


def test_load_manifest_hermes_home_overrides_plugin_dir(tmp_path, monkeypatch):
    import yaml as _y
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "users.yaml").write_text(YAML)  # 2 users
    home = tmp_path / "home"
    home.mkdir()
    (home / "users.yaml").write_text("users:\n  carol:\n    display_name: Carol\n")  # 1 user
    import limbic.identity as ident
    monkeypatch.setattr(ident, "__file__", str(plugin_dir / "identity.py"))
    m = load_manifest({}, hermes_home=str(home))
    assert m.user_count == 1  # HERMES_HOME wins over plugin default


def test_load_manifest_absent_is_single_user(tmp_path, caplog, monkeypatch):
    empty_dir = tmp_path / "plugin"
    empty_dir.mkdir()
    import limbic.identity as ident
    monkeypatch.setattr(ident, "__file__", str(empty_dir / "identity.py"))
    with caplog.at_level(logging.INFO, logger="limbic"):
        m = load_manifest({"users_manifest": str(tmp_path / "nope.yaml")},
                          hermes_home=str(tmp_path / "emptyhome"))
    assert m is None
    assert any("SINGLE-USER MODE" in r.getMessage() for r in caplog.records)


def test_load_manifest_ignores_roles(tmp_path):
    import yaml
    text = YAML + """
"""
    data = yaml.safe_load(text)
    data["users"]["alice"]["roles"] = {"phoenix": {"role": "admin", "toolsets": ["*"]}}
    p = tmp_path / "users.yaml"
    p.write_text(yaml.dump(data))
    m = load_manifest({"users_manifest": str(p)})
    assert m is not None
    ident = m.resolve("@alice:hs.org", "matrix")
    assert ident.canonical == "alice"
    assert not hasattr(ident, "roles")  # identity subset only


def test_provider_resolution_order(manifest, tmp_path, monkeypatch):
    # Provider: manifest used when no fallback applies; warning on unknown
    from limbic.provider import LimbicMemoryProvider
    from limbic.tests.test_provider_tools import FakeEmbedder, FakeNLI
    from limbic.store_sqlite import SQLiteStorage
    from limbic.trust import TrustEngine

    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=8)
    p._initialised = True
    p._manifest = manifest
    p._seen_user_ids = set()

    # manifest-first resolution (identity source is users.yaml)
    canonical, guardian_of, display, scope = p._resolve_identity("@bob:hs.org", {"platform": "matrix"})
    assert canonical == "bob"
    assert scope == "child"
    # guardian reverse map through provider path
    a = manifest.resolve_by_canonical("alice")
    assert a.guardian_of == ["bob"]
    p._storage.close()


def test_provider_multi_user_warning_no_manifest(tmp_path, caplog):
    from limbic.provider import LimbicMemoryProvider
    from limbic.tests.test_provider_tools import FakeEmbedder, FakeNLI
    from limbic.store_sqlite import SQLiteStorage
    from limbic.trust import TrustEngine

    p = LimbicMemoryProvider()
    p._load_config(str(tmp_path))
    p._embedder = FakeEmbedder()
    p._nli = FakeNLI()
    p._entity_extractor = None
    p._trust = TrustEngine(str(tmp_path / "limbic_params.json"))
    p._storage = SQLiteStorage(str(tmp_path / "limbic.db"), dims=8)
    p._initialised = True
    p._manifest = None
    p._seen_user_ids = set()

    with caplog.at_level(logging.WARNING, logger="limbic"):
        p._resolve_identity("@first:hs.org", {"platform": "matrix"})
        p._resolve_identity("@second:hs.org", {"platform": "matrix"})
    assert any("MULTIPLE user IDs" in r.getMessage() for r in caplog.records)
    p._storage.close()