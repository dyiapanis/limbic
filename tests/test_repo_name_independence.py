"""Regression pin (repo-name independence): the repo-root ``__init__.py`` must
tolerate being imported as a NON-package module — pytest's package walk does
exactly that when the checkout/tarball directory name isn't a valid package
identifier (GitHub tarballs: ``owner-repo-sha``), and a crash there killed the
whole suite at setup (101 collection errors on 0.6.1)."""
import importlib.util
import os
import sys


def test_root_init_tolerates_nonpackage_import(tmp_path, monkeypatch):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # simulate the failing import: load __init__.py under a hyphenated,
    # un-importable-as-package name, EXACTLY as pytest's walk does (no package)
    spec = importlib.util.spec_from_file_location(
        "dyiapanis-limbic-something", os.path.join(repo_root, "__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must NOT raise ImportError
    assert mod is not None
    # the skipped-import path must NOT fake re-exports on the bogus module
    assert not hasattr(mod, "LimbicMemoryProvider")  # re-exports belong to `limbic` proper


def test_root_init_still_reexports_as_package():
    # the real package path (`import limbic`) keeps the re-exports
    import limbic
    assert hasattr(limbic, "LimbicMemoryProvider")
    assert hasattr(limbic, "REMEMBER_SCHEMA")
