"""Pre-publish scan: personal / fleet markers must never enter the tracked tree.

Backstop for the eval-fixture harvest that shipped personal content to the
public repo (cohort shipped 2026-10-23, purged 2026-11-09). Scans every tracked TEXT
file for the standing marker list; any hit fails the suite before a push.

# ponytail: targeted token list, whole-tree scan — commit MESSAGES are not
# scanned; a fleet persona appearing as a natural word would need a full NER
# pass. Extend MARKERS when a new surface (device, project, platform) ships.
"""
import base64
import os
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
reporoot = os.path.dirname(here)

_MARKERS_B64 = [
    "ZGVtZXRyaXNAeWlhcGFuaXMuY28=",
    "cGhvZW5peDE4MTk=",
    "TklDTyBCSUJMRQ==",
    "R2Vla29t",
    "UG9sYXIgRmxvdw==",
    "SlVNUEVSIExBVw==",
    "S2lybmk=",
]
# obfuscated so the scan's own detection data never sits in the tree as plaintext
MARKERS = [base64.b64decode(m).decode() for m in _MARKERS_B64]


def _packaged_files():
    """No-git fallback (GitHub tarball/zip download): everything the archive
    carries is published content — scan the tree, minus build/vendored noise."""
    skip_dirs = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    for dirpath, dirnames, filenames in os.walk(reporoot):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            yield os.path.relpath(os.path.join(dirpath, name), reporoot).replace(os.sep, "/")


def tracked_text_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=reporoot, capture_output=True,
                             text=True, check=True).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        out = list(_packaged_files())  # no working git — tarball is all tracked content anyway
    for path in out:
        if path.startswith("models/"):
            continue  # vendored spaCy package, not authored here
        if path == "tests/test_no_personal_markers.py":
            continue  # own file: the obfuscated list is this file's whole content
        raw = open(os.path.join(reporoot, path), "rb").read(4_000_000)
        if b"\x00" in raw:
            continue  # binary heuristic — lexicons, ONNX, images
        yield path, raw.decode("utf-8", errors="replace")


def test_no_personal_markers():
    hits = []
    for path, text in tracked_text_files():
        for marker in MARKERS:
            if marker.lower() in text.lower():
                hits.append((path, marker))
    assert not hits, f"personal/fleet markers in tracked tree: {hits}"