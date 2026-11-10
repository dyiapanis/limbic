"""Pre-publish scan: personal / fleet markers must never enter the tracked tree.

Backstop for the eval-fixture harvest that shipped personal content to the
public repo (cohort shipped 2026-10-23, purged 2026-11-09). Scans every tracked TEXT
file for the standing marker list; any hit fails the suite before a push.

# ponytail: targeted token list, whole-tree scan — commit MESSAGES are not
# scanned; a fleet persona appearing as a natural word would need a full NER
# pass. Extend MARKERS when a new surface (device, project, platform) ships.
"""
import os
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
reporoot = os.path.dirname(here)

MARKERS = [
    "demetris@yiapanis.co",
    "phoenix1819",
    "NICO BIBLE",
    "Geekom",
    "Polar Flow",
    "JUMPER LAW",
    "Kirni",
]


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=reporoot, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    for path in out:
        if path.startswith("models/"):
            continue  # vendored spaCy package, not authored here
        if path == "tests/test_no_personal_markers.py":
            continue  # this scan's own definition — its MARKERS list would self-match
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
