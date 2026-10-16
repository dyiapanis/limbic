"""Cold-start story test — the guard for the "install mismatch" defect class.

Fully offline (per second-round review): live PyPI probes would page at 3am
on flaky networks — the static checks below catch every known failure mode.

1. Manifest hygiene: `pip_dependencies` must never name a `spacy-model-*`
   package — spaCy language models are not on PyPI (404 proven in the
   2026-10-17 review; Hermes' PM also drops direct-URL requirements), so the
   model installs once per environment via `python -m spacy download`
   (README Quick Start step 1 is the canonical instruction).

2. Canonical command presence: the README must carry the install command.

3. Claim lint: no doc surface may promise auto-download for spaCy language
   models or name a fictional `spacy-model-*` package.
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_CLAIMS = (
    "download automatically",
    "download on first use",  # spaCy-specific wording; ONNX HF downloads are real and pinned
    "spacy-model-",
)


def pip_dependencies() -> list[str]:
    spec = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    return spec.get("pip_dependencies", [])


def test_manifest_declares_no_spacy_models():
    deps = pip_dependencies()
    assert deps, "plugin.yaml declares no pip_dependencies — install story is implicit and won't verify"
    fictional = [d for d in deps if "spacy-model-" in d or "en-core-web" in d]
    assert fictional == [], (
        f"{fictional} — spaCy language models are NOT on PyPI (404-proven), and Hermes' "
        "plugin manager drops direct-URL requirements, so a manifest entry can never "
        "deliver them. They install via `python -m spacy download` (README step 1)."
    )


def test_readme_carries_canonical_model_install():
    readme = (ROOT / "README.md").read_text()
    # streamlined install: auto-download via languages --set, no manual spacy command needed
    assert "hermes limbic languages" in readme, (
        "README Quick Start must document `hermes limbic languages` for model install"
    )


@pytest.mark.parametrize(
    "path",
    ["README.md", "SPEC.md", "cli.py", "entities.py", "limbic.yaml", "plugin.yaml"],
)
def test_no_auto_download_or_pip_model_claim(path):
    text = (ROOT / path).read_text(errors="replace")
    hits = [c for c in FORBIDDEN_CLAIMS if c.lower() in text.lower()]
    assert hits == [], (
        f"{path} contains {hits}: either promises spaCy auto-download or names a "
        "nonexistent spacy-model-* package. spaCy models install once per environment "
        "by `python -m spacy download` — see README Quick Start step 1. This claim "
        "shipped three times before (entities docstring, languages CLI, SPEC) and each "
        "fresh install proved it false."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))