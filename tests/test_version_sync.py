"""Version sync guard — plugin.yaml is the single source of truth.

Guards against the 0.6.3/0.6.4 drift: README/SPEC banners left at 0.6.2
and the SPEC's plugin.yaml sample stuck at 0.5.1 since 0.6.0.

Standard: X.Y.Z, no "v" prefix (applies to plugin.yaml, tags, banners).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_RE = re.compile(r"^version:\s*(\S+)", re.M)


def test_version_sync():
    m = VERSION_RE.search((ROOT / "plugin.yaml").read_text())
    assert m, "plugin.yaml missing version line"
    version = m.group(1)
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"plugin.yaml version not X.Y.Z (no v prefix): {version}"
    )
    assert f"**Version {version}**" in (ROOT / "README.md").read_text(), (
        f"README banner stale (plugin.yaml is {version})"
    )
    spec = (ROOT / "SPEC.md").read_text()
    assert f"**Version {version}**" in spec, (
        f"SPEC banner stale (plugin.yaml is {version})"
    )
    samples = VERSION_RE.findall(spec)
    assert samples and all(s == version for s in samples), (
        f"SPEC plugin.yaml sample stale: {samples} (plugin.yaml is {version})"
    )