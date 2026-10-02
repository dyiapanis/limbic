#!/usr/bin/env python3
"""Fail if any symbol named in docs/anatomy-map.md no longer exists.

Lazy check: no parsing of the table structure, just extract dotted symbols from
the Code column and confirm each is findable in the plugin source. Ceiling:
proves the symbol exists, not that it still does what the row claims.

    python3 docs/verify_anatomy_map.py            # default plugin path
    python3 docs/verify_anatomy_map.py /path/to/limbic
"""
import re
import sys
from pathlib import Path

DEFAULT = Path.home() / ".hermes/plugins/limbic"
MAP = Path(__file__).with_name("anatomy-map.md")


def default_plugin() -> Path:
    """Prefer the code beside this script (repo self-check), else the deployment."""
    sibling = Path(__file__).resolve().parent.parent
    return sibling if (sibling / "provider.py").exists() else DEFAULT


def symbols(text: str) -> set[str]:
    """Dotted symbols from backticked spans in the table's Code column.

    Filenames (`store_sqlite.py`) are skipped — a `.py` leaf is a file, not a
    symbol.
    """
    out = set()
    for span in re.findall(r"`([^`]+)`", text):
        for tok in re.findall(r"\b(?:provider|store_sqlite|trust|entities|nli|embeddings|storage|identity)\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", span):
            if not tok.endswith(".py"):
                out.add(tok)
    return out


def main() -> int:
    plugin = Path(sys.argv[1]) if len(sys.argv) > 1 else default_plugin()
    text = MAP.read_text(encoding="utf-8")
    src = {p.stem: p.read_text(encoding="utf-8") for p in plugin.glob("*.py")}

    missing = []
    for sym in sorted(symbols(text)):
        mod, *rest = sym.split(".")
        # `mod.name` — check name appears anywhere in that module (methods too).
        needle = rest[-1]
        body = src.get(mod)
        if body is None:
            missing.append(f"{sym}  (no module {mod}.py)")
        elif needle not in body:
            missing.append(f"{sym}  ({needle} not found in {mod}.py)")

    if missing:
        print(f"FAIL — {len(missing)} mapped symbol(s) gone:")
        for m in missing:
            print("  " + m)
        return 1

    print(f"OK — {len(symbols(text))} mapped symbols present in {plugin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
