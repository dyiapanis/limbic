#!/usr/bin/env python3
"""Atomic soft-split — structure detection (sprint C5, plan §C5).

A fact longer than ~700 chars is one vector to the embedder: every section
averages into it and recall lands in the weak 0.3–0.4 cosine band (audit
2026-11-20: 43–51% of facts >600 chars, 98 >1,100 on phoenix). When content
carries DETECTED structure — numbered/bulleted list items, markdown headers,
or multiple substantive paragraphs — remember() splits it into child facts.
A single narrative (one block of prose, no markers) stays whole: ambiguous
content returns None and the fact is stored undivided.

Pure text, no models. The detector is deliberately conservative: a signal
must yield ≥2 fragments each ≥ min_child chars before anything splits.
"""
from __future__ import annotations

import re

# List items / bullets at line start. Numbered ("1. ", "2)"), bullets
# ("- ", "* ", "• ").
_LIST_ITEM = re.compile(r"^\s*(?:\d+[\.\)]\s+|[-*•]\s+)\S")
_MD_HEADER = re.compile(r"^#{1,6}\s+\S")


def split_structured(content: str, min_total: int = 700,
                     min_child: int = 120) -> list[str] | None:
    """Split structured content into child fragments, or None.

    Order of signals (conservative first): list items → markdown headers →
    paragraph clusters. Returns ≥2 fragments (each ≥ min_child after small
    fragments merge into their neighbour) or None.
    """
    text = content.strip()
    if len(text) <= min_total:
        return None

    for splitter in (_split_list_items, _split_headers, _split_paragraphs):
        frags = splitter(text)
        if frags and len(frags) >= 2:
            merged = _merge_small(frags, min_child)
            if len(merged) >= 2:
                return merged
    return None


def _split_list_items(text: str) -> list[str]:
    """One fragment per list item; indented/continuation lines join their
    item; leading (unmarked) lines join the first item's fragment."""
    frags: list[str] = []
    cur: list[str] = []
    for line in text.split("\n"):
        if _LIST_ITEM.match(line) and cur:
            frags.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        frags.append("\n".join(cur))
    return [f for f in frags if f.strip()]


def _split_headers(text: str) -> list[str]:
    """One fragment per markdown-header section (header line + body);
    text before the first header is its own fragment."""
    frags: list[str] = []
    cur: list[str] = []
    for line in text.split("\n"):
        if _MD_HEADER.match(line) and cur:
            frags.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        frags.append("\n".join(cur))
    return [f for f in frags if f.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """Paragraph clusters: fragments on blank-line breaks. A single
    narrative (no blank lines) yields one fragment → caller keeps it whole."""
    return [f for f in (p.strip() for p in re.split(r"\n\s*\n", text)) if f]


def _merge_small(frags: list[str], min_child: int) -> list[str]:
    """Fold fragments shorter than min_child into their accumulating
    neighbour (leading small pieces ride until one clears the floor)."""
    out: list[str] = []
    buf = ""
    for f in frags:
        buf = f"{buf}\n\n{f}" if buf else f
        if len(buf) >= min_child:
            out.append(buf)
            buf = ""
    if buf:  # trailing piece below the floor joins the last child
        out[-1] = f"{out[-1]}\n\n{buf}" if out else buf
    return out
