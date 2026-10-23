"""Limbic tools — the agent interface.

Three tools. The agent feeds experiences, queries memory, and removes a
memory when its owner explicitly asks. Everything else is internal.

  remember(content) → stored
  recall(query) → results
  forget(query | fact_id) → candidates → removed (owner-initiated only)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Trust boundary: cap tool input size. Facts are 1-3 sentences; 4000 chars is
# ~10x headroom. Larger inputs are rejected before embedding/storage.
MAX_CONTENT = 4000

REMEMBER_SCHEMA = {
    "name": "remember",
    "description": (
        "Store a fact the user would expect you to remember. "
        "Use proactively — don't wait to be asked. Store preferences, "
        "decisions, personal details, corrections, and context worth "
        "recalling in future conversations."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to store.",
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "recall",
    "description": (
        "Search your memory for relevant facts. "
        "Use before answering questions that depend on past context, "
        "user preferences, or conversation history."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for.",
            },
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "forget",
    "description": (
        "Remove a specific memory at the user's explicit request — the "
        "owner-initiated removal path (the only manual memory control). "
        "Two-step: FIRST call with query to list candidate facts (id, "
        "excerpt, created, trust); show them to the user; THEN call with "
        "the chosen fact_id to delete. Execute ONLY when the user "
        "explicitly asks to delete or remove a memory — never on your "
        "own initiative, never to 'clean up'. A hash-only receipt is "
        "kept; the content itself is not retained after deletion."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "What to search for — returns candidate facts for the "
                    "user's review. Nothing is deleted in this step."
                ),
            },
            "fact_id": {
                "type": "integer",
                "description": (
                    "The fact id to delete, chosen from a previous "
                    "candidate listing shown to the user."
                ),
            },
        },
        # Exactly one of the two — schema-level statement of the dispatch
        # rule (tools.py enforces the same exactly-one-of at runtime).
        "oneOf": [
            {"required": ["query"]},
            {"required": ["fact_id"]},
        ],
    },
}

def handle_tool_call(provider, tool_name: str, args: dict, session_id: str = "") -> str:
    """Dispatch tool calls to the provider."""
    try:
        if tool_name == "remember":
            content = args.get("content", "")
            if not isinstance(content, str):
                return json.dumps({"error": "content must be a string"})
            content = content.strip()
            if not content:
                return json.dumps({"error": "Missing required parameter: content"})
            if len(content) > MAX_CONTENT:
                return json.dumps({"error": f"content too long (max {MAX_CONTENT} chars)"})
            result = provider.remember(content, session_id=session_id)
            return json.dumps(result)

        elif tool_name == "recall":
            query = args.get("query", "")
            if not isinstance(query, str):
                return json.dumps({"error": "query must be a string"})
            query = query.strip()
            if not query:
                return json.dumps({"error": "Missing required parameter: query"})
            if len(query) > MAX_CONTENT:
                return json.dumps({"error": f"query too long (max {MAX_CONTENT} chars)"})
            result = provider.recall(query, session_id=session_id)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "forget":
            query = args.get("query", "")
            fact_id = args.get("fact_id")
            if isinstance(query, str):
                query = query.strip()
            if fact_id is not None and not isinstance(fact_id, int):
                if isinstance(fact_id, str) and fact_id.strip().isdigit():
                    fact_id = int(fact_id)
                else:
                    return json.dumps({"error": "fact_id must be an integer"})
            if bool(query) == bool(fact_id is not None):
                return json.dumps({"error": "pass exactly one of query / fact_id"})
            if query and len(query) > MAX_CONTENT:
                return json.dumps({"error": f"query too long (max {MAX_CONTENT} chars)"})
            result = provider.forget_memory(query=query or None, fact_id=fact_id,
                                            session_id=session_id)
            return json.dumps(result, ensure_ascii=False)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        # Trust boundary: the agent must never see a raw traceback or
        # internal paths — a type name plus a capped, generic message.
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "…"
        return json.dumps({"error": f"{type(e).__name__}: {msg}"})