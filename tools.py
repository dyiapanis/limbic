"""Limbic tools — the agent interface.

Two tools. The agent feeds experiences and queries memory.
Everything else is internal.

  remember(content) → stored
  recall(query) → results
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

        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        # Trust boundary: the agent must never see a raw traceback or
        # internal paths — a type name plus a capped, generic message.
        msg = str(e)
        if len(msg) > 200:
            msg = msg[:200] + "…"
        return json.dumps({"error": f"{type(e).__name__}: {msg}"})