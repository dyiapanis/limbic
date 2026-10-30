"""Tool schemas must survive strict providers (xAI rejects root unions with untyped branches).

xAI returns 400 invalid_client_tool_schema: "tool parameter root must be an object
type (root schema is an anyOf/oneOf union with a non-object branch)" for any tool
whose root carries a union branch lacking an explicit type. One such schema makes
EVERY model call fail, so the whole tool list is poisoned, not just that tool.
"""

import tools

SCHEMAS = {
    name: getattr(tools, name)
    for name in dir(tools)
    if name.endswith("_SCHEMA") and isinstance(getattr(tools, name), dict)
}


def test_every_tool_schema_has_schemas():
    assert SCHEMAS, "no *_SCHEMA constants found"


def test_root_is_object_typed():
    for name, schema in SCHEMAS.items():
        params = schema["parameters"]
        assert params.get("type") == "object", f"{name}: root must be type object"


def test_union_branches_are_object_typed():
    for name, schema in SCHEMAS.items():
        params = schema["parameters"]
        for keyword in ("oneOf", "anyOf"):
            for branch in params.get(keyword, []):
                assert branch.get("type") == "object", (
                    f"{name}: {keyword} branch {branch} lacks type=object — "
                    "strict providers (xAI) reject the whole request"
                )
