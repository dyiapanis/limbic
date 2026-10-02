"""Limbic identity — user manifest resolution from users.yaml.

users.yaml is a Limbic CONFIG FILE: user data that resides in Limbic.
Default slot is the plugin directory itself (plugins/limbic/users.yaml —
fleet-shared, gitignored; users.yaml.example ships in its place). The
limbic.yaml ``users_manifest`` key overrides the path, and a
$HERMES_HOME/users.yaml copy takes precedence per-profile (same
override pattern as limbic.yaml itself).

Limbic needs only the identity subset (canonical name, platform IDs,
scope, guardians); ``roles`` and any unknown keys are ignored, so the
file can also be shared with other infrastructure.

Modes:
  manifest present → multi-user mode (scopes + guardian semantics active)
  manifest absent  → single-user mode (raw platform IDs become buckets;
                     a loud init log says so; a second distinct user ID
                     triggers a one-time warning)

Path resolution order (first existing wins):
  1. limbic.yaml ``users_manifest`` (explicit override)
  2. ``$HERMES_HOME/users.yaml`` (per-profile override, like limbic.yaml)
  3. ``<plugin dir>/users.yaml`` (default — the file lives in Limbic)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Platforms compared case-insensitively (email addresses); others exact.
_CASE_INSENSITIVE_PLATFORMS = {"email"}


@dataclass
class Identity:
    canonical: str
    display_name: str
    scope: str = "adult"
    guardians: list = field(default_factory=list)
    guardian_of: list = field(default_factory=list)


class UserManifest:
    """In-memory view of users.yaml (identity subset only)."""

    def __init__(self, path: str, users: dict):
        self.path = path
        self._by_platform: dict[tuple[str, str], str] = {}
        self._users: dict[str, dict] = {}
        self._guardian_of: dict[str, list[str]] = {}

        for canonical, cfg in users.items():
            canon = canonical.lower().strip()
            self._users[canon] = cfg
            platform_ids = cfg.get("platform_ids", {}) or {}
            for platform, pid in platform_ids.items():
                p = str(platform).lower().strip()
                v = str(pid).strip()
                if p in _CASE_INSENSITIVE_PLATFORMS:
                    v = v.lower()
                self._by_platform[(p, v)] = canon
            # Reverse guardian map: bob.guardians=[alice] → alice.guardian_of=[bob]
            for g in cfg.get("guardians", []) or []:
                self._guardian_of.setdefault(str(g).lower().strip(), []).append(canon)

    @property
    def user_count(self) -> int:
        return len(self._users)

    @property
    def canonical_names(self) -> list[str]:
        return list(self._users.keys())

    def resolve_by_canonical(self, canonical: str) -> Optional[Identity]:
        """Identity for a canonical name (guardian lookup), or None."""
        cfg = self._users.get(str(canonical).lower().strip())
        if not cfg:
            return None
        return Identity(
            canonical=str(canonical).lower().strip(),
            display_name=cfg.get("display_name", canonical),
            scope=cfg.get("scope", "adult"),
            guardians=[str(g).lower() for g in cfg.get("guardians", []) or []],
            guardian_of=self._guardian_of.get(str(canonical).lower().strip(), []),
        )

    def resolve(self, platform_user_id: str, platform: str = "") -> Optional[Identity]:
        """Resolve a platform user ID to an Identity, or None if unknown."""
        if not platform_user_id:
            return None
        p = str(platform or "").lower().strip()
        v = str(platform_user_id).strip()
        if p in _CASE_INSENSITIVE_PLATFORMS:
            v = v.lower()
        canonical = self._by_platform.get((p, v))
        if not canonical and not p:
            # Platform not stated: match only unambiguous homespaced Matrix IDs.
            # A bare value match across all platforms would let anyone on an
            # unregistered platform quote a registered ID (email, phone) and
            # resolve to that user's identity and guardian map.
            if v.startswith("@") and ":" in v:
                canonical = self._by_platform.get(("matrix", v))
        if not canonical:
            return None
        cfg = self._users[canonical]
        return Identity(
            canonical=canonical,
            display_name=cfg.get("display_name", canonical),
            scope=cfg.get("scope", "adult"),
            guardians=[str(g).lower() for g in cfg.get("guardians", []) or []],
            guardian_of=self._guardian_of.get(canonical, []),
        )


def load_manifest(config: dict, hermes_home: str = "") -> Optional[UserManifest]:
    """Find and load users.yaml; None (single-user mode) if not found.

    Plugins are self-contained: the default slot is this plugin's own
    directory; per-profile and explicit config overrides layer on top.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    cfg_path = (config or {}).get("users_manifest", "")
    if cfg_path:
        candidates.append(os.path.expanduser(str(cfg_path)))
    if hermes_home:
        candidates.append(os.path.join(hermes_home, "users.yaml"))
    candidates.append(os.path.join(here, "users.yaml"))  # default: in Limbic

    for path in candidates:
        if os.path.isfile(path):
            try:
                import yaml
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                users = data.get("users", {})
                if not users:
                    logger.warning("limbic: users.yaml at %s has no users: block — ignoring", path)
                    return None
                logger.info("limbic: user manifest loaded (%s) — multi-user mode, %d users",
                            path, len(users))
                return UserManifest(path, users)
            except Exception as e:
                logger.warning("limbic: failed to load user manifest %s: %s", path, e)
                return None
    logger.info(
        "limbic: no users.yaml found — SINGLE-USER MODE; scopes and guardian "
        "semantics inactive (all memories attributed to the raw user ID)")
    return None


def _demo() -> None:
    """Self-check: manifest resolution + guardian reverse map."""
    import tempfile
    yaml_text = """
users:
  alice:
    display_name: Alice
    scope: adult
    platform_ids: {matrix: '@alice:hs.org', email: Alice@Example.com}
  bob:
    display_name: Bob
    scope: child
    guardians: [alice]
    platform_ids: {matrix: '@bob:hs.org'}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    import yaml as _y
    m = UserManifest(path, _y.safe_load(_y_text_of(path))["users"])
    a = m.resolve("@alice:hs.org", "matrix")
    assert a and a.canonical == "alice" and a.scope == "adult", a
    b = m.resolve("@bob:hs.org", "matrix")
    assert b and b.canonical == "bob" and b.scope == "child", b
    assert b.guardians == ["alice"], b.guardians
    assert a.guardian_of == ["bob"], a.guardian_of
    e = m.resolve("alice@example.com", "email")
    assert e and e.canonical == "alice", e  # case-insensitive email
    assert m.resolve("@nobody:hs.org", "matrix") is None
    os.unlink(path)
    print("identity demo ok")


def _y_text_of(path: str) -> str:
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    _demo()