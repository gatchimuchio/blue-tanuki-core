"""Redaction for audit payloads. Always run before hashing & writing."""
from __future__ import annotations

import re
from typing import Any

REDACTED_VALUE = "***"
REDACTED_MATCH = "***REDACTED***"


class Redactor:
    def __init__(
        self,
        secret_keys: set[str] | None = None,
        patterns: list[str] | None = None,
    ):
        self.secret_keys = {k.lower() for k in (secret_keys or set())}
        self.patterns = [re.compile(p) for p in (patterns or [])]

    def redact(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: (REDACTED_VALUE if k.lower() in self.secret_keys else self.redact(v))
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self.redact(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact(x) for x in obj)
        if isinstance(obj, str):
            out = obj
            for p in self.patterns:
                out = p.sub(REDACTED_MATCH, out)
            return out
        return obj


__all__ = ["Redactor", "REDACTED_VALUE", "REDACTED_MATCH"]
