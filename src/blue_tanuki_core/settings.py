"""Settings with pydantic-settings. SecretRef resolves fail-closed."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError


class SecretRef(BaseModel):
    source: Literal["env", "file", "literal"]
    name: str | None = None
    path: Path | None = None
    value: str | None = None

    def resolve(self) -> str:
        if self.source == "env":
            if not self.name:
                raise ConfigError("SecretRef.env", "name is required")
            v = os.environ.get(self.name)
            if v is None or v == "":
                raise ConfigError("SecretRef.env", f"env var {self.name} unresolved")
            return v
        if self.source == "file":
            if not self.path:
                raise ConfigError("SecretRef.file", "path is required")
            if not self.path.exists():
                raise ConfigError("SecretRef.file", f"{self.path} does not exist")
            return self.path.read_text().strip()
        if self.source == "literal":
            if self.value is None:
                raise ConfigError("SecretRef.literal", "value is required")
            return self.value
        raise ConfigError("SecretRef.source", f"unknown: {self.source}")


class LLMSettings(BaseModel):
    default_model: str = "claude-sonnet-4-20250514"
    audit_model: str = "claude-sonnet-4-20250514"
    audit_seed: int = 1
    timeout_s: float = 60
    max_input_tokens: int = 150_000
    max_usd_per_turn: float | None = 1.0
    max_tokens_per_turn: int | None = 30_000
    usd_per_1k_prompt: float = 0.0
    usd_per_1k_completion: float = 0.0
    anthropic_key: SecretRef | None = None
    openai_key: SecretRef | None = None


class AuditSettings(BaseModel):
    rotate_bytes: int = 100 * 1024 * 1024
    redact_keys: list[str] = Field(default_factory=lambda: [
        "anthropic_key", "openai_key", "api_key",
        "authorization", "token", "password", "secret",
    ])
    redact_patterns: list[str] = Field(default_factory=list)


class RuntimeSettings(BaseModel):
    grace_shutdown_s: float = 30
    pending_ttl_s: int = 3600
    module_timeout_s: float = 30


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BLUE_TANUKI_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )
    state_dir: Path = Field(default_factory=lambda: Path("~/.blue-tanuki").expanduser())
    policy_path: Path | None = None
    llm: LLMSettings = LLMSettings()
    audit: AuditSettings = AuditSettings()
    runtime: RuntimeSettings = RuntimeSettings()

    def ensure_dirs(self) -> None:
        (self.state_dir / "audit" / "sessions").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "config").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "tmp").mkdir(parents=True, exist_ok=True)


__all__ = [
    "Settings", "SecretRef",
    "LLMSettings", "AuditSettings", "RuntimeSettings",
]
