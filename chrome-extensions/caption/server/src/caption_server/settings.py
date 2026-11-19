from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .extension_identity import unpacked_extension_origin

DEFAULT_KEEP_TERMS = (
    "transformer",
    "token",
    "attention",
    "self-attention",
    "encoder",
    "decoder",
    "embedding",
    "API",
    "GPU",
    "LLM",
)


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAPTION_", env_file=".env", extra="ignore")

    data_dir: Path = Path.home() / ".caption-server"
    host: str = "127.0.0.1"
    port: int = 8765
    token: str = ""
    allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    dev_extension_dir: Path | None = None
    asr_model: str = "JacobLinCool/TEA-ASR-1.1-mini"
    asr_language: str = "Chinese"
    asr_device: str = "mps"
    translate_model: str = "mlx-community/HY-MT1.5-1.8B-8bit"
    translate_max_tokens: int = 256
    keep_terms: Annotated[list[str], NoDecode] = Field(default_factory=lambda: list(DEFAULT_KEEP_TERMS))
    bilingual_english: bool = True
    max_audio_bytes: int = 2_000_000
    max_srt_bytes: int = 5_000_000
    max_translation_lines: int = 200

    @field_validator("allowed_origins", "keep_terms", mode="before")
    @classmethod
    def _parse_csv(cls, value: object) -> object:
        return _split_csv(value)

    @property
    def subtitles_dir(self) -> Path:
        return self.data_dir / "subtitles"

    @property
    def token_file(self) -> Path:
        return self.data_dir / "token"

    @property
    def audit_log(self) -> Path:
        return self.data_dir / "audit.jsonl"

    @property
    def dev_extension_origin(self) -> str | None:
        return unpacked_extension_origin(self.dev_extension_dir) if self.dev_extension_dir else None

    @property
    def effective_allowed_origins(self) -> list[str]:
        dev_origin = self.dev_extension_origin
        return [*self.allowed_origins, dev_origin] if dev_origin else list(self.allowed_origins)

    @property
    def allowed_hosts(self) -> set[str]:
        loopback = {"127.0.0.1", "localhost", "[::1]"}
        names = loopback | {self.host}
        return names | {f"{name}:{self.port}" for name in names}
