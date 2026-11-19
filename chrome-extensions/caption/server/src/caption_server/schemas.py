from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from .overwrite_policy import SubtitleProgress, SubtitleStatus

URL_KEY_PATTERN = r"^[0-9a-f]{64}$"
SCHEMA_VERSION = 1


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SubtitleMetaInput(CamelModel):
    canonical_url: str = Field(min_length=1, max_length=4096)
    status: SubtitleStatus
    covered_seconds: float = Field(ge=0)
    covered_intervals: list[tuple[float, float]] = Field(default_factory=list, max_length=10_000)
    media_duration: float | None = Field(default=None, ge=0)
    source: Literal["asr", "youtube-captions"] = "asr"
    language: str | None = Field(default=None, max_length=32)
    session_id: str = Field(min_length=1, max_length=128)
    last_seq: int = Field(ge=0)

    @model_validator(mode="after")
    def _intervals_are_ordered(self) -> "SubtitleMetaInput":
        if any(end < start for start, end in self.covered_intervals):
            raise ValueError("coveredIntervals entries must satisfy start <= end")
        return self

    def progress(self) -> SubtitleProgress:
        return SubtitleProgress(self.status, self.covered_seconds, self.session_id, self.last_seq)


class SubtitleMeta(SubtitleMetaInput):
    schema_version: int = SCHEMA_VERSION
    url_key: str
    revision: int
    engine: dict[str, str] = Field(default_factory=dict)
    updated_at: str


class PutSubtitleRequest(CamelModel):
    meta: SubtitleMetaInput
    srt: str
    expected_revision: int | None = Field(default=None, ge=0)


class PutSubtitleResponse(CamelModel):
    written: bool
    revision: int
    reason: str | None = None


class SubtitleDocument(CamelModel):
    meta: SubtitleMeta
    srt: str


class TranscriptionResponse(CamelModel):
    text: str
    language: str
    translation: str | None
    display: str
    engine: dict[str, str]


class TranslationRequest(CamelModel):
    texts: list[str] = Field(min_length=1)


class TranslationResponse(CamelModel):
    translations: list[str]
    displays: list[str]
