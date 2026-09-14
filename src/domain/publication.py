from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PublicationStatus(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    PUBLISHED = "published"
    FAILED = "failed"


class Publication(BaseModel):
    """Сгенерированный материал и история использования знаний."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)

    # Тезисы, на которых основан материал.
    thesis_ids: list[str] = Field(..., min_length=1)

    # Конкретные фрагменты источников, переданные генератору.
    semantic_unit_ids: list[str] = Field(default_factory=list)

    title: str | None = None
    text: str = Field(..., min_length=1)

    status: PublicationStatus = PublicationStatus.DRAFT

    # Например: "vk", "telegram", "website".
    platform: str | None = None

    # ID публикации во внешней системе после успешной отправки.
    external_id: str | None = None

    published_at: datetime | None = None

    # Полезно хранить параметры генерации:
    # модель, prompt_version, temperature и т.п.
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
