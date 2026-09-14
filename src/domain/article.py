from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Article(BaseModel):
    """Исходная научная или научно-популярная публикация."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)

    title: str = Field(..., min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1800, le=2200)

    # Журнал, сборник, сайт, издательство и т.п.
    source: str | None = None
    url: str | None = None
    doi: str | None = None

    language: str = Field(default="ru", min_length=2)

    # Полный очищенный текст источника.
    full_text: str = Field(..., min_length=1)

    # Любые дополнительные данные, которые не хочется фиксировать в схеме:
    # том, выпуск, страницы, ISBN, тип документа и т.д.
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
