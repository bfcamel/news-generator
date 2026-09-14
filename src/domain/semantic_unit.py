from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SemanticUnit(BaseModel):
    """Законченный по смыслу фрагмент исходной статьи."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)

    # Ссылка на Article.id.
    article_id: str = Field(..., min_length=1)

    # Текст смысловой единицы.
    text: str = Field(..., min_length=1)

    # Порядковый номер внутри статьи.
    position: int = Field(..., ge=0)

    # Необязательная информация о месте фрагмента в статье.
    section_title: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)

    # Вектор документа для семантического поиска.
    doc_embedding: list[float] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_char_range(self) -> "SemanticUnit":
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be greater than or equal to char_start")
        return self
