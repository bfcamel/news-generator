from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AtomicThesis(BaseModel):
    """Один самостоятельный проверяемый факт из базы знаний."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)

    # Короткая атомарная формулировка одного факта.
    text: str = Field(..., min_length=1)

    # Один тезис должен быть подтверждён хотя бы одной смысловой единицей.
    semantic_unit_ids: list[str] = Field(..., min_length=1)

    # query_embedding — для поиска подтверждающего контекста.
    query_embedding: list[float] | None = None

    # doc_embedding можно хранить для сравнения тезисов друг с другом
    # и семантической дедупликации.
    doc_embedding: list[float] | None = None

    # Денормализованные счётчики для быстрого выбора тем.
    # Источником истины всё равно остаётся Publication.
    used_count: int = Field(default=0, ge=0)
    last_used_at: datetime | None = None

    # Позволяет временно исключить сомнительный или устаревший тезис.
    is_active: bool = True

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
