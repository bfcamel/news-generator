from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ThesisStatus(StrEnum):
    """Состояние атомарного тезиса."""

    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    DISABLED = "disabled"


class AtomicThesis(BaseModel):
    """
    Один самостоятельный проверяемый факт из базы знаний.

    Тезис должен выражать только одно утверждение и иметь
    хотя бы одну подтверждающую SemanticUnit.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    # Внутренний уникальный идентификатор тезиса.
    id: NonEmptyStr

    # Короткая атомарная формулировка одного факта.
    text: NonEmptyStr

    # SemanticUnit, которые подтверждают данный тезис.
    #
    # Один тезис может иметь несколько подтверждений
    # из одного или нескольких источников.
    semantic_unit_ids: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
    )

    # Вектор тезиса в режиме query.
    #
    # Используется для поиска подтверждающих
    # смысловых единиц среди doc_embedding.
    query_embedding: list[float] | None = Field(
        default=None,
        min_length=1,
    )

    # Вектор тезиса в режиме document.
    #
    # Используется, например, для:
    # - поиска похожих тезисов;
    # - дедупликации;
    # - тематической кластеризации.
    doc_embedding: list[float] | None = Field(
        default=None,
        min_length=1,
    )

    # Модель, которой были построены embeddings.
    embedding_model: NonEmptyStr | None = None

    # Текущее состояние тезиса.
    #
    # pending_review — создан, но ещё не допущен к использованию;
    # active — можно использовать для генерации;
    # disabled — временно или постоянно исключён.
    status: ThesisStatus = ThesisStatus.PENDING_REVIEW

    # Сколько раз тезис уже использовался в публикациях.
    #
    # Это денормализованное поле для быстрого выбора темы.
    # Источником истины остаётся Publication.
    used_count: int = Field(
        default=0,
        ge=0,
    )

    # Когда тезис использовался последний раз.
    last_used_at: datetime | None = None

    # Дополнительные служебные данные.
    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_structure(self) -> "AtomicThesis":
        # Одна и та же SemanticUnit не должна быть указана дважды.
        if len(self.semantic_unit_ids) != len(set(self.semantic_unit_ids)):
            raise ValueError(
                "semantic_unit_ids must contain unique values"
            )

        has_embedding = (
            self.query_embedding is not None
            or self.doc_embedding is not None
        )

        # Если есть хотя бы один embedding,
        # должна быть известна использованная модель.
        if has_embedding and self.embedding_model is None:
            raise ValueError(
                "embedding_model is required when embeddings are specified"
            )

        # И наоборот: название модели без векторов не имеет смысла.
        if self.embedding_model is not None and not has_embedding:
            raise ValueError(
                "embedding_model cannot be specified without embeddings"
            )

        return self