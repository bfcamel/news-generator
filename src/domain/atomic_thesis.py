from __future__ import annotations

import hashlib
import re

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.domain.types import NonEmptyStr


EMBEDDING_DIMENSION = 512


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def normalize_thesis_text(
    text: str,
) -> str:
    """
    Нормализует текст только для вычисления text_hash.

    Сам AtomicThesis.text не переписывается:
    сохраняется исходная русская формулировка.
    """

    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    ).casefold()


def calculate_text_hash(
    text: str,
) -> str:
    normalized = normalize_thesis_text(
        text
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


class ThesisStatus(StrEnum):
    """
    Жизненный цикл тезиса.

    pending_review:
        тезис создан, но ещё не прошёл ручную/автоматическую
        проверку.

    active:
        тезис разрешён к использованию при генерации публикаций.

    disabled:
        тезис сохранён в базе, но исключён из использования.
    """

    PENDING_REVIEW = "pending_review"
    ACTIVE = "active"
    DISABLED = "disabled"


class TaxonScope(StrEnum):
    """
    Таксономический охват утверждения.
    """

    SPECIES = "species"
    MULTI_SPECIES = "multi_species"
    GENUS = "genus"
    FAMILY = "family"
    UNSPECIFIED = "unspecified"


class AtomicThesis(BaseModel):
    """
    Одно самостоятельное проверяемое научное утверждение.

    Важные правила:
    - text хранится на русском языке;
    - один тезис может подтверждаться несколькими SemanticUnit;
    - semantic_unit_ids — источник истины для доказательной базы;
    - одинаковые утверждения из разных источников не создают
      новые тезисы: к существующему тезису добавляется support;
    - query_embedding нужен для поиска SemanticUnit;
    - doc_embedding нужен для thesis-to-thesis similarity.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: NonEmptyStr

    # Вся база AtomicThesis хранится в русском языке.
    language: Literal["ru"] = "ru"

    text: NonEmptyStr

    # SHA-256 от нормализованного text.
    # Используется только для точного dedup.
    text_hash: NonEmptyStr | None = None

    # Один тезис должен иметь хотя бы одну доказательную
    # SemanticUnit.
    semantic_unit_ids: list[NonEmptyStr] = Field(
        min_length=1
    )

    # Таксономический охват тезиса.
    taxa: list[NonEmptyStr] = Field(
        default_factory=list
    )
    taxon_scope: TaxonScope = (
        TaxonScope.UNSPECIFIED
    )

    query_embedding: list[float] | None = Field(
        default=None,
        min_length=EMBEDDING_DIMENSION,
        max_length=EMBEDDING_DIMENSION,
    )

    doc_embedding: list[float] | None = Field(
        default=None,
        min_length=EMBEDDING_DIMENSION,
        max_length=EMBEDDING_DIMENSION,
    )

    # Совместимо с уже существующим EmbeddingRepository.
    # Точная информация по query/doc моделям хранится также
    # в metadata.embeddings.query/doc.
    embedding_model: NonEmptyStr | None = None

    status: ThesisStatus = (
        ThesisStatus.PENDING_REVIEW
    )

    used_count: int = Field(
        default=0,
        ge=0,
    )

    last_used_at: datetime | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    @model_validator(
        mode="before"
    )
    @classmethod
    def fill_text_hash(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(
            data,
            dict,
        ):
            return data

        text = data.get(
            "text"
        )

        if (
            isinstance(
                text,
                str,
            )
            and text.strip()
            and not data.get(
                "text_hash"
            )
        ):
            data = dict(
                data
            )

            data[
                "text_hash"
            ] = calculate_text_hash(
                text
            )

        return data

    @field_validator(
        "semantic_unit_ids"
    )
    @classmethod
    def unique_semantic_unit_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value
            )
        )

    @field_validator(
        "taxa"
    )
    @classmethod
    def unique_taxa(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                value
            )
        )

    @model_validator(
        mode="after"
    )
    def validate_taxon_scope(
        self,
    ) -> "AtomicThesis":
        count = len(
            self.taxa
        )

        if (
            self.taxon_scope
            == TaxonScope.SPECIES
            and count != 1
        ):
            raise ValueError(
                "taxon_scope=species requires "
                "exactly one taxon"
            )

        if (
            self.taxon_scope
            == TaxonScope.MULTI_SPECIES
            and count < 2
        ):
            raise ValueError(
                "taxon_scope=multi_species requires "
                "at least two taxa"
            )

        if (
            self.taxon_scope
            in {
                TaxonScope.GENUS,
                TaxonScope.FAMILY,
            }
            and count < 1
        ):
            raise ValueError(
                f"taxon_scope={self.taxon_scope.value} "
                "requires at least one taxon"
            )

        return self
