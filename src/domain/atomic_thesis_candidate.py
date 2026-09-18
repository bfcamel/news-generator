from __future__ import annotations

import hashlib
import re

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.domain.atomic_thesis import TaxonScope
from src.domain.types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_candidate_text(
    text: str,
) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.strip(),
    ).casefold()


def calculate_candidate_hash(
    text: str,
) -> str:
    return hashlib.sha256(
        normalize_candidate_text(
            text
        ).encode("utf-8")
    ).hexdigest()


class CandidateStatus(StrEnum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ExtractionStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractedAtomicThesis(BaseModel):
    """
    Один тезис непосредственно из structured output YandexGPT.

    Это ещё НЕ AtomicThesis.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: NonEmptyStr

    taxa: list[NonEmptyStr] = Field(
        default_factory=list
    )

    taxon_scope: TaxonScope = (
        TaxonScope.UNSPECIFIED
    )

    @field_validator("taxa")
    @classmethod
    def unique_taxa(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(value)
        )


class AtomicThesisExtractionResponse(
    BaseModel
):
    """
    Один SemanticUnit -> 0..N кандидатов.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    theses: list[
        ExtractedAtomicThesis
    ] = Field(
        default_factory=list
    )


class CandidateAtomicThesis(BaseModel):
    """
    Временный кандидат.

    ВАЖНО:
    здесь намеренно нет embeddings.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: NonEmptyStr

    source_semantic_unit_id: NonEmptyStr
    source_document_id: NonEmptyStr

    text: NonEmptyStr

    # Исходная формулировка YandexGPT.
    # После ручного редактирования text может измениться.
    original_text: NonEmptyStr

    text_hash: NonEmptyStr | None = None

    taxa: list[NonEmptyStr] = Field(
        default_factory=list
    )

    taxon_scope: TaxonScope = (
        TaxonScope.UNSPECIFIED
    )

    status: CandidateStatus = (
        CandidateStatus.PENDING_REVIEW
    )

    extraction_model: NonEmptyStr

    prompt_version: NonEmptyStr = (
        "atomic_thesis_extractor_v1"
    )

    review_note: str | None = None

    approved_atomic_thesis_id: (
        NonEmptyStr | None
    ) = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )

    reviewed_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def fill_text_hash(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data

        text = data.get("text")

        if (
            isinstance(text, str)
            and text.strip()
            and not data.get("text_hash")
        ):
            data = dict(data)

            data["text_hash"] = (
                calculate_candidate_hash(
                    text
                )
            )

        return data

    @field_validator("taxa")
    @classmethod
    def unique_taxa(
        cls,
        value: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(value)
        )

    @model_validator(mode="after")
    def validate_taxon_scope(
        self,
    ) -> "CandidateAtomicThesis":
        count = len(self.taxa)

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


class SemanticUnitThesisExtraction(
    BaseModel
):
    """
    Запись о том, что конкретный SemanticUnit
    уже был обработан LLM.

    Нужна в том числе для случая theses=[].
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: NonEmptyStr

    semantic_unit_id: NonEmptyStr
    source_document_id: NonEmptyStr

    status: ExtractionStatus

    candidate_ids: list[
        NonEmptyStr
    ] = Field(
        default_factory=list
    )

    model: NonEmptyStr
    prompt_version: NonEmptyStr

    error: str | None = None

    created_at: datetime = Field(
        default_factory=utc_now
    )

    updated_at: datetime = Field(
        default_factory=utc_now
    )