from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.domain.atomic_thesis import (
    TaxonScope,
)
from src.domain.types import NonEmptyStr


REVIEW_SCHEMA_VERSION = (
    "atomic_thesis_review_v1"
)


class AtomicThesisReviewData(BaseModel):
    """
    Единственный блок экспортного файла,
    который разрешено менять при внешней проверке.

    Импорт в Web UI читает только этот блок.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: NonEmptyStr

    taxa: list[
        NonEmptyStr
    ] = Field(
        default_factory=list
    )

    taxon_scope: TaxonScope = (
        TaxonScope.UNSPECIFIED
    )

    review_note: str | None = None

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

    @field_validator(
        "review_note",
        mode="before",
    )
    @classmethod
    def normalize_review_note(
        cls,
        value,
    ):
        if value is None:
            return None

        value = str(
            value
        ).strip()

        return value or None

    @model_validator(
        mode="after"
    )
    def validate_taxon_scope(
        self,
    ) -> "AtomicThesisReviewData":
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
                f"taxon_scope="
                f"{self.taxon_scope.value} "
                "requires at least one taxon"
            )

        return self


class AtomicThesisReviewImport(BaseModel):
    """
    Формат, который принимает Web UI.

    Экспортный JSON содержит дополнительные
    evidence/source/candidate поля.

    extra='ignore' позволяет загрузить обратно
    тот же самый экспортный файл после изменения
    только блока review.
    """

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    schema_version: Literal[
        "atomic_thesis_review_v1"
    ]

    candidate_id: NonEmptyStr

    source_semantic_unit_id: NonEmptyStr

    review: AtomicThesisReviewData