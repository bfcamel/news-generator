from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .types import NonEmptyStr


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PublicationTrialDecision(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class PublicationEvidence(BaseModel):
    """
    Snapshot of one SemanticUnit and its source at generation time.

    Trial publications deliberately keep this data outside Elasticsearch.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    semantic_unit_id: NonEmptyStr
    source_document_id: NonEmptyStr
    similarity: float = Field(ge=0.0)

    text: NonEmptyStr
    taxa: list[NonEmptyStr] = Field(default_factory=list)
    section_title: NonEmptyStr | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    source_title: NonEmptyStr
    source_authors: list[NonEmptyStr] = Field(default_factory=list)
    source_year: int | None = None
    source_doi: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None


class PublicationDiscoveryMetrics(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    seed_semantic_unit_id: NonEmptyStr
    average_similarity: float = Field(ge=0.0)
    source_count: int = Field(ge=1)
    evidence_count: int = Field(ge=1)
    selection_score: float


class PublicationTrial(BaseModel):
    """
    A generated post candidate stored in a local JSON file during testing.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    id: NonEmptyStr

    topic: NonEmptyStr
    title: NonEmptyStr
    post_text: NonEmptyStr

    selected_semantic_unit_ids: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
    )
    used_semantic_unit_ids: list[NonEmptyStr] = Field(
        ...,
        min_length=1,
    )

    evidence: list[PublicationEvidence] = Field(
        ...,
        min_length=1,
    )
    discovery: PublicationDiscoveryMetrics

    decision: PublicationTrialDecision = (
        PublicationTrialDecision.PENDING
    )
    decision_note: str | None = None
    decided_at: datetime | None = None

    generation_model: NonEmptyStr
    prompt_file: NonEmptyStr
    prompt_sha256: NonEmptyStr

    metadata: dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_structure(self) -> "PublicationTrial":
        selected = set(
            self.selected_semantic_unit_ids
        )

        if len(selected) != len(
            self.selected_semantic_unit_ids
        ):
            raise ValueError(
                "selected_semantic_unit_ids must be unique"
            )

        if len(
            set(self.used_semantic_unit_ids)
        ) != len(
            self.used_semantic_unit_ids
        ):
            raise ValueError(
                "used_semantic_unit_ids must be unique"
            )

        unknown_used = (
            set(self.used_semantic_unit_ids)
            - selected
        )

        if unknown_used:
            raise ValueError(
                "used_semantic_unit_ids must be a subset "
                "of selected_semantic_unit_ids"
            )

        evidence_ids = {
            item.semantic_unit_id
            for item in self.evidence
        }

        if evidence_ids != selected:
            raise ValueError(
                "evidence must contain exactly the selected "
                "SemanticUnit ids"
            )

        if (
            self.decision
            == PublicationTrialDecision.PENDING
            and self.decided_at is not None
        ):
            raise ValueError(
                "decided_at is not allowed while decision is pending"
            )

        if (
            self.decision
            != PublicationTrialDecision.PENDING
            and self.decided_at is None
        ):
            raise ValueError(
                "decided_at is required after a decision"
            )

        return self
