from __future__ import annotations

import logging

from dataclasses import dataclass

from src.domain.atomic_thesis import (
    TaxonScope,
    ThesisStatus,
)
from src.domain.atomic_thesis_candidate import (
    CandidateAtomicThesis,
)
from src.repositories.atomic_thesis_candidate_repository import (
    AtomicThesisCandidateRepository,
)
from src.repositories.atomic_thesis_repository import (
    AtomicThesisRepository,
)
from src.services.atomic_thesis_service import (
    AtomicThesisService,
)


logger = logging.getLogger(
    "news_generator.atomic_thesis_moderation"
)


@dataclass(
    frozen=True,
    slots=True,
)
class CandidateApprovalResult:
    candidate: CandidateAtomicThesis
    atomic_thesis_id: str

    created_new_thesis: bool
    added_support: bool

    query_embedding_created: bool
    doc_embedding_created: bool

    query_embedding_error: str | None
    doc_embedding_error: str | None


class AtomicThesisModerationService:
    def __init__(
        self,
        *,
        candidate_repository: (
            AtomicThesisCandidateRepository
        ),
        atomic_thesis_repository: (
            AtomicThesisRepository
        ),
        atomic_thesis_service: (
            AtomicThesisService
        ),
    ) -> None:
        self.candidate_repository = (
            candidate_repository
        )

        self.atomic_thesis_repository = (
            atomic_thesis_repository
        )

        self.atomic_thesis_service = (
            atomic_thesis_service
        )

    async def edit(
        self,
        *,
        candidate_id: str,
        text: str,
        taxa: list[str],
        taxon_scope: TaxonScope,
        review_note: str | None,
    ) -> CandidateAtomicThesis:
        return await (
            self.candidate_repository
            .update_pending_candidate(
                candidate_id=candidate_id,
                text=text,
                taxa=taxa,
                taxon_scope=taxon_scope,
                review_note=review_note,
            )
        )

    async def approve(
        self,
        *,
        candidate_id: str,
        text: str,
        taxa: list[str],
        taxon_scope: TaxonScope,
        review_note: str | None,
    ) -> CandidateApprovalResult:
        candidate = await self.edit(
            candidate_id=candidate_id,
            text=text,
            taxa=taxa,
            taxon_scope=taxon_scope,
            review_note=review_note,
        )

        existing = (
            await self.atomic_thesis_repository
            .find_by_text_hash(
                str(
                    candidate.text_hash
                )
            )
        )

        if existing is not None:
            already_linked = (
                candidate
                .source_semantic_unit_id
                in existing.semantic_unit_ids
            )

            existing = (
                await self.atomic_thesis_service
                .add_support(
                    thesis_id=existing.id,
                    semantic_unit_id=(
                        candidate
                        .source_semantic_unit_id
                    ),
                )
            )

            query_created = (
                existing.query_embedding
                is not None
            )

            doc_created = (
                existing.doc_embedding
                is not None
            )

            query_error = None
            doc_error = None

            # Старый тезис мог существовать
            # до появления embeddings.
            if (
                existing.query_embedding
                is None
                or existing.doc_embedding
                is None
            ):
                embedding_result = (
                    await self.atomic_thesis_service
                    .generate_embeddings(
                        existing
                    )
                )

                query_created = (
                    embedding_result
                    .query_embedding_created
                )

                doc_created = (
                    embedding_result
                    .doc_embedding_created
                )

                query_error = (
                    embedding_result
                    .query_embedding_error
                )

                doc_error = (
                    embedding_result
                    .doc_embedding_error
                )

            approved = (
                await self.candidate_repository
                .mark_approved(
                    candidate_id=(
                        candidate.id
                    ),
                    atomic_thesis_id=(
                        existing.id
                    ),
                    review_note=(
                        review_note
                    ),
                )
            )

            return CandidateApprovalResult(
                candidate=approved,
                atomic_thesis_id=(
                    existing.id
                ),
                created_new_thesis=False,
                added_support=(
                    not already_linked
                ),
                query_embedding_created=(
                    query_created
                ),
                doc_embedding_created=(
                    doc_created
                ),
                query_embedding_error=(
                    query_error
                ),
                doc_embedding_error=(
                    doc_error
                ),
            )

        result = (
            await self.atomic_thesis_service
            .create_from_data(
                text=candidate.text,
                semantic_unit_ids=[
                    candidate
                    .source_semantic_unit_id
                ],
                taxa=candidate.taxa,
                taxon_scope=(
                    candidate.taxon_scope
                ),

                # Ручная проверка уже выполнена.
                status=ThesisStatus.ACTIVE,

                metadata={
                    "candidate_id": (
                        candidate.id
                    ),
                    "extraction_model": (
                        candidate
                        .extraction_model
                    ),
                    "prompt_version": (
                        candidate
                        .prompt_version
                    ),
                    "original_llm_text": (
                        candidate
                        .original_text
                    ),
                    "manually_verified": True,
                },

                # Embeddings появляются
                # только после approve.
                generate_embeddings=True,
            )
        )

        approved = (
            await self.candidate_repository
            .mark_approved(
                candidate_id=candidate.id,
                atomic_thesis_id=(
                    result.thesis.id
                ),
                review_note=review_note,
            )
        )

        logger.info(
            "AtomicThesis candidate approved: "
            "candidate=%s thesis=%s "
            "query_embedding=%s "
            "doc_embedding=%s",
            candidate.id,
            result.thesis.id,
            result.query_embedding_created,
            result.doc_embedding_created,
        )

        return CandidateApprovalResult(
            candidate=approved,
            atomic_thesis_id=(
                result.thesis.id
            ),
            created_new_thesis=True,
            added_support=False,
            query_embedding_created=(
                result.query_embedding_created
            ),
            doc_embedding_created=(
                result.doc_embedding_created
            ),
            query_embedding_error=(
                result.query_embedding_error
            ),
            doc_embedding_error=(
                result.doc_embedding_error
            ),
        )

    async def reject(
        self,
        *,
        candidate_id: str,
        review_note: str | None,
    ) -> CandidateAtomicThesis:
        return await (
            self.candidate_repository
            .mark_rejected(
                candidate_id=candidate_id,
                review_note=review_note,
            )
        )