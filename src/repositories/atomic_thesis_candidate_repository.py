from __future__ import annotations

import hashlib

from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.domain.atomic_thesis import (
    TaxonScope,
)
from src.domain.atomic_thesis_candidate import (
    CandidateAtomicThesis,
    CandidateStatus,
    ExtractionStatus,
    SemanticUnitThesisExtraction,
    calculate_candidate_hash,
)
from src.domain.semantic_unit import (
    SemanticUnit,
)
from src.infrastructure.elasticsearch.atomic_thesis_candidates_index import (
    ATOMIC_THESIS_CANDIDATES_INDEX,
    ATOMIC_THESIS_EXTRACTIONS_INDEX,
)


SEMANTIC_UNITS_INDEX = (
    "semantic_units"
)


class CandidateNotFoundError(
    LookupError
):
    pass


class CandidateAlreadyReviewedError(
    ValueError
):
    pass


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def make_extraction_id(
    *,
    semantic_unit_id: str,
    prompt_version: str,
) -> str:
    value = (
        f"{semantic_unit_id}|"
        f"{prompt_version}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return (
        f"extract_{digest[:32]}"
    )


class AtomicThesisCandidateRepository:
    def __init__(
        self,
        es: AsyncElasticsearch,
    ) -> None:
        self.es = es

    async def find_exact_candidate(
        self,
        *,
        semantic_unit_id: str,
        text_hash: str,
    ) -> CandidateAtomicThesis | None:
        response = await self.es.search(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            size=1,
            query={
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "source_semantic_unit_id": (
                                    semantic_unit_id
                                )
                            }
                        },
                        {
                            "term": {
                                "text_hash": (
                                    text_hash
                                )
                            }
                        },
                    ]
                }
            },
        )

        hits = (
            response[
                "hits"
            ][
                "hits"
            ]
        )

        if not hits:
            return None

        return (
            CandidateAtomicThesis
            .model_validate(
                hits[0][
                    "_source"
                ]
            )
        )

    async def create_candidate(
        self,
        candidate: CandidateAtomicThesis,
    ) -> CandidateAtomicThesis:
        existing = (
            await self.find_exact_candidate(
                semantic_unit_id=(
                    candidate
                    .source_semantic_unit_id
                ),
                text_hash=str(
                    candidate.text_hash
                ),
            )
        )

        if existing is not None:
            return existing

        await self.es.index(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            id=candidate.id,
            document=candidate.model_dump(
                mode="json"
            ),
            refresh="wait_for",
        )

        return candidate

    async def get_candidate(
        self,
        candidate_id: str,
    ) -> CandidateAtomicThesis | None:
        response = await self.es.get(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            id=candidate_id,
            ignore=[404],
        )

        if not response.get(
            "found"
        ):
            return None

        return (
            CandidateAtomicThesis
            .model_validate(
                response[
                    "_source"
                ]
            )
        )

    async def require_candidate(
        self,
        candidate_id: str,
    ) -> CandidateAtomicThesis:
        candidate = (
            await self.get_candidate(
                candidate_id
            )
        )

        if candidate is None:
            raise CandidateNotFoundError(
                candidate_id
            )

        return candidate

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | None = None,
        source_document_id: str | None = None,
        size: int = 500,
    ) -> list[CandidateAtomicThesis]:
        filters: list[
            dict[str, Any]
        ] = []

        if status is not None:
            filters.append(
                {
                    "term": {
                        "status": status.value
                    }
                }
            )

        if source_document_id:
            filters.append(
                {
                    "term": {
                        "source_document_id": (
                            source_document_id
                        )
                    }
                }
            )

        query: dict[str, Any]

        if filters:
            query = {
                "bool": {
                    "filter": filters
                }
            }
        else:
            query = {
                "match_all": {}
            }

        response = await self.es.search(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            size=size,
            query=query,
            sort=[
                {
                    "created_at": {
                        "order": "asc",
                    }
                }
            ],
        )

        return [
            CandidateAtomicThesis
            .model_validate(
                hit[
                    "_source"
                ]
            )
            for hit in response[
                "hits"
            ][
                "hits"
            ]
        ]

    async def count_candidates(
        self,
        status: CandidateStatus,
    ) -> int:
        response = await self.es.count(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            query={
                "term": {
                    "status": status.value
                }
            },
        )

        return int(
            response[
                "count"
            ]
        )

    async def update_pending_candidate(
        self,
        *,
        candidate_id: str,
        text: str,
        taxa: list[str],
        taxon_scope: TaxonScope,
        review_note: str | None,
    ) -> CandidateAtomicThesis:
        current = (
            await self.require_candidate(
                candidate_id
            )
        )

        if (
            current.status
            != CandidateStatus.PENDING_REVIEW
        ):
            raise (
                CandidateAlreadyReviewedError(
                    candidate_id
                )
            )

        now = utc_now()

        data = current.model_dump(
            mode="python"
        )

        data.update(
            {
                "text": text,
                "text_hash": (
                    calculate_candidate_hash(
                        text
                    )
                ),
                "taxa": taxa,
                "taxon_scope": taxon_scope,
                "review_note": review_note,
                "updated_at": now,
            }
        )

        updated = (
            CandidateAtomicThesis
            .model_validate(data)
        )

        await self.es.update(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            id=candidate_id,
            doc={
                "text": updated.text,
                "text_hash": (
                    updated.text_hash
                ),
                "taxa": updated.taxa,
                "taxon_scope": (
                    updated
                    .taxon_scope
                    .value
                ),
                "review_note": (
                    updated.review_note
                ),
                "updated_at": (
                    now.isoformat()
                ),
            },
            refresh="wait_for",
        )

        return updated

    async def mark_approved(
        self,
        *,
        candidate_id: str,
        atomic_thesis_id: str,
        review_note: str | None,
    ) -> CandidateAtomicThesis:
        current = (
            await self.require_candidate(
                candidate_id
            )
        )

        if (
            current.status
            != CandidateStatus.PENDING_REVIEW
        ):
            raise (
                CandidateAlreadyReviewedError(
                    candidate_id
                )
            )

        now = utc_now()

        await self.es.update(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            id=candidate_id,
            doc={
                "status": (
                    CandidateStatus
                    .APPROVED
                    .value
                ),
                "approved_atomic_thesis_id": (
                    atomic_thesis_id
                ),
                "review_note": review_note,
                "reviewed_at": (
                    now.isoformat()
                ),
                "updated_at": (
                    now.isoformat()
                ),
            },
            refresh="wait_for",
        )

        return (
            await self.require_candidate(
                candidate_id
            )
        )

    async def mark_rejected(
        self,
        *,
        candidate_id: str,
        review_note: str | None,
    ) -> CandidateAtomicThesis:
        current = (
            await self.require_candidate(
                candidate_id
            )
        )

        if (
            current.status
            != CandidateStatus.PENDING_REVIEW
        ):
            raise (
                CandidateAlreadyReviewedError(
                    candidate_id
                )
            )

        now = utc_now()

        await self.es.update(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            id=candidate_id,
            doc={
                "status": (
                    CandidateStatus
                    .REJECTED
                    .value
                ),
                "review_note": review_note,
                "reviewed_at": (
                    now.isoformat()
                ),
                "updated_at": (
                    now.isoformat()
                ),
            },
            refresh="wait_for",
        )

        return (
            await self.require_candidate(
                candidate_id
            )
        )

    async def save_extraction(
        self,
        extraction: (
            SemanticUnitThesisExtraction
        ),
    ) -> None:
        await self.es.index(
            index=(
                ATOMIC_THESIS_EXTRACTIONS_INDEX
            ),
            id=extraction.id,
            document=(
                extraction.model_dump(
                    mode="json"
                )
            ),
            refresh="wait_for",
        )

    async def completed_unit_ids(
        self,
        *,
        source_document_id: str,
        prompt_version: str,
    ) -> set[str]:
        response = await self.es.search(
            index=(
                ATOMIC_THESIS_EXTRACTIONS_INDEX
            ),
            size=10_000,
            query={
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "source_document_id": (
                                    source_document_id
                                )
                            }
                        },
                        {
                            "term": {
                                "prompt_version": (
                                    prompt_version
                                )
                            }
                        },
                        {
                            "term": {
                                "status": (
                                    ExtractionStatus
                                    .COMPLETED
                                    .value
                                )
                            }
                        },
                    ]
                }
            },
            source_includes=[
                "semantic_unit_id"
            ],
        )

        return {
            str(
                hit[
                    "_source"
                ][
                    "semantic_unit_id"
                ]
            )
            for hit in response[
                "hits"
            ][
                "hits"
            ]
        }

    async def units_for_generation(
            self,
            *,
            source_document_id: str,
            prompt_version: str,
            limit: int,
    ) -> list[SemanticUnit]:
        processed = (
            await self.completed_unit_ids(
                source_document_id=(
                    source_document_id
                ),
                prompt_version=(
                    prompt_version
                ),
            )
        )

        response = await self.es.search(
            index=SEMANTIC_UNITS_INDEX,
            size=1000,
            query={
                "term": {
                    "source_document_id": (
                        source_document_id
                    )
                }
            },
            sort=[
                {
                    "position": {
                        "order": "asc"
                    }
                }
            ],

            # Для извлечения тезисов embeddings SemanticUnit
            # вообще не нужны.
            #
            # Важно исключать ОБА поля одновременно,
            # иначе SemanticUnit получит embedding_model
            # без doc_embedding и Pydantic отклонит объект.
            source_excludes=[
                "doc_embedding",
                "embedding_model",
            ],
        )

        result: list[
            SemanticUnit
        ] = []

        for hit in response[
            "hits"
        ][
            "hits"
        ]:
            unit = (
                SemanticUnit
                .model_validate(
                    hit[
                        "_source"
                    ]
                )
            )

            if unit.id in processed:
                continue

            result.append(
                unit
            )

            if len(result) >= limit:
                break

        return result

    async def get_semantic_units(
            self,
            unit_ids: list[str],
    ) -> dict[str, SemanticUnit]:
        if not unit_ids:
            return {}

        response = await self.es.mget(
            index=SEMANTIC_UNITS_INDEX,
            ids=list(
                dict.fromkeys(
                    unit_ids
                )
            ),

            # В Web UI вектор не нужен.
            # embedding_model тоже исключаем,
            # чтобы модель SemanticUnit оставалась валидной.
            source_excludes=[
                "doc_embedding",
                "embedding_model",
            ],
        )

        result: dict[
            str,
            SemanticUnit,
        ] = {}

        for doc in response[
            "docs"
        ]:
            if not doc.get(
                    "found"
            ):
                continue

            unit = (
                SemanticUnit
                .model_validate(
                    doc[
                        "_source"
                    ]
                )
            )

            result[
                unit.id
            ] = unit

        return result