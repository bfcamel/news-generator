from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.domain.atomic_thesis import (
    AtomicThesis,
    ThesisStatus,
)
from src.infrastructure.elasticsearch.atomic_theses_index import (
    ATOMIC_THESES_INDEX,
)
from src.infrastructure.elasticsearch.client import es


SEMANTIC_UNITS_INDEX = "semantic_units"


def utc_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


class AtomicThesisAlreadyExistsError(
    ValueError
):
    def __init__(
        self,
        existing_id: str,
    ) -> None:
        self.existing_id = existing_id

        super().__init__(
            "AtomicThesis with the same "
            f"text_hash already exists: {existing_id}"
        )


class AtomicThesisNotFoundError(
    LookupError
):
    pass


class MissingSemanticUnitsError(
    ValueError
):
    def __init__(
        self,
        unit_ids: list[str],
    ) -> None:
        self.unit_ids = unit_ids

        super().__init__(
            "SemanticUnit not found: "
            + ", ".join(
                unit_ids
            )
        )


@dataclass(
    frozen=True,
    slots=True,
)
class ThesisSupportInfo:
    semantic_unit_count: int
    source_document_ids: list[str]

    @property
    def source_document_count(
        self,
    ) -> int:
        return len(
            self.source_document_ids
        )


class AtomicThesisRepository:
    """
    CRUD и технические операции для atomic_theses.

    Семантическая эквивалентность здесь НЕ определяется.
    Repository умеет только:
    - exact dedup по text_hash;
    - хранить связи с SemanticUnit;
    - выполнять vector candidate search.
    """

    def __init__(
        self,
        client: AsyncElasticsearch = es,
    ) -> None:
        self.es = client

    async def create(
        self,
        thesis: AtomicThesis,
    ) -> AtomicThesis:
        duplicate = (
            await self.find_by_text_hash(
                str(
                    thesis.text_hash
                )
            )
        )

        if duplicate is not None:
            raise AtomicThesisAlreadyExistsError(
                duplicate.id
            )

        missing = (
            await self.missing_semantic_unit_ids(
                thesis.semantic_unit_ids
            )
        )

        if missing:
            raise MissingSemanticUnitsError(
                missing
            )

        await self.es.index(
            index=ATOMIC_THESES_INDEX,
            id=thesis.id,
            document=thesis.model_dump(
                mode="json"
            ),
        )

        return thesis

    async def get(
        self,
        thesis_id: str,
    ) -> AtomicThesis | None:
        try:
            response = await self.es.get(
                index=ATOMIC_THESES_INDEX,
                id=thesis_id,
            )
        except Exception as exc:
            # elasticsearch.NotFoundError is intentionally
            # not imported here so the repository stays
            # tolerant of client-version differences.
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            if status_code == 404:
                return None

            meta = getattr(
                exc,
                "meta",
                None,
            )

            if (
                meta is not None
                and getattr(
                    meta,
                    "status",
                    None,
                )
                == 404
            ):
                return None

            raise

        return AtomicThesis.model_validate(
            response[
                "_source"
            ]
        )

    async def require(
        self,
        thesis_id: str,
    ) -> AtomicThesis:
        thesis = await self.get(
            thesis_id
        )

        if thesis is None:
            raise AtomicThesisNotFoundError(
                thesis_id
            )

        return thesis

    async def count(
        self,
    ) -> int:
        response = await self.es.count(
            index=ATOMIC_THESES_INDEX
        )

        return int(
            response[
                "count"
            ]
        )

    async def list_all(
        self,
        *,
        status: ThesisStatus | None = None,
        size: int = 1000,
    ) -> list[AtomicThesis]:
        query: dict[str, Any]

        if status is None:
            query = {
                "match_all": {}
            }
        else:
            query = {
                "term": {
                    "status": status.value
                }
            }

        response = await self.es.search(
            index=ATOMIC_THESES_INDEX,
            size=size,
            query=query,
            sort=[
                {
                    "created_at": {
                        "order": "desc",
                    }
                }
            ],
        )

        return [
            AtomicThesis.model_validate(
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

    async def find_by_text_hash(
        self,
        text_hash: str,
    ) -> AtomicThesis | None:
        response = await self.es.search(
            index=ATOMIC_THESES_INDEX,
            size=1,
            query={
                "term": {
                    "text_hash": text_hash
                }
            },
        )

        hits = response[
            "hits"
        ][
            "hits"
        ]

        if not hits:
            return None

        return AtomicThesis.model_validate(
            hits[
                0
            ][
                "_source"
            ]
        )

    async def list_by_semantic_unit(
        self,
        semantic_unit_id: str,
        *,
        size: int = 100,
    ) -> list[AtomicThesis]:
        response = await self.es.search(
            index=ATOMIC_THESES_INDEX,
            size=size,
            query={
                "term": {
                    "semantic_unit_ids": (
                        semantic_unit_id
                    )
                }
            },
        )

        return [
            AtomicThesis.model_validate(
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

    async def missing_semantic_unit_ids(
        self,
        unit_ids: list[str],
    ) -> list[str]:
        unique_ids = list(
            dict.fromkeys(
                unit_ids
            )
        )

        if not unique_ids:
            return []

        response = await self.es.mget(
            index=SEMANTIC_UNITS_INDEX,
            ids=unique_ids,
            source=False,
        )

        found = {
            str(
                doc[
                    "_id"
                ]
            )
            for doc in response[
                "docs"
            ]
            if doc.get(
                "found",
                False,
            )
        }

        return [
            unit_id
            for unit_id in unique_ids
            if unit_id not in found
        ]

    async def add_semantic_unit_support(
        self,
        *,
        thesis_id: str,
        semantic_unit_id: str,
    ) -> AtomicThesis:
        missing = (
            await self.missing_semantic_unit_ids(
                [
                    semantic_unit_id
                ]
            )
        )

        if missing:
            raise MissingSemanticUnitsError(
                missing
            )

        thesis = await self.require(
            thesis_id
        )

        if (
            semantic_unit_id
            in thesis.semantic_unit_ids
        ):
            return thesis

        semantic_unit_ids = [
            *thesis.semantic_unit_ids,
            semantic_unit_id,
        ]

        await self.es.update(
            index=ATOMIC_THESES_INDEX,
            id=thesis_id,
            doc={
                "semantic_unit_ids": (
                    semantic_unit_ids
                ),
                "updated_at": utc_iso(),
            },
        )

        return await self.require(
            thesis_id
        )

    async def set_status(
        self,
        *,
        thesis_id: str,
        status: ThesisStatus,
    ) -> AtomicThesis:
        await self.require(
            thesis_id
        )

        await self.es.update(
            index=ATOMIC_THESES_INDEX,
            id=thesis_id,
            doc={
                "status": status.value,
                "updated_at": utc_iso(),
            },
        )

        return await self.require(
            thesis_id
        )

    async def mark_used(
        self,
        thesis_id: str,
        *,
        used_at: datetime | None = None,
    ) -> AtomicThesis:
        await self.require(
            thesis_id
        )

        used_at = (
            used_at
            or datetime.now(
                timezone.utc
            )
        )

        await self.es.update(
            index=ATOMIC_THESES_INDEX,
            id=thesis_id,
            script={
                "lang": "painless",
                "source": (
                    "ctx._source.used_count += 1; "
                    "ctx._source.last_used_at = params.used_at; "
                    "ctx._source.updated_at = params.used_at;"
                ),
                "params": {
                    "used_at": used_at.isoformat(),
                },
            },
        )

        return await self.require(
            thesis_id
        )

    async def support_info(
        self,
        thesis_id: str,
    ) -> ThesisSupportInfo:
        thesis = await self.require(
            thesis_id
        )

        response = await self.es.mget(
            index=SEMANTIC_UNITS_INDEX,
            ids=thesis.semantic_unit_ids,
            source_includes=[
                "source_document_id"
            ],
        )

        source_document_ids = list(
            dict.fromkeys(
                str(
                    doc[
                        "_source"
                    ][
                        "source_document_id"
                    ]
                )
                for doc in response[
                    "docs"
                ]
                if (
                    doc.get(
                        "found",
                        False,
                    )
                    and doc.get(
                        "_source"
                    )
                    and doc[
                        "_source"
                    ].get(
                        "source_document_id"
                    )
                )
            )
        )

        return ThesisSupportInfo(
            semantic_unit_count=len(
                thesis.semantic_unit_ids
            ),
            source_document_ids=(
                source_document_ids
            ),
        )

    async def find_similar_by_doc_embedding(
        self,
        vector: list[float],
        *,
        k: int = 10,
        num_candidates: int = 100,
        status: ThesisStatus | None = None,
    ) -> list[
        tuple[
            AtomicThesis,
            float,
        ]
    ]:
        """
        Candidate retrieval only.

        Cosine score MUST NOT be treated as evidence that
        two theses are equivalent.
        """

        knn: dict[str, Any] = {
            "field": "doc_embedding",
            "query_vector": vector,
            "k": k,
            "num_candidates": max(
                num_candidates,
                k,
            ),
        }

        if status is not None:
            knn[
                "filter"
            ] = {
                "term": {
                    "status": status.value
                }
            }

        response = await self.es.search(
            index=ATOMIC_THESES_INDEX,
            size=k,
            knn=knn,
            source_excludes=[
                "query_embedding",
                "doc_embedding",
            ],
        )

        return [
            (
                AtomicThesis.model_validate(
                    hit[
                        "_source"
                    ]
                ),
                float(
                    hit[
                        "_score"
                    ]
                ),
            )
            for hit in response[
                "hits"
            ][
                "hits"
            ]
        ]
