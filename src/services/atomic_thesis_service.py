from __future__ import annotations

import asyncio
import logging

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.domain.atomic_thesis import (
    AtomicThesis,
    TaxonScope,
    ThesisStatus,
)
from src.infrastructure.embeddings import (
    EmbeddingResult,
    YandexEmbeddingClient,
)
from src.repositories.atomic_thesis_repository import (
    AtomicThesisRepository,
    ThesisSupportInfo,
)
from src.repositories.embedding_repository import (
    EmbeddingRepository,
    EmbeddingTarget,
)


logger = logging.getLogger(
    "news_generator.atomic_theses"
)


@dataclass(
    frozen=True,
    slots=True,
)
class AtomicThesisCreateResult:
    thesis: AtomicThesis
    query_embedding_created: bool
    doc_embedding_created: bool
    query_embedding_error: str | None = None
    doc_embedding_error: str | None = None

    @property
    def embeddings_complete(
        self,
    ) -> bool:
        return (
            self.query_embedding_created
            and self.doc_embedding_created
        )


class AtomicThesisService:
    """
    Application service для добавления AtomicThesis.

    Пока здесь НЕТ LLM-логики извлечения и определения
    семантической эквивалентности. Это сознательно отдельный
    следующий слой.

    Сервис уже умеет:
    - создать тезис;
    - проверить существование support SemanticUnit;
    - сохранить его в Elasticsearch;
    - автоматически построить query/doc embeddings;
    - добавить новый SemanticUnit как дополнительное
      подтверждение существующего тезиса;
    - менять статус тезиса.
    """

    def __init__(
        self,
        *,
        repository: AtomicThesisRepository,
        embedding_repository: EmbeddingRepository,
        embedding_client: YandexEmbeddingClient,
    ) -> None:
        self.repository = repository
        self.embedding_repository = (
            embedding_repository
        )
        self.embedding_client = (
            embedding_client
        )

    async def create_from_data(
        self,
        *,
        text: str,
        semantic_unit_ids: list[str],
        taxa: list[str] | None = None,
        taxon_scope: TaxonScope = (
            TaxonScope.UNSPECIFIED
        ),
        status: ThesisStatus = (
            ThesisStatus.PENDING_REVIEW
        ),
        metadata: dict[str, Any] | None = None,
        generate_embeddings: bool = True,
    ) -> AtomicThesisCreateResult:
        thesis = AtomicThesis(
            id=(
                f"thesis_{uuid4().hex}"
            ),
            text=text,
            semantic_unit_ids=(
                semantic_unit_ids
            ),
            taxa=taxa or [],
            taxon_scope=taxon_scope,
            status=status,
            metadata=metadata or {},
        )

        return await self.create(
            thesis,
            generate_embeddings=(
                generate_embeddings
            ),
        )

    async def create(
        self,
        thesis: AtomicThesis,
        *,
        generate_embeddings: bool = True,
    ) -> AtomicThesisCreateResult:
        await self.repository.create(
            thesis
        )

        logger.info(
            "AtomicThesis created: "
            "id=%s support_units=%d "
            "taxa=%r scope=%s status=%s",
            thesis.id,
            len(
                thesis.semantic_unit_ids
            ),
            thesis.taxa,
            thesis.taxon_scope.value,
            thesis.status.value,
        )

        if not generate_embeddings:
            return AtomicThesisCreateResult(
                thesis=thesis,
                query_embedding_created=False,
                doc_embedding_created=False,
            )

        return await self.generate_embeddings(
            thesis
        )

    async def generate_embeddings(
        self,
        thesis: AtomicThesis,
    ) -> AtomicThesisCreateResult:
        query_task = asyncio.create_task(
            self.embedding_client.embed_query(
                thesis.text
            )
        )

        doc_task = asyncio.create_task(
            self.embedding_client.embed_doc(
                thesis.text
            )
        )

        query_raw, doc_raw = await asyncio.gather(
            query_task,
            doc_task,
            return_exceptions=True,
        )

        query_result: EmbeddingResult | None
        doc_result: EmbeddingResult | None

        query_error: str | None = None
        doc_error: str | None = None

        if isinstance(
            query_raw,
            BaseException,
        ):
            query_result = None
            query_error = str(
                query_raw
            )

            logger.error(
                "QUERY embedding failed for "
                "AtomicThesis id=%s: %s",
                thesis.id,
                query_error,
            )
        else:
            query_result = query_raw

        if isinstance(
            doc_raw,
            BaseException,
        ):
            doc_result = None
            doc_error = str(
                doc_raw
            )

            logger.error(
                "DOC embedding failed for "
                "AtomicThesis id=%s: %s",
                thesis.id,
                doc_error,
            )
        else:
            doc_result = doc_raw

        if (
            query_result is not None
            or doc_result is not None
        ):
            target = EmbeddingTarget(
                es_id=thesis.id,
                domain_id=thesis.id,
                text=thesis.text,
                metadata=dict(
                    thesis.metadata
                    or {}
                ),
                doc_embedding_exists=False,
                query_embedding_exists=False,
            )

            await (
                self.embedding_repository
                .save_atomic_thesis_embeddings(
                    target=target,
                    query_vector=(
                        query_result.vector
                        if query_result
                        is not None
                        else None
                    ),
                    query_metadata=(
                        query_result.metadata()
                        if query_result
                        is not None
                        else None
                    ),
                    doc_vector=(
                        doc_result.vector
                        if doc_result
                        is not None
                        else None
                    ),
                    doc_metadata=(
                        doc_result.metadata()
                        if doc_result
                        is not None
                        else None
                    ),
                )
            )

        return AtomicThesisCreateResult(
            thesis=thesis,
            query_embedding_created=(
                query_result is not None
            ),
            doc_embedding_created=(
                doc_result is not None
            ),
            query_embedding_error=(
                query_error
            ),
            doc_embedding_error=(
                doc_error
            ),
        )

    async def add_support(
        self,
        *,
        thesis_id: str,
        semantic_unit_id: str,
    ) -> AtomicThesis:
        thesis = await (
            self.repository
            .add_semantic_unit_support(
                thesis_id=thesis_id,
                semantic_unit_id=(
                    semantic_unit_id
                ),
            )
        )

        logger.info(
            "Support added to AtomicThesis: "
            "thesis_id=%s semantic_unit_id=%s "
            "support_units=%d",
            thesis_id,
            semantic_unit_id,
            len(
                thesis.semantic_unit_ids
            ),
        )

        return thesis

    async def activate(
        self,
        thesis_id: str,
    ) -> AtomicThesis:
        return await self.repository.set_status(
            thesis_id=thesis_id,
            status=ThesisStatus.ACTIVE,
        )

    async def disable(
        self,
        thesis_id: str,
    ) -> AtomicThesis:
        return await self.repository.set_status(
            thesis_id=thesis_id,
            status=ThesisStatus.DISABLED,
        )

    async def mark_pending_review(
        self,
        thesis_id: str,
    ) -> AtomicThesis:
        return await self.repository.set_status(
            thesis_id=thesis_id,
            status=(
                ThesisStatus.PENDING_REVIEW
            ),
        )

    async def support_info(
        self,
        thesis_id: str,
    ) -> ThesisSupportInfo:
        return await self.repository.support_info(
            thesis_id
        )
