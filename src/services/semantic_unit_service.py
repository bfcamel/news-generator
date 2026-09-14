from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from src.domain.semantic_unit import SemanticUnit
from src.infrastructure.embeddings import YandexEmbeddingClient
from src.repositories.embedding_repository import (
    EmbeddingRepository,
    EmbeddingTarget,
)
from src.repositories.semantic_unit_repository import (
    SemanticUnitRepository,
)


logger = logging.getLogger(
    "news_generator.semantic_units"
)


@dataclass(frozen=True, slots=True)
class SemanticUnitCreateResult:
    """
    Result of creating one SemanticUnit.

    The SemanticUnit itself is considered successfully created
    as soon as it has been written to Elasticsearch.

    Embedding generation is deliberately non-fatal:
    if Yandex is temporarily unavailable, the SemanticUnit remains
    in the database without doc_embedding and can be repaired later
    with scripts/generate_embeddings.py.
    """

    unit: SemanticUnit
    embedding_created: bool
    embedding_error: str | None = None


@dataclass(frozen=True, slots=True)
class SemanticUnitBatchResult:
    """
    Result of creating several SemanticUnits.
    """

    results: list[SemanticUnitCreateResult]

    @property
    def created_count(self) -> int:
        return len(self.results)

    @property
    def embeddings_created_count(self) -> int:
        return sum(
            result.embedding_created
            for result in self.results
        )

    @property
    def embeddings_failed_count(self) -> int:
        return (
            self.created_count
            - self.embeddings_created_count
        )


class SemanticUnitService:
    """
    Application service responsible for the SemanticUnit lifecycle.

    Responsibilities:
    1. Save a SemanticUnit through SemanticUnitRepository.
    2. Generate its DOC embedding through YandexEmbeddingClient.
    3. Save the embedding through EmbeddingRepository.

    Important:
    - Repository.create() remains pure Elasticsearch CRUD.
    - Embedding failure must not roll back a successfully saved
      SemanticUnit.
    - New SemanticUnits are immediately ready for semantic search
      whenever embedding generation succeeds.
    """

    def __init__(
        self,
        *,
        repository: SemanticUnitRepository,
        embedding_repository: EmbeddingRepository,
        embedding_client: YandexEmbeddingClient,
    ) -> None:
        self.repository = repository
        self.embedding_repository = embedding_repository
        self.embedding_client = embedding_client

    async def create(
        self,
        unit: SemanticUnit,
        *,
        generate_embedding: bool = True,
    ) -> SemanticUnitCreateResult:
        """
        Save one SemanticUnit and optionally generate its DOC embedding.

        Repository errors are propagated because they mean that the
        SemanticUnit itself was not created.

        Embedding errors are caught and returned in the result, because
        the scientific data has already been safely persisted.
        """

        await self.repository.create(
            unit
        )

        logger.info(
            "SemanticUnit created: id=%s source_document_id=%s",
            unit.id,
            unit.source_document_id,
        )

        if not generate_embedding:
            return SemanticUnitCreateResult(
                unit=unit,
                embedding_created=False,
            )

        return await self._generate_embedding(
            unit
        )

    async def create_many(
        self,
        units: list[SemanticUnit],
        *,
        generate_embeddings: bool = True,
    ) -> SemanticUnitBatchResult:
        """
        Save several SemanticUnits.

        All units are persisted first. Only after successful persistence
        do we start embedding generation.

        This is preferable for JSON imports:
        a temporary Yandex API problem cannot interrupt the actual import,
        and embeddings can be generated concurrently afterwards.

        Repository errors are propagated. If an import route needs
        per-unit duplicate/error handling, it should keep that handling
        around calls to repository/service at the route/import layer.
        """

        if not units:
            return SemanticUnitBatchResult(
                results=[]
            )

        for unit in units:
            await self.repository.create(
                unit
            )

            logger.info(
                "SemanticUnit created: id=%s source_document_id=%s",
                unit.id,
                unit.source_document_id,
            )

        if not generate_embeddings:
            return SemanticUnitBatchResult(
                results=[
                    SemanticUnitCreateResult(
                        unit=unit,
                        embedding_created=False,
                    )
                    for unit in units
                ]
            )

        results = await asyncio.gather(
            *[
                self._generate_embedding(
                    unit
                )
                for unit in units
            ]
        )

        return SemanticUnitBatchResult(
            results=list(results)
        )

    async def generate_embedding(
        self,
        unit: SemanticUnit,
    ) -> SemanticUnitCreateResult:
        """
        Public helper for generating/re-generating an embedding for
        an already existing SemanticUnit.

        The caller is responsible for ensuring that the unit already
        exists in Elasticsearch.
        """

        return await self._generate_embedding(
            unit
        )

    async def _generate_embedding(
        self,
        unit: SemanticUnit,
    ) -> SemanticUnitCreateResult:
        try:
            embedding = (
                await self.embedding_client.embed_doc(
                    unit.text
                )
            )

            await (
                self.embedding_repository
                .save_semantic_unit_doc_embedding(
                    target=EmbeddingTarget(
                        es_id=unit.id,
                        domain_id=unit.id,
                        text=unit.text,
                        metadata=dict(
                            unit.metadata
                            or {}
                        ),
                        doc_embedding_exists=False,
                    ),
                    vector=embedding.vector,
                    embedding_metadata=(
                        embedding.metadata()
                    ),
                )
            )

            logger.info(
                "DOC embedding created for SemanticUnit: "
                "id=%s model=%s version=%s dim=%s",
                unit.id,
                embedding.model_uri,
                embedding.model_version,
                embedding.dimension,
            )

            return SemanticUnitCreateResult(
                unit=unit,
                embedding_created=True,
            )

        except Exception as exc:
            logger.exception(
                "SemanticUnit %s was saved, but DOC embedding "
                "generation failed",
                unit.id,
            )

            return SemanticUnitCreateResult(
                unit=unit,
                embedding_created=False,
                embedding_error=str(exc),
            )
