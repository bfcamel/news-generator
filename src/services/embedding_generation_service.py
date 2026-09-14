from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.infrastructure.embeddings import (
    EmbeddingResult,
    YandexEmbeddingClient,
)
from src.repositories.embedding_repository import (
    EmbeddingRepository,
    EmbeddingTarget,
)


logger = logging.getLogger(
    "news_generator.embeddings"
)


ProgressCallback = Callable[
    [
        str,
        int,
        int | None,
        str,
    ],
    Awaitable[None] | None,
]


@dataclass(slots=True)
class GenerationStats:
    selected: int = 0
    generated_doc: int = 0
    generated_query: int = 0
    skipped_doc: int = 0
    skipped_query: int = 0
    failed: int = 0


class EmbeddingGenerationService:
    def __init__(
        self,
        *,
        client: YandexEmbeddingClient,
        repository: EmbeddingRepository,
    ) -> None:
        self.client = client
        self.repository = repository

    async def generate_semantic_units(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> GenerationStats:
        stats = GenerationStats()

        targets = [
            target
            async for target
            in self.repository.iter_semantic_units(
                force=force,
                limit=limit,
            )
        ]

        stats.selected = len(
            targets
        )

        semaphore = asyncio.Semaphore(
            self.client.settings.concurrency
        )

        async def worker(
            index: int,
            target: EmbeddingTarget,
        ) -> None:
            async with semaphore:
                try:
                    result = (
                        await self.client.embed_doc(
                            target.text
                        )
                    )

                    await (
                        self.repository
                        .save_semantic_unit_doc_embedding(
                            target=target,
                            vector=result.vector,
                            embedding_metadata=(
                                _result_metadata(
                                    result
                                )
                            ),
                        )
                    )

                    stats.generated_doc += 1

                    await _notify(
                        progress,
                        "semantic_units",
                        index,
                        stats.selected,
                        target.domain_id,
                    )

                except Exception:
                    stats.failed += 1

                    logger.exception(
                        "Failed to embed SemanticUnit "
                        "id=%s",
                        target.domain_id,
                    )

        await asyncio.gather(
            *[
                worker(
                    index,
                    target,
                )
                for index, target
                in enumerate(
                    targets,
                    start=1,
                )
            ]
        )

        return stats

    async def generate_atomic_theses(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        generate_query: bool = True,
        generate_doc: bool = True,
        progress: ProgressCallback | None = None,
    ) -> GenerationStats:
        if (
            not generate_query
            and not generate_doc
        ):
            raise ValueError(
                "At least one of generate_query "
                "or generate_doc must be True"
            )

        stats = GenerationStats()

        targets = [
            target
            async for target
            in self.repository.iter_atomic_theses(
                force=force,
                limit=limit,
            )
        ]

        stats.selected = len(
            targets
        )

        semaphore = asyncio.Semaphore(
            self.client.settings.concurrency
        )

        async def worker(
            index: int,
            target: EmbeddingTarget,
        ) -> None:
            async with semaphore:
                query_result: (
                    EmbeddingResult
                    | None
                ) = None

                doc_result: (
                    EmbeddingResult
                    | None
                ) = None

                need_query = (
                    generate_query
                    and (
                        force
                        or not (
                            target
                            .query_embedding_exists
                        )
                    )
                )

                need_doc = (
                    generate_doc
                    and (
                        force
                        or not (
                            target
                            .doc_embedding_exists
                        )
                    )
                )

                if (
                    generate_query
                    and not need_query
                ):
                    stats.skipped_query += 1

                if (
                    generate_doc
                    and not need_doc
                ):
                    stats.skipped_doc += 1

                try:
                    if need_query:
                        query_result = (
                            await (
                                self.client
                                .embed_query(
                                    target.text
                                )
                            )
                        )

                    if need_doc:
                        doc_result = (
                            await (
                                self.client
                                .embed_doc(
                                    target.text
                                )
                            )
                        )

                    if (
                        query_result is None
                        and doc_result is None
                    ):
                        return

                    await (
                        self.repository
                        .save_atomic_thesis_embeddings(
                            target=target,
                            query_vector=(
                                query_result.vector
                                if query_result
                                else None
                            ),
                            query_metadata=(
                                _result_metadata(
                                    query_result
                                )
                                if query_result
                                else None
                            ),
                            doc_vector=(
                                doc_result.vector
                                if doc_result
                                else None
                            ),
                            doc_metadata=(
                                _result_metadata(
                                    doc_result
                                )
                                if doc_result
                                else None
                            ),
                        )
                    )

                    if query_result:
                        stats.generated_query += 1

                    if doc_result:
                        stats.generated_doc += 1

                    await _notify(
                        progress,
                        "atomic_theses",
                        index,
                        stats.selected,
                        target.domain_id,
                    )

                except Exception:
                    stats.failed += 1

                    logger.exception(
                        "Failed to embed AtomicThesis "
                        "id=%s",
                        target.domain_id,
                    )

        await asyncio.gather(
            *[
                worker(
                    index,
                    target,
                )
                for index, target
                in enumerate(
                    targets,
                    start=1,
                )
            ]
        )

        return stats


def _result_metadata(
    result: EmbeddingResult,
) -> dict[str, object]:
    return result.metadata()


async def _notify(
    callback: ProgressCallback | None,
    collection: str,
    current: int,
    total: int | None,
    domain_id: str,
) -> None:
    if callback is None:
        return

    result = callback(
        collection,
        current,
        total,
        domain_id,
    )

    if asyncio.iscoroutine(
        result
    ):
        await result
