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
