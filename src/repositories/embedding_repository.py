from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from elasticsearch import AsyncElasticsearch


SEMANTIC_UNITS_INDEX = "semantic_units"


@dataclass(frozen=True, slots=True)
class EmbeddingTarget:
    es_id: str
    domain_id: str
    text: str
    metadata: dict[str, Any]
    doc_embedding_exists: bool


@dataclass(frozen=True, slots=True)
class IndexEmbeddingStats:
    total: int
    missing_doc: int

    @property
    def complete_doc(self) -> int:
        return (
            self.total
            - self.missing_doc
        )


class EmbeddingRepository:
    """
    Elasticsearch access used by SemanticUnit embedding generation.

    A scroll iterator is used instead of a fixed size=1000 query,
    so generation also works when the knowledge base grows.
    """

    def __init__(
        self,
        es: AsyncElasticsearch,
    ) -> None:
        self.es = es

    async def semantic_units_stats(
        self,
    ) -> IndexEmbeddingStats:
        total = await self.es.count(
            index=SEMANTIC_UNITS_INDEX
        )

        missing_doc = await self.es.count(
            index=SEMANTIC_UNITS_INDEX,
            query={
                "bool": {
                    "must_not": {
                        "exists": {
                            "field": (
                                "doc_embedding"
                            )
                        }
                    }
                }
            },
        )

        return IndexEmbeddingStats(
            total=int(
                total["count"]
            ),
            missing_doc=int(
                missing_doc["count"]
            ),
        )

    async def iter_semantic_units(
        self,
        *,
        force: bool = False,
        limit: int | None = None,
        batch_size: int = 250,
    ) -> AsyncIterator[EmbeddingTarget]:
        if force:
            query: dict[str, Any] = {
                "match_all": {}
            }
        else:
            query = {
                "bool": {
                    "must_not": {
                        "exists": {
                            "field": (
                                "doc_embedding"
                            )
                        }
                    }
                }
            }

        async for hit in self._scan(
            index=SEMANTIC_UNITS_INDEX,
            query=query,
            batch_size=batch_size,
            limit=limit,
        ):
            source = hit["_source"]

            text = str(
                source.get(
                    "text",
                    ""
                )
            ).strip()

            if not text:
                continue

            yield EmbeddingTarget(
                es_id=hit["_id"],
                domain_id=str(
                    source.get(
                        "id",
                        hit["_id"],
                    )
                ),
                text=text,
                metadata=dict(
                    source.get(
                        "metadata"
                    )
                    or {}
                ),
                doc_embedding_exists=(
                    source.get(
                        "doc_embedding"
                    )
                    is not None
                ),
            )

    async def save_semantic_unit_doc_embedding(
        self,
        *,
        target: EmbeddingTarget,
        vector: list[float],
        embedding_metadata: dict[str, Any],
    ) -> None:
        metadata = dict(
            target.metadata
        )

        embeddings_meta = dict(
            metadata.get(
                "embeddings"
            )
            or {}
        )

        embeddings_meta["doc"] = embedding_metadata
        metadata["embeddings"] = embeddings_meta

        await self.es.update(
            index=SEMANTIC_UNITS_INDEX,
            id=target.es_id,
            doc={
                "doc_embedding": vector,
                "embedding_model": str(
                    embedding_metadata[
                        "model_uri"
                    ]
                ),
                "metadata": metadata,
                "updated_at": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            },
        )

    async def _scan(
        self,
        *,
        index: str,
        query: dict[str, Any],
        batch_size: int,
        limit: int | None,
    ) -> AsyncIterator[dict[str, Any]]:
        response = await self.es.search(
            index=index,
            query=query,
            size=batch_size,
            sort=["_doc"],
            scroll="2m",
        )

        scroll_id = response.get(
            "_scroll_id"
        )

        emitted = 0

        try:
            while True:
                hits = response[
                    "hits"
                ][
                    "hits"
                ]

                if not hits:
                    break

                for hit in hits:
                    yield hit
                    emitted += 1

                    if (
                        limit is not None
                        and emitted >= limit
                    ):
                        return

                if not scroll_id:
                    break

                response = await self.es.scroll(
                    scroll_id=scroll_id,
                    scroll="2m",
                )

                scroll_id = response.get(
                    "_scroll_id",
                    scroll_id,
                )

        finally:
            if scroll_id:
                try:
                    await self.es.clear_scroll(
                        scroll_id=scroll_id
                    )
                except Exception:
                    pass
