from __future__ import annotations

from elasticsearch import AsyncElasticsearch

from src.infrastructure.elasticsearch.client import es


ATOMIC_THESES_INDEX = "atomic_theses"
EMBEDDING_DIMENSION = 512


ATOMIC_THESES_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "id": {
            "type": "keyword",
        },
        "language": {
            "type": "keyword",
        },
        "text": {
            "type": "text",
        },
        "text_hash": {
            "type": "keyword",
        },
        "semantic_unit_ids": {
            "type": "keyword",
        },
        "taxa": {
            "type": "keyword",
        },
        "taxon_scope": {
            "type": "keyword",
        },
        "query_embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSION,
            "index": True,
            "similarity": "cosine",
        },
        "doc_embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSION,
            "index": True,
            "similarity": "cosine",
        },
        "embedding_model": {
            "type": "keyword",
        },
        "status": {
            "type": "keyword",
        },
        "used_count": {
            "type": "integer",
        },
        "last_used_at": {
            "type": "date",
        },
        # Корень индекса strict, но metadata намеренно dynamic:
        # сюда уже записывается metadata.embeddings.query/doc,
        # а позже можно безопасно добавлять prompt_version,
        # extraction_model и другую техническую информацию.
        "metadata": {
            "type": "object",
            "dynamic": True,
        },
        "created_at": {
            "type": "date",
        },
        "updated_at": {
            "type": "date",
        },
    },
}


async def ensure_atomic_theses_index(
    client: AsyncElasticsearch = es,
) -> None:
    exists = await client.indices.exists(
        index=ATOMIC_THESES_INDEX
    )

    if exists:
        return

    await client.indices.create(
        index=ATOMIC_THESES_INDEX,
        mappings=ATOMIC_THESES_MAPPING,
    )
