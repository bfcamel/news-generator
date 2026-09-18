from __future__ import annotations

from elasticsearch import AsyncElasticsearch

from src.infrastructure.elasticsearch.client import es


ATOMIC_THESIS_CANDIDATES_INDEX = (
    "atomic_thesis_candidates"
)

ATOMIC_THESIS_EXTRACTIONS_INDEX = (
    "atomic_thesis_extractions"
)


ATOMIC_THESIS_CANDIDATES_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "id": {
            "type": "keyword",
        },
        "source_semantic_unit_id": {
            "type": "keyword",
        },
        "source_document_id": {
            "type": "keyword",
        },
        "text": {
            "type": "text",
        },
        "original_text": {
            "type": "text",
        },
        "text_hash": {
            "type": "keyword",
        },
        "taxa": {
            "type": "keyword",
        },
        "taxon_scope": {
            "type": "keyword",
        },
        "status": {
            "type": "keyword",
        },
        "extraction_model": {
            "type": "keyword",
        },
        "prompt_version": {
            "type": "keyword",
        },
        "review_note": {
            "type": "text",
        },
        "approved_atomic_thesis_id": {
            "type": "keyword",
        },
        "metadata": {
            "type": "object",
            "enabled": False,
        },
        "created_at": {
            "type": "date",
        },
        "updated_at": {
            "type": "date",
        },
        "reviewed_at": {
            "type": "date",
        },
    },
}


ATOMIC_THESIS_EXTRACTIONS_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "id": {
            "type": "keyword",
        },
        "semantic_unit_id": {
            "type": "keyword",
        },
        "source_document_id": {
            "type": "keyword",
        },
        "status": {
            "type": "keyword",
        },
        "candidate_ids": {
            "type": "keyword",
        },
        "model": {
            "type": "keyword",
        },
        "prompt_version": {
            "type": "keyword",
        },
        "error": {
            "type": "text",
        },
        "created_at": {
            "type": "date",
        },
        "updated_at": {
            "type": "date",
        },
    },
}


async def ensure_atomic_thesis_candidate_indices(
    client: AsyncElasticsearch = es,
) -> None:
    candidates_exists = (
        await client.indices.exists(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            )
        )
    )

    if not candidates_exists:
        await client.indices.create(
            index=(
                ATOMIC_THESIS_CANDIDATES_INDEX
            ),
            mappings=(
                ATOMIC_THESIS_CANDIDATES_MAPPING
            ),
        )

    extractions_exists = (
        await client.indices.exists(
            index=(
                ATOMIC_THESIS_EXTRACTIONS_INDEX
            )
        )
    )

    if not extractions_exists:
        await client.indices.create(
            index=(
                ATOMIC_THESIS_EXTRACTIONS_INDEX
            ),
            mappings=(
                ATOMIC_THESIS_EXTRACTIONS_MAPPING
            ),
        )