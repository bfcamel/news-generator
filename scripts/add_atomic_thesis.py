from __future__ import annotations

import argparse
import asyncio
import json

from src.domain.atomic_thesis import (
    TaxonScope,
    ThesisStatus,
)
from src.infrastructure.elasticsearch.atomic_theses_index import (
    ensure_atomic_theses_index,
)
from src.infrastructure.elasticsearch.client import es
from src.infrastructure.embeddings import (
    EmbeddingSettings,
    YandexEmbeddingClient,
)
from src.repositories.atomic_thesis_repository import (
    AtomicThesisRepository,
)
from src.repositories.embedding_repository import (
    EmbeddingRepository,
)
from src.services.atomic_thesis_service import (
    AtomicThesisService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Manually create an AtomicThesis."
        )
    )

    parser.add_argument(
        "--text",
        required=True,
        help=(
            "Russian atomic thesis text."
        ),
    )

    parser.add_argument(
        "--unit",
        action="append",
        required=True,
        dest="semantic_unit_ids",
        help=(
            "Supporting SemanticUnit id. "
            "Repeat --unit for multiple supports."
        ),
    )

    parser.add_argument(
        "--taxon",
        action="append",
        default=[],
        dest="taxa",
        help=(
            "Taxon. Repeat --taxon if needed."
        ),
    )

    parser.add_argument(
        "--scope",
        choices=[
            item.value
            for item in TaxonScope
        ],
        default=(
            TaxonScope.UNSPECIFIED.value
        ),
    )

    parser.add_argument(
        "--status",
        choices=[
            item.value
            for item in ThesisStatus
        ],
        default=(
            ThesisStatus.PENDING_REVIEW.value
        ),
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help=(
            "Create thesis without calling "
            "Yandex Embeddings."
        ),
    )

    return parser


async def main() -> None:
    args = (
        build_parser()
        .parse_args()
    )

    await ensure_atomic_theses_index()

    settings = (
        EmbeddingSettings.from_env()
    )

    repository = AtomicThesisRepository(
        es
    )

    embedding_repository = (
        EmbeddingRepository(
            es
        )
    )

    client = YandexEmbeddingClient(
        settings
    )

    service = AtomicThesisService(
        repository=repository,
        embedding_repository=(
            embedding_repository
        ),
        embedding_client=client,
    )

    try:
        result = await service.create_from_data(
            text=args.text,
            semantic_unit_ids=(
                args.semantic_unit_ids
            ),
            taxa=args.taxa,
            taxon_scope=TaxonScope(
                args.scope
            ),
            status=ThesisStatus(
                args.status
            ),
            generate_embeddings=(
                not args.no_embeddings
            ),
        )

        support = await service.support_info(
            result.thesis.id
        )

        print(
            json.dumps(
                {
                    "id": result.thesis.id,
                    "text": result.thesis.text,
                    "status": (
                        result.thesis
                        .status
                        .value
                    ),
                    "semantic_unit_count": (
                        support
                        .semantic_unit_count
                    ),
                    "source_document_count": (
                        support
                        .source_document_count
                    ),
                    "source_document_ids": (
                        support
                        .source_document_ids
                    ),
                    "query_embedding_created": (
                        result
                        .query_embedding_created
                    ),
                    "doc_embedding_created": (
                        result
                        .doc_embedding_created
                    ),
                    "query_embedding_error": (
                        result
                        .query_embedding_error
                    ),
                    "doc_embedding_error": (
                        result
                        .doc_embedding_error
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        await client.close()
        await es.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
