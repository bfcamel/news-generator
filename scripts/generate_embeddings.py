from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import asdict

from src.infrastructure.elasticsearch.client import es
from src.infrastructure.embeddings import (
    EmbeddingSettings,
    YandexEmbeddingClient,
)
from src.repositories.embedding_repository import (
    EmbeddingRepository,
)
from src.services.embedding_generation_service import (
    EmbeddingGenerationService,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Yandex Text Embeddings v2 "
            "for SemanticUnit.text."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "status",
        help=(
            "Show SemanticUnit embedding coverage "
            "without calling Yandex API."
        ),
    )

    semantic_parser = subparsers.add_parser(
        "semantic-units",
        help=(
            "Generate DOC embeddings for "
            "SemanticUnit.text."
        ),
    )

    _add_common_generation_arguments(
        semantic_parser
    )

    return parser


def _add_common_generation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate embeddings even if "
            "the field is already populated."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of SemanticUnits "
            "to process."
        ),
    )


async def print_status(
    repository: EmbeddingRepository,
) -> None:
    semantic = (
        await repository
        .semantic_units_stats()
    )

    print()
    print("Semantic Units")
    print(
        f"  total:        "
        f"{semantic.total}"
    )
    print(
        f"  doc complete: "
        f"{semantic.complete_doc}"
    )
    print(
        f"  doc missing:  "
        f"{semantic.missing_doc}"
    )


async def progress(
    collection: str,
    current: int,
    total: int | None,
    domain_id: str,
) -> None:
    total_text = (
        str(total)
        if total is not None
        else "?"
    )

    print(
        f"[{collection}] "
        f"{current}/{total_text} "
        f"{domain_id}"
    )


async def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if (
        args.command != "status"
        and args.limit is not None
        and args.limit < 1
    ):
        parser.error(
            "--limit must be >= 1"
        )

    repository = EmbeddingRepository(
        es
    )

    try:
        if args.command == "status":
            await print_status(
                repository
            )
            return

        settings = (
            EmbeddingSettings.from_env()
        )

        print(
            "DOC model:   "
            f"{settings.doc_model_uri}"
        )
        print(
            "Dimension:   "
            f"{settings.dimension}"
        )
        print(
            "Concurrency: "
            f"{settings.concurrency}"
        )

        async with YandexEmbeddingClient(
            settings
        ) as client:
            service = (
                EmbeddingGenerationService(
                    client=client,
                    repository=repository,
                )
            )

            stats = (
                await service
                .generate_semantic_units(
                    force=args.force,
                    limit=args.limit,
                    progress=progress,
                )
            )

            _print_stats(
                "Semantic Units",
                stats,
            )

    finally:
        await es.close()


def _print_stats(
    title: str,
    stats: object,
) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    for key, value in asdict(
        stats
    ).items():
        print(
            f"{key:>18}: "
            f"{value}"
        )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
