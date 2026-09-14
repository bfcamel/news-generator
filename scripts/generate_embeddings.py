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
            "for SemanticUnits and AtomicTheses."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    status_parser = subparsers.add_parser(
        "status",
        help=(
            "Show embedding coverage "
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

    thesis_parser = subparsers.add_parser(
        "atomic-theses",
        help=(
            "Generate QUERY and/or DOC embeddings "
            "for Russian AtomicThesis.text."
        ),
    )

    _add_common_generation_arguments(
        thesis_parser
    )

    thesis_parser.add_argument(
        "--query-only",
        action="store_true",
        help=(
            "Generate only query embeddings."
        ),
    )

    thesis_parser.add_argument(
        "--doc-only",
        action="store_true",
        help=(
            "Generate only doc embeddings."
        ),
    )

    all_parser = subparsers.add_parser(
        "all",
        help=(
            "Generate missing SemanticUnit DOC "
            "and AtomicThesis QUERY+DOC embeddings."
        ),
    )

    _add_common_generation_arguments(
        all_parser
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
            "Maximum number of documents "
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
    print(
        "Semantic Units"
    )
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

    atomic = (
        await repository
        .atomic_theses_stats()
    )

    print()
    print(
        "Atomic Theses"
    )

    if atomic is None:
        print(
            "  index does not exist yet"
        )
        return

    print(
        f"  total:          "
        f"{atomic.total}"
    )
    print(
        f"  query complete: "
        f"{atomic.complete_query}"
    )
    print(
        f"  query missing:  "
        f"{atomic.missing_query}"
    )
    print(
        f"  doc complete:   "
        f"{atomic.complete_doc}"
    )
    print(
        f"  doc missing:    "
        f"{atomic.missing_doc}"
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
            "QUERY model: "
            f"{settings.query_model_uri}"
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

            if args.command == "semantic-units":
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

            elif args.command == "atomic-theses":
                if (
                    args.query_only
                    and args.doc_only
                ):
                    parser.error(
                        "--query-only and --doc-only "
                        "cannot be used together"
                    )

                generate_query = (
                    not args.doc_only
                )

                generate_doc = (
                    not args.query_only
                )

                stats = (
                    await service
                    .generate_atomic_theses(
                        force=args.force,
                        limit=args.limit,
                        generate_query=generate_query,
                        generate_doc=generate_doc,
                        progress=progress,
                    )
                )

                _print_stats(
                    "Atomic Theses",
                    stats,
                )

            elif args.command == "all":
                semantic_stats = (
                    await service
                    .generate_semantic_units(
                        force=args.force,
                        limit=args.limit,
                        progress=progress,
                    )
                )

                _print_stats(
                    "Semantic Units",
                    semantic_stats,
                )

                atomic_stats = (
                    await repository
                    .atomic_theses_stats()
                )

                if atomic_stats is None:
                    print()
                    print(
                        "AtomicTheses index does not "
                        "exist yet; skipping it."
                    )
                else:
                    thesis_stats = (
                        await service
                        .generate_atomic_theses(
                            force=args.force,
                            limit=args.limit,
                            progress=progress,
                        )
                    )

                    _print_stats(
                        "Atomic Theses",
                        thesis_stats,
                    )

    finally:
        await es.close()


def _print_stats(
    title: str,
    stats: object,
) -> None:
    print()
    print(
        "=" * 72
    )
    print(
        title
    )
    print(
        "=" * 72
    )

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
