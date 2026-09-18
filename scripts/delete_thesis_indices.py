from __future__ import annotations

import argparse
import asyncio

from src.infrastructure.elasticsearch.client import es


THESIS_INDICES = (
    "atomic_theses",
    "atomic_thesis_candidates",
    "atomic_thesis_extractions",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete obsolete AtomicThesis-related "
            "Elasticsearch indices."
        )
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Actually delete the indices. "
            "Without this flag the script "
            "only shows what would be deleted."
        ),
    )

    return parser


async def main() -> None:
    args = build_parser().parse_args()

    try:
        existing_indices: list[str] = []

        print()
        print("Проверка индексов Elasticsearch")
        print("=" * 60)

        for index_name in THESIS_INDICES:
            exists = await es.indices.exists(
                index=index_name
            )

            if exists:
                existing_indices.append(
                    index_name
                )

                print(
                    f"[FOUND]   {index_name}"
                )
            else:
                print(
                    f"[MISSING] {index_name}"
                )

        print()

        if not existing_indices:
            print(
                "Индексов, связанных с тезисами, "
                "уже нет."
            )
            return

        if not args.yes:
            print(
                "Ничего не удалено."
            )
            print()
            print(
                "Будут удалены:"
            )

            for index_name in existing_indices:
                print(
                    f"  - {index_name}"
                )

            print()
            print(
                "Для фактического удаления запусти:"
            )
            print()
            print(
                "uv run python "
                "scripts/delete_thesis_indices.py "
                "--yes"
            )

            return

        print(
            "Удаление индексов..."
        )
        print()

        for index_name in existing_indices:
            await es.indices.delete(
                index=index_name,
            )

            print(
                f"[DELETED] {index_name}"
            )

        print()
        print(
            "Готово. Все индексы, связанные "
            "с AtomicThesis, удалены."
        )

    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )