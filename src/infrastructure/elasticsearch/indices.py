import asyncio
import logging
from typing import Any

from .client import es
from .mappings import (
    SEMANTIC_UNITS_INDEX,
    SEMANTIC_UNITS_MAPPINGS,
    SEMANTIC_UNITS_SETTINGS,
    SOURCE_DOCUMENTS_INDEX,
    SOURCE_DOCUMENTS_MAPPINGS,
    SOURCE_DOCUMENTS_SETTINGS,
)


logger = logging.getLogger(__name__)


INDICES: tuple[
    tuple[str, dict[str, Any], dict[str, Any]],
    ...,
] = (
    (
        SOURCE_DOCUMENTS_INDEX,
        SOURCE_DOCUMENTS_SETTINGS,
        SOURCE_DOCUMENTS_MAPPINGS,
    ),
    (
        SEMANTIC_UNITS_INDEX,
        SEMANTIC_UNITS_SETTINGS,
        SEMANTIC_UNITS_MAPPINGS,
    ),
)


async def ensure_indices() -> None:
    """
    Создаёт необходимые индексы Elasticsearch,
    если они ещё не существуют.

    Уже существующие индексы не изменяются.
    """

    for index_name, settings, mappings in INDICES:
        exists = await es.indices.exists(
            index=index_name,
        )

        if exists:
            logger.info(
                "Elasticsearch index '%s' already exists",
                index_name,
            )
            continue

        await es.indices.create(
            index=index_name,
            settings=settings,
            mappings=mappings,
        )

        logger.info(
            "Created Elasticsearch index '%s'",
            index_name,
        )


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    try:
        await ensure_indices()
    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(main())