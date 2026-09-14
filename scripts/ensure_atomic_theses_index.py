from __future__ import annotations

import asyncio

from src.infrastructure.elasticsearch.atomic_theses_index import (
    ATOMIC_THESES_INDEX,
    ensure_atomic_theses_index,
)
from src.infrastructure.elasticsearch.client import es


async def main() -> None:
    try:
        await ensure_atomic_theses_index()

        mapping = await es.indices.get_mapping(
            index=ATOMIC_THESES_INDEX
        )

        print(
            f"Index '{ATOMIC_THESES_INDEX}' is ready."
        )

        properties = mapping[
            ATOMIC_THESES_INDEX
        ][
            "mappings"
        ][
            "properties"
        ]

        print(
            "Fields:",
            ", ".join(
                sorted(
                    properties
                )
            ),
        )

    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
