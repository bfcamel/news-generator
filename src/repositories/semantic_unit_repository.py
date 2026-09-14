from src.domain.semantic_unit import SemanticUnit
from src.infrastructure.elasticsearch.client import es
from src.infrastructure.elasticsearch.mappings import SEMANTIC_UNITS_INDEX


class SemanticUnitRepository:
    async def create(self, unit: SemanticUnit) -> SemanticUnit:
        existing = await es.search(
            index=SEMANTIC_UNITS_INDEX,
            query={
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "source_document_id": unit.source_document_id,
                            }
                        },
                        {
                            "term": {
                                "text_hash": unit.text_hash,
                            }
                        },
                    ]
                }
            },
            size=1,
        )

        if existing["hits"]["hits"]:
            raise ValueError(
                "Такой SemanticUnit уже существует в этом источнике"
            )

        await es.index(
            index=SEMANTIC_UNITS_INDEX,
            id=unit.id,
            document=unit.model_dump(mode="json"),
            refresh="wait_for",
        )

        return unit

    async def count_by_source_document(
            self,
    ) -> dict[str, int]:
        response = await es.search(
            index=SEMANTIC_UNITS_INDEX,
            size=0,
            aggs={
                "by_source": {
                    "terms": {
                        "field": "source_document_id",
                        "size": 10_000,
                    }
                }
            },
        )

        buckets = response[
            "aggregations"
        ][
            "by_source"
        ][
            "buckets"
        ]

        return {
            bucket["key"]: bucket["doc_count"]
            for bucket in buckets
        }