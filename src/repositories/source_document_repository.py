from src.domain.source_document import SourceDocument
from src.infrastructure.elasticsearch.client import es
from src.infrastructure.elasticsearch.mappings import SOURCE_DOCUMENTS_INDEX


class SourceDocumentRepository:
    async def create(self, document: SourceDocument) -> SourceDocument:
        existing = await es.search(
            index=SOURCE_DOCUMENTS_INDEX,
            query={
                "term": {
                    "fingerprint": document.fingerprint,
                }
            },
            size=1,
        )

        if existing["hits"]["hits"]:
            raise ValueError("Такой источник уже существует")

        await es.index(
            index=SOURCE_DOCUMENTS_INDEX,
            id=document.id,
            document=document.model_dump(mode="json"),
            refresh="wait_for",
        )

        return document

    async def list_all(self) -> list[SourceDocument]:
        response = await es.search(
            index=SOURCE_DOCUMENTS_INDEX,
            query={"match_all": {}},
            sort=[
                {
                    "created_at": {
                        "order": "desc",
                    }
                }
            ],
            size=1000,
        )

        return [
            SourceDocument.model_validate(hit["_source"])
            for hit in response["hits"]["hits"]
        ]

    async def get(self, document_id: str) -> SourceDocument | None:
        response = await es.get(
            index=SOURCE_DOCUMENTS_INDEX,
            id=document_id,
            ignore=[404],
        )

        if not response.get("found"):
            return None

        return SourceDocument.model_validate(response["_source"])

