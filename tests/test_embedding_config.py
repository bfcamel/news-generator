from src.infrastructure.embeddings.config import (
    EmbeddingSettings,
)


def test_model_uris() -> None:
    settings = EmbeddingSettings(
        api_key="test",
        folder_id="folder123",
        dimension=512,
    )

    assert settings.doc_model_uri == (
        "emb://folder123/"
        "text-embeddings-v2-doc/latest"
    )

    assert settings.query_model_uri == (
        "emb://folder123/"
        "text-embeddings-v2-query/latest"
    )
