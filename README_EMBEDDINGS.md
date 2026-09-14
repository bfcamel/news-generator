# Embeddings subsystem

## What it does

The subsystem implements production embedding generation for the knowledge base.

Current rules:

- `SemanticUnit.text` is stored in the source language.
- `SemanticUnit.doc_embedding` is generated with
  `text-embeddings-v2-doc`.
- `AtomicThesis.text` is stored in Russian.
- `AtomicThesis.query_embedding` is generated with
  `text-embeddings-v2-query`.
- `AtomicThesis.doc_embedding` is also generated with
  `text-embeddings-v2-doc` for future thesis-to-thesis similarity and deduplication.
- Vector dimension is 512.

The exact Yandex `modelVersion`, token count, URI and dimension are stored inside
`metadata.embeddings`.

## .env

Required:

```env
YANDEX_API_KEY=***
YANDEX_FOLDER_ID=***
YANDEX_KEY_ID=***
```

`YANDEX_KEY_ID` is not used by this embedding client but can remain for other
Yandex Cloud functionality.

Optional:

```env
YANDEX_EMBEDDING_DIM=512
YANDEX_EMBEDDING_CONCURRENCY=3
YANDEX_EMBEDDING_TIMEOUT=60
YANDEX_EMBEDDING_MAX_RETRIES=5
```

## Dependencies

```bash
uv add httpx python-dotenv
```

For tests:

```bash
uv add --dev pytest pytest-asyncio
```

## Quick connection test

```bash
uv run python scripts/test_embedding_connection.py
```

## Coverage status

This command never calls Yandex API:

```bash
uv run python scripts/generate_embeddings.py status
```

## Generate missing SemanticUnit embeddings

```bash
uv run python scripts/generate_embeddings.py semantic-units
```

First small production test:

```bash
uv run python scripts/generate_embeddings.py semantic-units --limit 10
```

Regenerate every SemanticUnit:

```bash
uv run python scripts/generate_embeddings.py semantic-units --force
```

## AtomicThesis embeddings

Generate missing `query_embedding` and `doc_embedding`:

```bash
uv run python scripts/generate_embeddings.py atomic-theses
```

Only query embeddings:

```bash
uv run python scripts/generate_embeddings.py atomic-theses --query-only
```

Only doc embeddings:

```bash
uv run python scripts/generate_embeddings.py atomic-theses --doc-only
```

Regenerate both:

```bash
uv run python scripts/generate_embeddings.py atomic-theses --force
```

## Everything

```bash
uv run python scripts/generate_embeddings.py all
```

If the `atomic_theses` index does not exist yet, `all` generates SemanticUnit
embeddings and safely skips AtomicTheses.

## Stored metadata

Example for a SemanticUnit:

```json
{
  "doc_embedding": [0.1, 0.2],
  "embedding_model": "emb://.../text-embeddings-v2-doc/latest",
  "metadata": {
    "embeddings": {
      "doc": {
        "kind": "doc",
        "model_uri": "emb://.../text-embeddings-v2-doc/latest",
        "model_version": "...",
        "dimension": 512,
        "num_tokens": 37
      }
    }
  }
}
```

AtomicThesis stores separate metadata for `query` and `doc`.

## Error handling

The client retries network errors, 408, 425, 429 and common 5xx responses with
exponential backoff and jitter. A single failed document is logged and does not
abort generation of the rest of the collection.

## Elasticsearch mapping requirement

`semantic_units.doc_embedding` must be a `dense_vector` with `dims: 512`.

When `atomic_theses` is created, both:

- `query_embedding`
- `doc_embedding`

must be `dense_vector` fields with `dims: 512`.
