# Embeddings subsystem

## What it does

The subsystem implements production embedding generation for the knowledge base.

Current rules:

- `SemanticUnit.text` is stored in the source language.
- `SemanticUnit.doc_embedding` is generated with
  `text-embeddings-v2-doc`.
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

Optional:

```env
YANDEX_EMBEDDING_DIM=512
YANDEX_EMBEDDING_CONCURRENCY=3
YANDEX_EMBEDDING_TIMEOUT=60
YANDEX_EMBEDDING_MAX_RETRIES=5
```

## Quick connection test

```bash
uv run python scripts/test_embedding_connection.py
```

## Coverage status

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

## Elasticsearch mapping requirement

`semantic_units.doc_embedding` must be a `dense_vector` with `dims: 512`.
