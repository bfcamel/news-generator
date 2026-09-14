# AtomicThesis technical base

Это технический фундамент для следующего слоя базы знаний.

## Модель

`AtomicThesis` хранит:

- `id`
- `language = "ru"`
- `text`
- `text_hash`
- `semantic_unit_ids`
- `taxa`
- `taxon_scope`
- `query_embedding`
- `doc_embedding`
- `embedding_model`
- `status`
- `used_count`
- `last_used_at`
- `metadata`
- timestamps

### Status

- `pending_review` — создан, но не допущен к публикациям.
- `active` — можно использовать.
- `disabled` — сохранён, но не использовать.

### TaxonScope

- `species`
- `multi_species`
- `genus`
- `family`
- `unspecified`

## Почему два embedding

Для русского `AtomicThesis.text`:

- `query_embedding` (`text-embeddings-v2-query`) используется для
  поиска релевантных SemanticUnit.
- `doc_embedding` (`text-embeddings-v2-doc`) используется для
  поиска похожих AtomicThesis при дедупликации.

Cosine similarity используется только для candidate retrieval.
Он не доказывает семантическую эквивалентность.

## Установка файлов

Скопировать:

```text
src/domain/atomic_thesis.py
src/infrastructure/elasticsearch/atomic_theses_index.py
src/repositories/atomic_thesis_repository.py
src/services/atomic_thesis_service.py
scripts/ensure_atomic_theses_index.py
scripts/add_atomic_thesis.py
tests/test_atomic_thesis.py
```

Существующий `EmbeddingRepository` менять не нужно: структура
совместима с уже реализованными `query_embedding`,
`doc_embedding` и `embedding_model`.

## Создать индекс

```bash
uv run python -m scripts.ensure_atomic_theses_index
```

После этого:

```bash
uv run python -m scripts.generate_embeddings status
```

должен показывать уже существующий индекс Atomic Theses с нулём
документов.

## Интеграция в ensure_indices()

Чтобы индекс автоматически создавался при запуске FastAPI,
добавить в `src/infrastructure/elasticsearch/indices.py`:

```python
from src.infrastructure.elasticsearch.atomic_theses_index import (
    ensure_atomic_theses_index,
)
```

и в существующую `ensure_indices()`:

```python
await ensure_atomic_theses_index()
```

## Ручной smoke test

Нужен реальный `unit_...` из `semantic_units`.

```bash
uv run python -m scripts.add_atomic_thesis \
  --text "Беременность у дромадера длится около 13 месяцев." \
  --unit unit_REAL_ID \
  --taxon "Camelus dromedarius" \
  --scope species
```

Результат должен показать:

```json
{
  "status": "pending_review",
  "semantic_unit_count": 1,
  "source_document_count": 1,
  "query_embedding_created": true,
  "doc_embedding_created": true
}
```

## Exact dedup

Перед записью Repository проверяет `text_hash`.

Это ловит одинаковые формулировки с разным регистром и
лишними пробелами.

Но:

```text
"Беременность длится около 13 месяцев."
```

и

```text
"Продолжительность беременности составляет примерно 13 месяцев."
```

имеют разные hashes.

Их эквивалентность позже определяет отдельный pipeline:

```text
CandidateAtomicThesis
→ doc embedding
→ Top-K похожих AtomicThesis
→ LLM equivalence verifier
→ CREATE или ADD SUPPORT
```

## Несколько источников

`semantic_unit_ids` может содержать несколько SemanticUnit.

`support_info()` вычисляет число независимых SourceDocument
через `SemanticUnit.source_document_id`, поэтому:

```text
5 SemanticUnit из одной статьи
```

не превращаются в:

```text
5 независимых источников
```

## Embedding failure

Тезис сначала сохраняется в Elasticsearch.

Если Yandex временно недоступен, тезис остаётся в базе.
Недостающие embeddings можно восстановить:

```bash
uv run python -m scripts.generate_embeddings atomic-theses
```

## Следующий слой

После проверки этого фундамента можно строить:

```text
SemanticUnit
→ LLM thesis extraction
→ CandidateAtomicThesis
→ candidate similarity search
→ LLM equivalence/support check
→ AtomicThesis create / merge support
```
