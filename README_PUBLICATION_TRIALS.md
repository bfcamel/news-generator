# Publication generation trial

This is the temporary test pipeline for generating posts directly from
Semantic Units.

## Flow

1. The system samples Semantic Units that already have `doc_embedding`.
2. For each seed it retrieves nearest Semantic Units with Elasticsearch KNN.
3. It scores candidate groups by semantic similarity, evidence count and
   independent source count.
4. Recent test groups are temporarily excluded to reduce repetition.
5. The best evidence group is sent to YandexGPT in one request.
6. YandexGPT returns:
   - topic;
   - title;
   - post text;
   - IDs of Semantic Units actually used in the text.
7. The result is saved locally as JSON.
8. In the Web UI the user reviews the post and all selected evidence and marks
   it as "publish" or "reject".

No publication is sent to VK in this test mode.

## Storage

Trial results are stored in:

```text
var/publication_trials/
```

One JSON file is created per generated candidate.

This directory is ignored by Git and is not stored in Elasticsearch.

To reset only the test history, stop the app and remove this directory:

```bash
rm -rf var/publication_trials
```

This does not touch SourceDocuments, SemanticUnits or their embeddings.

## Prompt

The generation prompt is intentionally stored separately:

```text
src/prompts/post_generation.txt
```

The generator reads this file again for every generation request, so prompt
changes take effect without moving data or rebuilding Elasticsearch.

Each trial JSON stores the prompt path and SHA-256 hash used for that exact
generation.

## LLM calls

Normal generation uses exactly one YandexGPT request per post candidate.

Topic/evidence discovery itself uses Elasticsearch embeddings and does not call
an LLM.

## Web UI

Run the app as usual:

```bash
uv run uvicorn src.web.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/publications
```

Then click **Сгенерировать следующий пост**.
