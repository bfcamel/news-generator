# Add these fields to the atomic_theses Elasticsearch mapping
# when that index is implemented.

ATOMIC_THESIS_EMBEDDING_FIELDS = {
    "query_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": True,
        "similarity": "cosine",
    },
    "doc_embedding": {
        "type": "dense_vector",
        "dims": 512,
        "index": True,
        "similarity": "cosine",
    },
    "embedding_model": {
        "type": "keyword",
    },
}
