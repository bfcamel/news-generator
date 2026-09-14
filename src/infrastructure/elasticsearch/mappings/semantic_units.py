SEMANTIC_UNITS_INDEX = "semantic_units"


SEMANTIC_UNITS_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}


SEMANTIC_UNITS_MAPPINGS = {
    "dynamic": "strict",
    "properties": {
        "id": {
            "type": "keyword",
        },

        "taxa": {
            "type": "keyword"
        },

        "source_document_id": {
            "type": "keyword",
        },

        "text": {
            "type": "text",
        },

        "text_hash": {
            "type": "keyword",
        },

        "position": {
            "type": "integer",
        },

        "section_title": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 512,
                }
            },
        },

        "char_start": {
            "type": "integer",
        },

        "char_end": {
            "type": "integer",
        },

        "page_start": {
            "type": "integer",
        },

        "page_end": {
            "type": "integer",
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

        "metadata": {
            "type": "object",
            "enabled": False,
        },

        "created_at": {
            "type": "date",
        },

        "updated_at": {
            "type": "date",
        },
    },
}