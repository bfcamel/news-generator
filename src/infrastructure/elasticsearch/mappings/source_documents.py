SOURCE_DOCUMENTS_INDEX = "source_documents"


SOURCE_DOCUMENTS_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}


SOURCE_DOCUMENTS_MAPPINGS = {
    "dynamic": "strict",
    "properties": {
        "id": {
            "type": "keyword",
        },

        "primary_taxon": {
            "type": "keyword",
        },

        "document_type": {
            "type": "keyword",
        },

        "title": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 512,
                }
            },
        },

        "authors": {
            "type": "keyword",
        },

        "editors": {
            "type": "keyword",
        },

        "year": {
            "type": "integer",
        },

        "container_type": {
            "type": "keyword",
        },

        "container_title": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 512,
                }
            },
        },

        "publisher": {
            "type": "text",
            "fields": {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 512,
                }
            },
        },

        "chapter_number": {
            "type": "keyword",
        },

        "page_start": {
            "type": "integer",
        },

        "page_end": {
            "type": "integer",
        },

        "doi": {
            "type": "keyword",
            "ignore_above": 512,
        },

        "isbn": {
            "type": "keyword",
        },

        "url": {
            "type": "keyword",
            "ignore_above": 2048,
        },

        "language": {
            "type": "keyword",
        },

        "full_text": {
            "type": "text",
        },

        "fingerprint": {
            "type": "keyword",
        },

        # metadata сохраняется в _source,
        # но Elasticsearch не создаёт для его содержимого
        # отдельные индексируемые поля.
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