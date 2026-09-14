from elasticsearch import AsyncElasticsearch


es = AsyncElasticsearch(
    "http://localhost:9200"
)