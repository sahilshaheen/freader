import os
from haystack.document_stores import ElasticsearchDocumentStore, FAISSDocumentStore


def load_es_store(**kwargs):
    document_store = ElasticsearchDocumentStore(**kwargs)
    return document_store


def load_faiss_store(**kwargs):
    if kwargs["faiss_index_path"] is None:
        sql_url = os.getenv("SQL_URL")
        if not sql_url:
            raise ValueError(
                "SQL_URL environment variable is required when creating a new store"
            )
        kwargs["sql_url"] = sql_url
        kwargs["embedding_dim"] = 384
    document_store = FAISSDocumentStore(**kwargs)
    return document_store
