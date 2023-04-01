from haystack.nodes import BM25Retriever, EmbeddingRetriever


def load_bm25_retriever(store):
    retriever = BM25Retriever(document_store=store)
    return retriever


def load_embedding_retriever(store, **kwargs):
    retriever = EmbeddingRetriever(document_store=store, **kwargs)
    return retriever
