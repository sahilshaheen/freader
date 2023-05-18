# "Saving and loading" multiple FAISS indices in a single store is not supported: https://github.com/deepset-ai/haystack/issues/3554
import time
from src.schema import IndexORM


def retry(max_retries, wait_interval):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    print(f"Failed with {e}")
                    print(
                        f"Retry {retries} of {max_retries} in {wait_interval} seconds..."
                    )
                    time.sleep(wait_interval)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def list_depth(lst):
    if not isinstance(lst, list) or not lst:  # Base case
        return 0
    return 1 + max(list_depth(item) for item in lst)


def upsert_index(session, index_name, urls):
    index = session.query(IndexORM).filter(IndexORM.name == index_name).first()
    if index is None:
        index = IndexORM(name=index_name, urls=urls)
        session.add(index)
    else:
        for url in urls:
            if url not in index.urls:
                index.urls.append(url)
    session.commit()


def get_all_indices(session):
    return session.query(IndexORM).all()


def run_indexing_pipeline(pipeline, index_name, urls, depth):
    pipeline.run(
        params={
            "Crawler": {"urls": urls, "crawler_depth": depth},
            "Shaper": {"meta": {"index_name": index_name}},
        }
    )


def run_query_pipeline(
    pipeline,
    query,
    index_name,
    top_k,
    prompt_template="question-answering",
):
    retriever_params = {}
    if top_k is not None:
        retriever_params["top_k"] = top_k
    else:
        retriever_params["top_k"] = 5
    shaper_params = {}
    if index_name is not None:
        shaper_params["meta"] = {"index_name": index_name}
    return pipeline.run(
        query,
        params={
            "Retriever": retriever_params,
            "Shaper": shaper_params,
            "Prompter": {"prompt_template": prompt_template},
        },
    )


def add_metadata_to_documents(documents, meta):
    for doc in documents:
        doc.meta = {**doc.meta, **meta}
    return documents


def filter_documents_by_metadata(documents, meta={}):
    filtered_documents = []
    for doc in documents:
        if all([doc.meta.get(key) == value for key, value in meta.items()]):
            filtered_documents.append(doc)
    return filtered_documents
