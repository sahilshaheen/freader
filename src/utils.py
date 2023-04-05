# "Saving and loading" multiple FAISS indices in a single store is not supported: https://github.com/deepset-ai/haystack/issues/3554


def run_indexing_pipeline(pipeline, index_name, urls, depth):
    pipeline.run(
        params={
            "Crawler": {"urls": urls, "crawler_depth": depth},
            "Shaper": {"meta": {"index_name": index_name}},
        }
    )


def run_query_pipeline(
    pipeline, query, index_name, prompt_template="question-answering"
):
    return pipeline.run(
        query,
        params={
            "Prompter": {"prompt_template": prompt_template},
            "Shaper": {
                "meta": {"index_name": index_name} if index_name is not None else {}
            },
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
