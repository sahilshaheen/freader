# "Saving and loading" multiple FAISS indices in a single store is not supported: https://github.com/deepset-ai/haystack/issues/3554


def run_indexing_pipeline(pipeline, index_name, urls, depth):
    pipeline.run(
        params={
            "crawler": {"urls": urls, "crawler_depth": depth},
            # "document_store": {"index": index_name},
        }
    )


def run_query_pipeline(
    pipeline, query, index_name, prompt_template="question-answering"
):
    return pipeline.run(
        query,
        params={
            "prompt_node": {"prompt_template": prompt_template},
            # "retriever": {"index": index_name},
        },
    )
