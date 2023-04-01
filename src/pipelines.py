from haystack.pipelines import Pipeline
from haystack.nodes import Crawler, PreProcessor, BM25Retriever, PromptNode


def load_indexing_pipeline(document_store):
    crawler = Crawler()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=crawler, name="crawler", inputs=["File"])
    indexing_pipeline.add_node(
        component=preprocessor, name="preprocessor", inputs=["crawler"]
    )
    indexing_pipeline.add_node(
        component=document_store, name="document_store", inputs=["preprocessor"]
    )

    return indexing_pipeline


def load_query_pipeline(
    retriever, model_name_or_path="google/flan-t5-base", api_key=None
):
    prompt_node = PromptNode(model_name_or_path=model_name_or_path, api_key=api_key)
    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    query_pipeline.add_node(
        component=prompt_node, name="prompt_node", inputs=["retriever"]
    )
    return query_pipeline
