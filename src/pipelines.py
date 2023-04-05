from haystack.pipelines import Pipeline
from haystack.nodes import Crawler, PreProcessor, PromptNode, Shaper

from src.utils import add_metadata_to_documents, filter_documents_by_metadata


def load_indexing_pipeline(store):
    crawler = Crawler()
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=100,
        split_respect_sentence_boundary=True,
    )
    shaper = Shaper("value_to_list", outputs=["documents"])
    shaper.function = add_metadata_to_documents

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_node(component=crawler, name="Crawler", inputs=["File"])
    indexing_pipeline.add_node(
        component=preprocessor, name="Preprocessor", inputs=["Crawler"]
    )
    indexing_pipeline.add_node(component=shaper, name="Shaper", inputs=["Preprocessor"])
    indexing_pipeline.add_node(component=store, name="DocumentStore", inputs=["Shaper"])

    return indexing_pipeline


def load_query_pipeline(
    retriever, model_name_or_path="google/flan-t5-base", api_key=None
):
    prompt_node = PromptNode(model_name_or_path=model_name_or_path, api_key=api_key)
    shaper = Shaper("value_to_list", outputs=["documents"])
    shaper.function = filter_documents_by_metadata

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=shaper, name="Shaper", inputs=["Retriever"])
    query_pipeline.add_node(component=prompt_node, name="Prompter", inputs=["Shaper"])

    return query_pipeline
