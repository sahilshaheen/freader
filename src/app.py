import os
import logging
import uvicorn
from dotenv import load_dotenv
from src.store import load_faiss_store
from src.schema import IndexRequest, QueryRequest
from src.utils import run_indexing_pipeline, run_query_pipeline
from fastapi import FastAPI, BackgroundTasks
from src.pipelines import load_indexing_pipeline, load_query_pipeline
from src.retrievers import load_embedding_retriever
from src.utils import upsert_index, get_all_indices

load_dotenv()
use_oai = os.getenv("USE_OPENAI_API")
api_key = os.getenv("OPENAI_API_KEY")
faiss_index_path = os.getenv("FAISS_INDEX_PATH")
if not faiss_index_path:
    raise ValueError(
        "FAISS index path is missing. Please set the FAISS_INDEX_PATH environment variable"
    )
logger = logging.getLogger("uvicorn")
store = load_faiss_store(
    faiss_index_path=faiss_index_path if os.path.exists(faiss_index_path) else None,
)
indexing_pipeline = load_indexing_pipeline(store)
retriever = load_embedding_retriever(
    store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    model_format="sentence_transformers",
)
if use_oai:
    if not api_key:
        raise ValueError(
            "OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable"
        )
    logger.warning("Using OpenAI for query pipeline.")
    # retriever = load_embedding_retriever(
    #     store,
    #     batch_size=8,
    #     embedding_model="ada",
    #     api_key=api_key,
    #     max_seq_len=1024,
    # )
    query_pipeline = load_query_pipeline(
        retriever, model_name_or_path="text-davinci-003", api_key=api_key
    )
else:
    query_pipeline = load_query_pipeline(retriever)
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Server is up and running!"}


@app.get("/indices")
async def get_indices():
    return get_all_indices(store.session)


@app.post("/index")
async def create_index(background_tasks: BackgroundTasks, request: IndexRequest):
    def scrape_and_index():
        run_indexing_pipeline(
            indexing_pipeline, request.index_name, request.urls, request.crawler_depth
        )
        store.update_embeddings(retriever)
        store.save(faiss_index_path)
        upsert_index(store.session, request.index_name, request.urls)

    background_tasks.add_task(scrape_and_index)

    return {"message": "Indexing job has been started in the background."}


@app.post("/query")
async def run_query(request: QueryRequest):
    query_res = run_query_pipeline(
        query_pipeline, request.query, request.index_name, request.top_k
    )
    return list(map(lambda answer: answer.to_dict(), query_res["answers"]))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
