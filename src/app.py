import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException

from src.schema import IndexRequest, QueryRequest, Base
from src.chains import get_readers, add_reader
from src.utils import upsert_index, get_all_indices, get_session
from sqlalchemy.orm import Session

load_dotenv()
logger = logging.getLogger("uvicorn")


readers = get_readers()

with get_session() as session:
    Base.metadata.create_all(session.get_bind())

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Server is up and running!"}


@app.get("/indices")
async def get_indices(session: Session = Depends(get_session)):
    return get_all_indices(session)


@app.post("/index")
async def index(background_tasks: BackgroundTasks, request: IndexRequest):
    if request.urls is None and request.raw is None:
        raise HTTPException(
            status_code=400, detail="Either urls or raw must be provided."
        )

    def scrape_and_index():
        if request.index_name not in readers:
            add_reader(readers, request.index_name)
        if request.urls:
            readers[request.index_name].index_multiple_urls(request.urls)
            upsert_index(get_session(), request.index_name, request.urls)
            logger.info(f"Succesfully indexed {request.urls} for {request.index_name}")
        if request.raw:
            readers[request.index_name].index_raw(request.raw, request.metadata)
            logger.info(f"Successfully indexed raw data for {request.index_name}")

    background_tasks.add_task(scrape_and_index)

    return {"message": "Indexing job has been started in the background."}


@app.post("/query")
async def run_query(request: QueryRequest):
    output = readers[request.index_name].query(request.query)

    return output


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
