import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Depends

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
async def create_index(background_tasks: BackgroundTasks, request: IndexRequest):
    def scrape_and_index():
        if request.index_name not in readers:
            add_reader(readers, request.index_name)
        readers[request.index_name].index_multiple(request.urls)
        upsert_index(get_session(), request.index_name, request.urls)
        logger.info(f"Succesfully indexed {request.urls} for {request.index_name}")

    background_tasks.add_task(scrape_and_index)

    return {"message": "Indexing job has been started in the background."}


@app.post("/query")
async def run_query(request: QueryRequest):
    output = readers[request.index_name].query(request.query)

    return output


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
