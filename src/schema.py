from pydantic import BaseModel
from typing import List
from typing_extensions import Literal


class IndexRequest(BaseModel):
    index_name: str
    urls: List[str]
    crawler_depth: Literal[0, 1] = 0


class QueryRequest(BaseModel):
    query: str
    index_name: str
