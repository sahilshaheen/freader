from pydantic import BaseModel
from typing import List, Optional
from typing_extensions import Literal
from haystack.document_stores.sql import ORMBase
from sqlalchemy import Column, String, ARRAY
from sqlalchemy.orm import relationship


class IndexRequest(BaseModel):
    index_name: str
    urls: List[str]
    crawler_depth: Literal[0, 1] = 0


class QueryRequest(BaseModel):
    query: str
    index_name: Optional[str]


class IndexORM(ORMBase):
    __tablename__ = "index"
    name = Column(String, unique=True)
    urls = Column(ARRAY(String), nullable=False)
