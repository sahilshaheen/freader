from typing import List, Optional

from pydantic import BaseModel
from typing_extensions import Literal
from sqlalchemy import Column, String, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import PrimaryKeyConstraint


class IndexRequest(BaseModel):
    index_name: str
    metadata: dict = None
    urls: List[str] = None
    raw: str = None
    crawler_depth: Literal[0, 1] = 0


class QueryRequest(BaseModel):
    query: str
    index_name: Optional[str] = "default"


Base = declarative_base()


class IndexORM(Base):
    __tablename__ = "index"

    name = Column(String, primary_key=True)
    urls = Column(ARRAY(String), nullable=False)

    __table_args__ = (PrimaryKeyConstraint("name"),)
