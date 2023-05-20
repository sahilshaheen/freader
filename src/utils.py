import os
import json
import time
from src.schema import IndexORM
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_session():
    engine = create_engine(os.environ["SQL_URL"])
    Session = sessionmaker(bind=engine)
    session = Session()
    return session


def retry(max_retries, wait_interval):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    print(f"Failed with {e}")
                    print(
                        f"Retry {retries} of {max_retries} in {wait_interval} seconds..."
                    )
                    time.sleep(wait_interval)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def list_depth(lst):
    if not isinstance(lst, list) or not lst:  # Base case
        return 0
    return 1 + max(list_depth(item) for item in lst)


def upsert_index(session, index_name, urls):
    index = session.query(IndexORM).filter(IndexORM.name == index_name).first()
    if index is None:
        index = IndexORM(name=index_name, urls=urls)
        session.add(index)
    else:
        for url in urls:
            if url not in index.urls:
                index.urls.append(url)
    session.commit()


def get_all_indices(session):
    return session.query(IndexORM).all()


def read_indices_list():
    with open("indices.json", "r") as f:
        return json.loads(f.read())


def write_indices_list(**kwargs):
    current_indices = read_indices_list()
    if kwargs["index_name"] not in current_indices:
        current_indices[kwargs["index_name"]] = kwargs
        with open("indices.json", "w") as f:
            f.write(json.dumps(current_indices))
