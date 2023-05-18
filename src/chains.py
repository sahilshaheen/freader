import os
import json
import logging
from typing import Literal, Optional

import requests
import anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader, UnstructuredHTMLLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings, FakeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFaceHub, HuggingFacePipeline
from langchain.chat_models import ChatAnthropic
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema import HumanMessage, AIMessage

from src.utils import (
    retry,
    list_depth,
    read_indices_list,
    write_indices_list,
)

LLM_CACHE = {}
EMBEDDING_MODEL_CACHE = {}

# import torch
# from transformers import pipeline

# Does not work on Apple silicon
# dolly_pipe = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
#                          trust_remote_code=True, device_map="auto", return_full_text=True)

CHAINS = {"qa": RetrievalQA, "qa_source": RetrievalQAWithSourcesChain}

LLMS = {
    "openai": OpenAI,
    "hf": HuggingFaceHub,
    "hf-local": lambda **kwargs: HuggingFacePipeline.from_model_id(**kwargs),
    "hf-pipe": HuggingFacePipeline,
    "claude": ChatAnthropic,
}

EMBEDDING_MODELS = {
    "instructor_large": HuggingFaceInstructEmbeddings,
    "fake": FakeEmbeddings,
}

MODEL_KWARGS = {"default": {}, "fake": {"size": 1}}

LLM_ARGS = {
    "default": {},
    "tagger": {"temperature": 0.8},
    "reader": {"temperature": 0},
    "stablelm": {
        "model_id": "stabilityai/stablelm-tuned-alpha-3b",
        "task": "text-generation",
        "model_kwargs": {},
    },
    "dolly": {
        "model_id": "databricks/dolly-v2-3b",
        "task": "text-generation",
        "model_kwargs": {},
    },
    # "dolly-pipe": {
    #     "pipeline": dolly_pipe
    # }
}


class LLM:
    def __init__(
        self,
        llm_id="openai",
        llm_kwargs_id="default",
    ):
        if (llm_id, llm_kwargs_id) in LLM_CACHE:
            self.llm = LLM_CACHE[(llm_id, llm_kwargs_id)]
        else:
            self.llm = LLMS[llm_id](**LLM_ARGS[llm_kwargs_id])
            LLM_CACHE[(llm_id, llm_kwargs_id)] = self.llm


class Freader(LLM):
    def __init__(
        self,
        index_name: str,
        faiss_path: str,
        chain_id: str = "qa_source",
        llm_id="openai",
        llm_kwargs_id="reader",
        model_id="instructor_large",
        model_kwargs_id="default",
        loader_type: Literal["bs", "unstructured"] = "bs",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        chain_type: str = "stuff",
    ):
        super().__init__(llm_id=llm_id, llm_kwargs_id=llm_kwargs_id)
        self.loader_type = loader_type
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.index_name = index_name
        if (model_id, model_kwargs_id) in EMBEDDING_MODEL_CACHE:
            self.model = EMBEDDING_MODEL_CACHE[(model_id, model_kwargs_id)]
        else:
            self.model = EMBEDDING_MODELS[model_id](**MODEL_KWARGS[model_kwargs_id])
            EMBEDDING_MODEL_CACHE[(model_id, model_kwargs_id)] = self.model
        self.faiss_path = faiss_path
        self.chain_type = chain_type
        self.chain_id = chain_id
        if os.path.exists(faiss_path):
            self.db = FAISS.load_local(faiss_path, self.model)
            self.chain = CHAINS[self.chain_id].from_chain_type(
                llm=self.llm, chain_type=chain_type, retriever=self.db.as_retriever()
            )
            if self.chain_id == "qa_source":
                self.chain.return_source_documents = True

    def infer_with_llm(self, text):
        return self.llm(text)

    def _index(self, url, loader_type):
        try:
            if loader_type is None:
                loader_type = self.loader_type
            res = requests.get(url)
            temp_filename = f"{url.split('/')[-1]}.html"
            with open(temp_filename, "w") as f:
                f.write(res.text)
            if self.loader_type == "unstructured":
                loader = UnstructuredHTMLLoader(file_path=temp_filename)
            elif self.loader_type == "bs":
                loader = BSHTMLLoader(file_path=temp_filename)
            else:
                raise ValueError("loader_type must be one of 'bs' or 'unstructured'")
            docs = loader.load()
            os.remove(temp_filename)
            text = docs[0].page_content
            metadata = docs[0].metadata
            docs = self.splitter.create_documents(
                texts=[text], metadatas=[{**metadata, "source": url}]
            )
            if hasattr(self, "db"):
                self.db.add_documents(docs)
            else:
                self.db = FAISS.from_documents(docs, self.model)
                self.chain = CHAINS[self.chain_id].from_chain_type(
                    llm=self.llm,
                    chain_type=self.chain_type,
                    retriever=self.db.as_retriever(),
                )
            self._save()
            return True
        except Exception as e:
            logging.error(f"Failed to index {url}: {e}")
        return False

    def index_multiple(
        self, urls, loader_type: Optional[Literal["bs", "unstructured"]] = None
    ):
        for url in urls:
            is_successful = self._index(url, loader_type=loader_type)
            if not is_successful:
                return False
        return True

    def query(self, text: str):
        if not hasattr(self, "chain"):
            raise ValueError("You must index some documents before you can query.")
        if self.chain_id == "qa":
            return self.chain.run(text)
        if self.chain_id == "qa_source":
            return self.chain({"question": text})
        raise ValueError(f"Unknown chain_id: {self.chain_id}")

    def _save(self):
        try:
            self.db.save_local(self.faiss_path)
            write_indices_list(self.index_name, self.faiss_path)
            return True
        except Exception as e:
            logging.error(f"Failed to save FAISS index to {self.faiss_path}: {e}")
        return False


class SongTagger(LLM):
    PROMPT = """Here are the song titles along with the artist name(s):\n{songs}"""
    MESSAGES = [
        HumanMessage(
            content='I want you to generate tags for the following songs. The tags must be musically and culturally informative as I plan to use them to find songs and make playlists. Generate around 5 tags per song that are information-rich. It is okay if you cannot meet this number, so do not generate low-quality tags to meet the requirement. It is also okay if you can generate more than 5 if you think the extra tags are high quality. The output must be in the format of a Python list of lists where each element corresponds to one song. Use double quotes and return ONLY the list. Use double quotes and return ONLY the list like so [["tag_1", "tag_2"], ["tag_2"]]. Do you understand?'
        ),
        AIMessage(
            content="I understand. I will aim to generate 3 to 8 high-quality, musically and culturally informative tags per song in the format of a Python list of lists with each inner list corresponding to the tags for one song."
        ),
    ]
    DUMMY = {"name": "Love Me Do", "artist": "The Beatles"}

    def __init__(self):
        super().__init__(llm_id="claude", llm_kwargs_id="tagger")
        self.prompter = HumanMessagePromptTemplate.from_template(SongTagger.PROMPT)

    def _format_tracks(self, tracks):
        formatted_tracks = "\n".join(
            [f'"{track["name"]}" by {track["artist"]}' for track in tracks]
        )
        return formatted_tracks

    @retry(3, 0.1)
    def _get_output_and_verify(self, input, songs):
        try:
            output = self.llm(input)
            output_obj = json.loads(output.content)
            assert (
                type(output_obj) == list
            ), f"Expected output to be a list, got {output_obj}"
            assert len(songs) + 1 == len(
                output_obj
            ), f"Expected {len(songs) + 1} songs (including dummy), got {len(output_obj)}"
            assert (
                list_depth(output_obj) == 2
            ), f"Expected output to be a list of lists, got {output_obj}"
            return output_obj[1:]
        except:
            print("Output:", output)
            raise

    def _normalize_output(self, output):
        normalized = []
        for song in output:
            song_tags = []
            for tag in song:
                # Remove special characters
                tag = "".join([c for c in tag if c.isalnum() or c == " "])

                # Lowercase
                tag = tag.lower()

                # Remove extra whitespaces
                tag = " ".join([word for word in tag.split(" ") if word])

                song_tags.append(tag)
            normalized.append(song_tags)
        return normalized

    def __call__(self, songs):
        songs_str = self._format_tracks([SongTagger.DUMMY, *songs])
        human_message = self.prompter.format(**{"songs": songs_str})
        input = [*SongTagger.MESSAGES, human_message]
        output = self._get_output_and_verify(input, songs)
        return self._normalize_output(output)


def get_readers():
    indices = read_indices_list()
    return {
        index_name: Freader(index_name, faiss_path=index_path)
        for index_name, index_path in indices.items()
    }


def add_reader(readers, index_name):
    index_path = f"data/{index_name}"
    reader = Freader(index_name, index_path)
    readers[index_name] = reader
