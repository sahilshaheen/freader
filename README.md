# **FRIDA**: **F**lexible **R**etriever for **I**nformation **D**iscovery and **A**ssistance

A simple FastAPI server that leverages langchain to index information and query the resulting indices using LLMs.

Roadmap:
- FAISS -> FAISS + DB
- Flexibility in input: URLs -> URLs, raw text, etc. (support as many document loaders from lanchain as necessary)
- Topic modelling with input
- Search filters
