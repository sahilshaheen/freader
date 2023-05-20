# **FRIDA**: **F**lexible **R**etriever for **I**nformation **D**iscovery and **A**ssistance

A simple FastAPI server that leverages langchain to index information and query the resulting indices in many ways.

## Roadmap
- [ ] Flexibility in input
  - [x] URLs
  - [x] Raw Text
  - [ ] YouTube (Medium priority)
- [ ] Configure separate reader for SIFT (High priority)
  - No splitting of text
  - No QA required 
- [ ] Topic modelling with input (Medium priority)
  - [x] Implement taggers
  - [ ] Incorporate tags into metadata while indexing
- [ ] FAISS -> FAISS + DB (Low priority)
- [ ] Search filters (Low priority)

## (Not So Obvious) Configuration

Freaders corresponding to indices created at runtime will be initialized with the default values. This behaviour can be overwritten for indices by populating the configuration for an index in `indices.json`:

```json
{
 "default": {
    "index_name": "default",
    "faiss_path": "data/default"
  },
  "code": {
    "index_name": "code",
    "faiss_path": "data/code",
    "model_id": "openai",
    "retrieval_only": true,
    "chunk_size": 1000,
    "chunk_overlap": 0
  }
}
```

Now when the "code" index is loaded/created at runtime, it takes the specified initialization values.
