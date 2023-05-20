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
