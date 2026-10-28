# Architectural Decisions and Notes

## Vector Store Selection
We chose ChromaDB with local persistent storage for the following reasons:
1. Pure local execution: zero cloud dependencies, runs completely offline on developer laptops.
2. Built-in HNSW indexing with cosine distance metric for sub-millisecond retrieval.
3. Metadata filtering: supports filtering by file_path, source_type, and symbol tags directly during queries.
4. Lightweight footprint compared to deploying external Milvus or Qdrant containers.

## Embedding Model Selection
We selected `sentence-transformers/all-MiniLM-L6-v2` as the default local embedding model because:
1. Fast CPU inference: 384-dimensional embeddings compute in milliseconds on standard laptops without requiring a dedicated GPU.
2. Compact size: ~80MB weight footprint, fast download and low memory usage.
3. Strong semantic search capability for both source code identifiers and natural language developer queries.

## Chunking Strategy: AST First, Fallback to Windows
For Python code, chunking relies on Python's standard `ast` module to split top-level functions and classes cleanly at syntactic boundaries.
For markdown and documentation files, chunking splits on Markdown headings (`#`, `##`).
For all other file formats (or when AST syntax parsing fails), a sliding line window (50 lines, 10 overlap) ensures continuous coverage without dropping tokens.

## Tool Calling Protocol: Plain Python Functions & Ollama Schemas
We avoided heavy orchestration frameworks like LangChain in favor of plain Python functions exposed directly through standard JSON tool schemas.
This matches the local Ollama `/api/chat` function-calling protocol seamlessly (especially `qwen2.5-coder:7b-instruct`) and keeps dependencies minimal.

## Comparison Mode: Side-by-Side Output with vs. without RepoMemory
We introduced `--mode compare` (set as the default execution mode in `agent.py`) to provide an instant, empirical view of the difference between using RepoMemory and running without it.
1. Output WITH our project: Displays structured code snippets with exact line ranges, symbol identifiers, and retrieved architectural context from `DECISIONS.md`.
2. Output WITHOUT our project: Shows the unindexed monolithic file dump experience, illustrating context window bloat, missing AST boundaries, and the absence of team memory.
3. Head-to-Head Comparison Table: Reports prompt token reduction, total token reduction, latency, and context precision directly in the terminal.

