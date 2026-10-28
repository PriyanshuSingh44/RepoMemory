"""
RepoMemory Tools: Retrieval and memory management for coding agents.
Provides search_code, recall_decisions, and save_note with Ollama-compatible tool schemas.
"""

import os
import datetime
from typing import List, Dict, Any, Optional

try:
    import chromadb
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class RepoMemoryTools:
    """Manages retrieval and note storage over indexed repository memory."""

    def __init__(
        self,
        repo_path: str = ".",
        db_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.db_path = db_path or os.path.join(self.repo_path, ".repomemory_db")
        self.model_name = model_name
        self.embedder = None
        self.chroma_client = None
        self.code_col = None
        self.knowledge_col = None

    def _init_models(self):
        """Lazy load embedding model and Chroma collections."""
        if self.embedder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is required for RepoMemoryTools.")
            self.embedder = SentenceTransformer(self.model_name)

        if self.chroma_client is None:
            if chromadb is None:
                raise ImportError("chromadb is required for RepoMemoryTools.")
            if not os.path.exists(self.db_path):
                raise FileNotFoundError(
                    f"RepoMemory database not found at {self.db_path}. "
                    f"Please run indexer.py first."
                )
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            try:
                self.code_col = self.chroma_client.get_collection(name="repo_code")
            except Exception:
                self.code_col = self.chroma_client.create_collection(
                    name="repo_code", metadata={"hnsw:space": "cosine"}
                )
            try:
                self.knowledge_col = self.chroma_client.get_collection(name="team_knowledge")
            except Exception:
                self.knowledge_col = self.chroma_client.create_collection(
                    name="team_knowledge", metadata={"hnsw:space": "cosine"}
                )

    def search_code(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search indexed codebase for code chunks relevant to query.
        Returns top-k chunks with file path, line range, and code text.
        """
        self._init_models()
        query_emb = self.embedder.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        count = self.code_col.count()
        if count == 0:
            return []

        n_results = min(k, count)
        res = self.code_col.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                results.append({
                    "file_path": meta.get("file_path", "unknown"),
                    "start_line": meta.get("start_line", 0),
                    "end_line": meta.get("end_line", 0),
                    "chunk_type": meta.get("chunk_type", "code"),
                    "symbol_name": meta.get("symbol_name", ""),
                    "score": round(1.0 - float(dist), 4),
                    "text": doc
                })
        return results

    def recall_decisions(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Search indexed commit history and decision notes for rationale and context.
        """
        self._init_models()
        query_emb = self.embedder.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        count = self.knowledge_col.count()
        if count == 0:
            return []

        n_results = min(k, count)
        res = self.knowledge_col.query(
            query_embeddings=query_emb,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        results = []
        if res and res.get("documents") and res["documents"][0]:
            docs = res["documents"][0]
            metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(docs)
            dists = res["distances"][0] if res.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                results.append({
                    "source_type": meta.get("source_type", "doc"),
                    "file_name": meta.get("file_name", ""),
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "date": meta.get("date", ""),
                    "subject": meta.get("subject", ""),
                    "score": round(1.0 - float(dist), 4),
                    "text": doc
                })
        return results

    def save_note(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        """
        Append a new architectural or design decision note to DECISIONS.md and index it immediately.
        """
        self._init_models()
        note_title = title or f"Decision Note ({datetime.date.today().isoformat()})"
        notes_file = os.path.join(self.repo_path, "DECISIONS.md")

        entry = f"\n\n## {note_title}\n{text.strip()}\n"
        with open(notes_file, "a", encoding="utf-8") as f:
            f.write(entry)

        # Index directly into Chroma
        note_text = f"# {note_title}\n{text.strip()}"
        emb = self.embedder.encode(
            [note_text], show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        note_id = f"note:{int(datetime.datetime.now().timestamp())}"
        self.knowledge_col.add(
            ids=[note_id],
            documents=[note_text],
            embeddings=emb,
            metadatas=[{
                "source_type": "doc",
                "file_name": "DECISIONS.md",
                "title": note_title,
                "start_line": 0,
                "end_line": 0
            }]
        )

        return {
            "status": "success",
            "message": f"Saved note '{note_title}' to DECISIONS.md and indexed to Chroma.",
            "note_id": note_id
        }

    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """Format Python tool result as clean, token-efficient text for LLM message history."""
        if tool_name == "search_code":
            if not result:
                return "No matching code chunks found."
            chunks_str = []
            for idx, r in enumerate(result, 1):
                sym = f" ({r['chunk_type']} {r['symbol_name']})" if r.get("symbol_name") else ""
                header = f"--- [Result {idx}] {r['file_path']}:{r['start_line']}-{r['end_line']}{sym} (relevance: {r['score']}) ---"
                chunks_str.append(f"{header}\n{r['text']}")
            return "\n\n".join(chunks_str)

        elif tool_name == "recall_decisions":
            if not result:
                return "No matching decisions or commit notes found."
            notes_str = []
            for idx, r in enumerate(result, 1):
                source = r.get("source_type", "doc")
                if source == "commit":
                    hdr = f"--- [Commit] {r.get('subject')} by {r.get('author')} ({r.get('date')}) ---"
                else:
                    hdr = f"--- [Decision Note] {r.get('file_name')} - {r.get('title')} ---"
                notes_str.append(f"{hdr}\n{r['text']}")
            return "\n\n".join(notes_str)

        elif tool_name == "save_note":
            return result.get("message", "Note saved.")

        return str(result)

    def execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Execute a tool by name and return string representation for LLM."""
        if name == "search_code":
            query = args.get("query", "")
            k = int(args.get("k", 5))
            res = self.search_code(query, k=k)
            return self.format_tool_result(name, res)
        elif name == "recall_decisions":
            query = args.get("query", "")
            k = int(args.get("k", 3))
            res = self.recall_decisions(query, k=k)
            return self.format_tool_result(name, res)
        elif name == "save_note":
            text = args.get("text", "")
            title = args.get("title")
            res = self.save_note(text, title=title)
            return self.format_tool_result(name, res)
        else:
            return f"Unknown tool: {name}"


# Ollama-compatible JSON Tool Schemas
OLLAMA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Semantic code search across the indexed repository. Retrieves top code chunks with file paths and line ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query describing code functionality, function names, classes, or patterns (e.g. 'ast parsing function' or 'how is chunking handled?')"
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to retrieve (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_decisions",
            "description": "Search commit messages and architectural decision notes (DECISIONS.md) to understand why things were built or how design evolved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query about past decisions, architecture rationale, or commit history."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of entries to retrieve (default: 3)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Record a new architecture decision or note to DECISIONS.md and immediately index it into memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The decision content, trade-offs, or note to record."
                    },
                    "title": {
                        "type": "string",
                        "description": "Short title for the note."
                    }
                },
                "required": ["text"]
            }
        }
    }
]


if __name__ == "__main__":
    import sys
    print("Testing RepoMemoryTools schemas...")
    print(f"Loaded {len(OLLAMA_TOOL_SCHEMAS)} tool definitions.")
    for tool in OLLAMA_TOOL_SCHEMAS:
        fn = tool["function"]
        print(f" - {fn['name']}: {fn['description'][:60]}...")
    print("RepoMemoryTools module ready.")
