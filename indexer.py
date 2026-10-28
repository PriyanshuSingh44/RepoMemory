import os
import ast
import json
import hashlib
import argparse
import subprocess
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "build", "dist",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache", ".eggs", "site-packages",
    "env", ".repomemory_db", ".chroma"
}

DEFAULT_IGNORE_EXTS = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pyc", ".pyd", ".obj", ".o",
    ".a", ".lib", ".iso", ".tar", ".gz", ".zip", ".7z", ".rar", ".png",
    ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".bmp", ".mp4",
    ".mp3", ".wav", ".pdf", ".docx", ".xlsx", ".parquet", ".db", ".sqlite",
    ".sqlite3", ".csv", ".json"
}

DEFAULT_IGNORE_FILES = {
    "package-lock.json", "poetry.lock", "yarn.lock", "Cargo.lock",
    "Pipfile.lock", "pnpm-lock.yaml", "repomemory_manifest.json", "results.csv"
}


def is_binary_file(file_path: str) -> bool:
    """Check if file is binary by looking for null bytes in initial chunk."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except Exception:
        return True


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def chunk_python_ast(content: str, rel_path: str) -> List[Dict[str, Any]]:
    """
    Chunk Python code by top-level functions and classes using ast.
    Falls back to line windowing if parsing fails.
    """
    lines = content.splitlines()
    if not lines:
        return []

    try:
        tree = ast.parse(content)
    except Exception:
        return chunk_by_line_window(content, rel_path)

    chunks = []
    handled_intervals = []

    # First collect top-level functions and classes
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = getattr(node, "end_lineno", start_line)
            chunk_text = "\n".join(lines[start_line - 1 : end_line]).strip()
            if chunk_text:
                symbol_type = "class" if isinstance(node, ast.ClassDef) else "function"
                chunks.append({
                    "text": chunk_text,
                    "file_path": rel_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "chunk_type": symbol_type,
                    "symbol_name": node.name
                })
                handled_intervals.append((start_line, end_line))

    # If module has code before first function/class (e.g. imports, global setup)
    if handled_intervals:
        first_start = handled_intervals[0][0]
        if first_start > 1:
            header_text = "\n".join(lines[: first_start - 1]).strip()
            if header_text:
                chunks.insert(0, {
                    "text": header_text,
                    "file_path": rel_path,
                    "start_line": 1,
                    "end_line": first_start - 1,
                    "chunk_type": "module_header",
                    "symbol_name": ""
                })
    else:
        # No functions or classes found; capture as lines or full module
        return chunk_by_line_window(content, rel_path)

    return chunks


def chunk_by_line_window(
    content: str, rel_path: str, window_size: int = 50, overlap: int = 10
) -> List[Dict[str, Any]]:
    """Fixed-size line window chunking with overlap for non-Python or fallback files."""
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    step = max(1, window_size - overlap)
    total_lines = len(lines)

    for i in range(0, total_lines, step):
        start_line = i + 1
        end_line = min(total_lines, i + window_size)
        window_lines = lines[i:end_line]
        chunk_text = "\n".join(window_lines).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "file_path": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_type": "lines",
                "symbol_name": ""
            })
        if end_line >= total_lines:
            break

    return chunks


def chunk_markdown(content: str, rel_path: str) -> List[Dict[str, Any]]:
    """Chunk markdown documents by headings (# or ##)."""
    lines = content.splitlines()
    if not lines:
        return []

    chunks = []
    current_lines = []
    current_heading = "Overview"
    start_line = 1

    for idx, line in enumerate(lines, start=1):
        if line.startswith("#"):
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "file_path": rel_path,
                        "start_line": start_line,
                        "end_line": idx - 1,
                        "chunk_type": "decision_note",
                        "symbol_name": current_heading
                    })
                current_lines = []
            current_heading = line.strip("# ").strip()
            start_line = idx
        current_lines.append(line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "file_path": rel_path,
                "start_line": start_line,
                "end_line": len(lines),
                "chunk_type": "decision_note",
                "symbol_name": current_heading
            })

    return chunks


class RepoIndexer:
    """Indexes code, commits, and decisions for vector retrieval."""

    def __init__(
        self,
        repo_path: str,
        db_path: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.db_path = db_path or os.path.join(self.repo_path, ".repomemory_db")
        self.model_name = model_name
        self.manifest_file = os.path.join(self.db_path, "repomemory_manifest.json")

        self.embedder = None
        self.chroma_client = None
        self.code_col = None
        self.knowledge_col = None

    def _init_models(self):
        """Lazy initialization of Chroma and SentenceTransformer."""
        if self.embedder is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed.")
            print(f"Loading embedding model: {self.model_name}...")
            self.embedder = SentenceTransformer(self.model_name)

        if self.chroma_client is None:
            if chromadb is None:
                raise ImportError("chromadb is not installed.")
            os.makedirs(self.db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.code_col = self.chroma_client.get_or_create_collection(
                name="repo_code",
                metadata={"hnsw:space": "cosine"}
            )
            self.knowledge_col = self.chroma_client.get_or_create_collection(
                name="team_knowledge",
                metadata={"hnsw:space": "cosine"}
            )

    def load_manifest(self) -> Dict[str, Any]:
        """Load incremental indexing manifest."""
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"files": {}, "indexed_commits": []}

    def save_manifest(self, manifest: Dict[str, Any]):
        """Save incremental indexing manifest."""
        os.makedirs(self.db_path, exist_ok=True)
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def walk_repo_files(self) -> List[str]:
        """Walk repo and return list of relative file paths, ignoring junk."""
        valid_files = []
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")
            ]

            for file in files:
                if file in DEFAULT_IGNORE_FILES or file.startswith("."):
                    continue
                ext = os.path.splitext(file)[1].lower()
                if ext in DEFAULT_IGNORE_EXTS:
                    continue

                abs_path = os.path.join(root, file)
                # Skip files larger than 500KB or binary files
                try:
                    if os.path.getsize(abs_path) > 500 * 1024:
                        continue
                except OSError:
                    continue

                if is_binary_file(abs_path):
                    continue

                rel_path = os.path.relpath(abs_path, self.repo_path).replace("\\", "/")
                valid_files.append(rel_path)

        return sorted(valid_files)

    def index_repo(self, force: bool = False) -> Dict[str, int]:
        """Index all valid files in the repo incrementally."""
        self._init_models()
        if force:
            try:
                self.chroma_client.delete_collection("repo_code")
            except Exception:
                pass
            try:
                self.chroma_client.delete_collection("team_knowledge")
            except Exception:
                pass
            self.code_col = self.chroma_client.get_or_create_collection(
                name="repo_code", metadata={"hnsw:space": "cosine"}
            )
            self.knowledge_col = self.chroma_client.get_or_create_collection(
                name="team_knowledge", metadata={"hnsw:space": "cosine"}
            )

        manifest = self.load_manifest() if not force else {"files": {}, "indexed_commits": []}
        all_files = self.walk_repo_files()

        old_file_hashes = manifest.get("files", {})
        new_file_hashes = {}
        files_to_index = []
        files_skipped = 0

        for rel_path in all_files:
            abs_path = os.path.join(self.repo_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            current_hash = compute_sha256(content)
            new_file_hashes[rel_path] = current_hash

            if not force and old_file_hashes.get(rel_path) == current_hash:
                files_skipped += 1
            else:
                files_to_index.append((rel_path, content, current_hash))

        # Check for deleted files
        deleted_files = set(old_file_hashes.keys()) - set(new_file_hashes.keys())
        for del_path in deleted_files:
            try:
                self.code_col.delete(where={"file_path": del_path})
            except Exception:
                pass

        total_chunks_added = 0

        # Process changed/new files
        for rel_path, content, _ in files_to_index:
            # Delete any existing chunks for this file
            try:
                self.code_col.delete(where={"file_path": rel_path})
            except Exception:
                pass

            if rel_path.endswith((".md", ".txt")):
                # Documentation belongs in team_knowledge, not code collection
                continue
            elif rel_path.endswith(".py"):
                chunks = chunk_python_ast(content, rel_path)
            else:
                chunks = chunk_by_line_window(content, rel_path)

            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ).tolist()

            ids = [
                f"{rel_path}:{c['start_line']}-{c['end_line']}"
                for c in chunks
            ]
            metadatas = [
                {
                    "file_path": c["file_path"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "chunk_type": c["chunk_type"],
                    "symbol_name": c["symbol_name"]
                }
                for c in chunks
            ]

            self.code_col.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_chunks_added += len(chunks)

        manifest["files"] = new_file_hashes
        self.save_manifest(manifest)

        # Index decisions / notes if present
        notes_indexed = self.index_notes_and_decisions()

        # Index git commits
        commits_indexed = self.index_git_commits(force=force)

        return {
            "files_scanned": len(all_files),
            "files_indexed": len(files_to_index),
            "files_skipped": files_skipped,
            "chunks_added": total_chunks_added,
            "notes_indexed": notes_indexed,
            "commits_indexed": commits_indexed
        }

    def index_notes_and_decisions(self) -> int:
        """Find DECISIONS.md, NOTES.md, or similar project docs and index them."""
        self._init_models()
        candidate_names = [
            "DECISIONS.md", "decisions.md", "NOTES.md", "notes.md",
            "ARCHITECTURE.md", "architecture.md"
        ]

        total_indexed = 0
        for name in candidate_names:
            abs_path = os.path.join(self.repo_path, name)
            if not os.path.isfile(abs_path):
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            # Delete old notes chunks for this file
            try:
                self.knowledge_col.delete(where={"source_type": "doc", "file_name": name})
            except Exception:
                pass

            chunks = chunk_markdown(content, name)
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.encode(
                texts, show_progress_bar=False, normalize_embeddings=True
            ).tolist()

            ids = [f"doc:{name}:{c['start_line']}-{c['end_line']}" for c in chunks]
            metadatas = [
                {
                    "source_type": "doc",
                    "file_name": name,
                    "title": c["symbol_name"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"]
                }
                for c in chunks
            ]

            self.knowledge_col.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_indexed += len(chunks)

        return total_indexed

    def index_git_commits(self, max_count: int = 100, force: bool = False) -> int:
        """Extract and index recent git commits into team_knowledge collection."""
        self._init_models()
        git_dir = os.path.join(self.repo_path, ".git")
        if not os.path.exists(git_dir):
            return 0

        manifest = self.load_manifest()
        already_indexed = set(manifest.get("indexed_commits", [])) if not force else set()

        try:
            cmd = [
                "git", "-C", self.repo_path, "log", f"-n{max_count}",
                '--pretty=format:@@@COMMIT@@@%n%H%n%an%n%ad%n%s%n%b'
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, encoding="utf-8", errors="ignore"
            )
        except Exception:
            return 0

        raw_commits = result.stdout.split("@@@COMMIT@@@\n")
        new_commits = []

        for raw in raw_commits:
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split("\n", 4)
            if len(parts) < 4:
                continue
            commit_hash = parts[0].strip()
            author = parts[1].strip()
            date = parts[2].strip()
            subject = parts[3].strip()
            body = parts[4].strip() if len(parts) > 4 else ""

            if commit_hash in already_indexed:
                continue

            doc_text = f"Commit {commit_hash[:8]}: {subject}\nAuthor: {author}\nDate: {date}"
            if body:
                doc_text += f"\n\n{body}"

            new_commits.append({
                "hash": commit_hash,
                "author": author,
                "date": date,
                "subject": subject,
                "text": doc_text
            })

        if not new_commits:
            return 0

        texts = [c["text"] for c in new_commits]
        embeddings = self.embedder.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        ).tolist()

        ids = [f"commit:{c['hash'][:12]}" for c in new_commits]
        metadatas = [
            {
                "source_type": "commit",
                "commit_hash": c["hash"],
                "author": c["author"],
                "date": c["date"],
                "subject": c["subject"]
            }
            for c in new_commits
        ]

        self.knowledge_col.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        all_indexed = list(already_indexed.union(c["hash"] for c in new_commits))
        manifest["indexed_commits"] = all_indexed[:500]
        self.save_manifest(manifest)

        return len(new_commits)


def main():
    parser = argparse.ArgumentParser(description="RepoMemory Indexer")
    parser.add_argument("--repo", default=".", help="Path to repo to index (default: current directory)")
    parser.add_argument("--db", default=None, help="Path to Chroma DB directory")
    parser.add_argument("--force", action="store_true", help="Force complete re-indexing")
    args = parser.parse_args()

    print(f"Indexing repository at: {os.path.abspath(args.repo)}")
    indexer = RepoIndexer(repo_path=args.repo, db_path=args.db)
    summary = indexer.index_repo(force=args.force)

    print("\n--- Indexing Summary ---")
    print(f"Files scanned:   {summary['files_scanned']}")
    print(f"Files indexed:   {summary['files_indexed']}")
    print(f"Files skipped:   {summary['files_skipped']} (incremental)")
    print(f"Code chunks:     {summary['chunks_added']}")
    print(f"Notes/Decisions: {summary['notes_indexed']}")
    print(f"Git commits:     {summary['commits_indexed']}")
    print("------------------------")
    print("Done!")


if __name__ == "__main__":
    main()
