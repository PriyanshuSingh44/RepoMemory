"""Expose RepoMemory as tools to MCP-compatible coding agents."""

import argparse
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from indexer import RepoIndexer
from tools import RepoMemoryTools


def create_server(repo_path: str, db_path: Optional[str] = None) -> FastMCP:
    """Create an MCP server scoped to one target repository."""
    repo_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_path):
        raise NotADirectoryError(f"Repository directory does not exist: {repo_path}")

    memory = RepoMemoryTools(repo_path=repo_path, db_path=db_path)
    indexer = RepoIndexer(repo_path=repo_path, db_path=db_path)
    server = FastMCP("RepoMemory")

    @server.tool()
    def index_repository() -> Dict[str, int]:
        """Create or incrementally update the local index for the configured repository.

        Call this before the first search and after repository files have changed.
        """
        return indexer.index_repo()

    @server.tool()
    def search_code(query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Find semantically relevant code chunks with file paths and line numbers."""
        return memory.search_code(query, k=k)

    @server.tool()
    def recall_decisions(query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Find relevant project decision notes and indexed Git commit messages."""
        return memory.recall_decisions(query, k=k)

    @server.tool()
    def save_note(text: str, title: Optional[str] = None) -> Dict[str, str]:
        """Append a project decision note to the target repository's DECISIONS.md.

        This writes to the configured repository; use only for durable project
        rationale that should be recorded in that file.
        """
        return memory.save_note(text, title=title)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RepoMemory as an MCP server")
    parser.add_argument(
        "--repo",
        required=True,
        help="Target repository directory to expose to the connected agent",
    )
    parser.add_argument("--db", default=None, help="ChromaDB directory (defaults to <repo>/.repomemory_db)")
    args = parser.parse_args()

    server = create_server(args.repo, db_path=args.db)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
