"""Command-line interface for the Stage 1 policy retrieval baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chunking import chunk_documents
from .indexer import build_index, search_index
from .loaders import load_directory


DEFAULT_DATA_DIR = Path("data/raw")
DEFAULT_INDEX_DIR = Path("artifacts/tfidf_index")


def _build_command(args: argparse.Namespace) -> int:
    documents = load_directory(args.data_dir)
    chunks = chunk_documents(
        documents,
        strategy=args.strategy,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )
    metadata = build_index(chunks, args.index_dir, strategy=args.strategy)
    print("Index built successfully")
    print(json.dumps(metadata, indent=2))
    return 0


def _search_command(args: argparse.Namespace) -> int:
    results = search_index(args.query, args.index_dir, top_k=args.top_k)
    print(f"Query: {args.query}\n")
    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f}")
        print(f"Document: {chunk.document_title}")
        print(f"Section: {chunk.section}")
        print(f"Source: {chunk.source_path}")
        print(f"Text: {chunk.text}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local university policy retrieval baseline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Load policies and build an index")
    build.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    build.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    build.add_argument("--strategy", choices=["policy", "fixed"], default="policy")
    build.add_argument("--max-words", type=int, default=180)
    build.add_argument("--overlap-words", type=int, default=30)
    build.set_defaults(handler=_build_command)

    search = subparsers.add_parser("search", help="Search an existing index")
    search.add_argument("query")
    search.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    search.add_argument("--top-k", type=int, default=3)
    search.set_defaults(handler=_search_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())

