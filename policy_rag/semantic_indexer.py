"""Build a local semantic index for policy chunks."""


from __future__ import annotations

from functools import lru_cache
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentence_transformers
from sentence_transformers import SentenceTransformer

from .models import Chunk


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"


def build_semantic_index(
    chunks: list[Chunk],
    index_dir: str | Path,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """Encode policy chunks and save a local semantic index."""

    if not chunks:
        raise ValueError("Cannot build a semantic index without chunks")

    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=True,
    )

    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    (output / CHUNKS_FILE).write_text(
        json.dumps(
            [asdict(chunk) for chunk in chunks],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    np.save(output / EMBEDDINGS_FILE, embeddings)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "sentence_transformers_version": getattr(
            sentence_transformers,
            "__version__",
            "unknown",
        ),
        "device": "cpu",
        "document_count": len(
            {chunk.document_id for chunk in chunks}
        ),
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "chunking_strategy": chunks[0].metadata.get(
            "strategy",
            "unknown",
        ),
    }

    (output / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata

def _load_chunks(path: Path) -> list[Chunk]:
    """Load saved chunks from JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]

@lru_cache(maxsize=2)
def _load_local_model(
    model_name: str,
) -> SentenceTransformer:
    """Load and reuse a locally cached embedding model."""

    return SentenceTransformer(
        model_name,
        device="cpu",
        local_files_only=True,
    )

def search_semantic_index(
    query: str,
    index_dir: str | Path,
    top_k: int = 3,
) -> list[tuple[float, Chunk]]:
    """Search the semantic index using cosine similarity."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    directory = Path(index_dir)
    required_files = [
        CHUNKS_FILE,
        EMBEDDINGS_FILE,
        METADATA_FILE,
    ]

    missing = [
        filename
        for filename in required_files
        if not (directory / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Semantic index is incomplete. Missing: "
            + ", ".join(missing)
        )

    chunks = _load_chunks(directory / CHUNKS_FILE)

    embeddings = np.load(
        directory / EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    metadata = json.loads(
        (directory / METADATA_FILE).read_text(encoding="utf-8")
    )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "The number of embeddings does not match the chunks"
        )

    model = _load_local_model(metadata["model_name"])

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    scores = embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][
        : min(top_k, len(chunks))
    ]

    return [
        (float(scores[index]), chunks[int(index)])
        for index in ranked_indices
    ]