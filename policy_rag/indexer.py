"""Build and query a transparent TF-IDF policy retrieval baseline."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk


CHUNKS_FILE = "chunks.json"
VECTORIZER_FILE = "vectorizer.joblib"
MATRIX_FILE = "matrix.joblib"
METADATA_FILE = "metadata.json"


def build_index(chunks: list[Chunk], index_dir: str | Path, strategy: str) -> dict:
    """Create and save a local TF-IDF index."""

    if not chunks:
        raise ValueError("Cannot build an index without chunks")

    output = Path(index_dir)
    output.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=1,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(chunk.text for chunk in chunks)

    (output / CHUNKS_FILE).write_text(
        json.dumps([asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    joblib.dump(vectorizer, output / VECTORIZER_FILE)
    joblib.dump(matrix, output / MATRIX_FILE)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "document_count": len({chunk.document_id for chunk in chunks}),
        "chunk_count": len(chunks),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }
    (output / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def _load_chunks(path: Path) -> list[Chunk]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


def search_index(
    query: str,
    index_dir: str | Path,
    top_k: int = 3,
) -> list[tuple[float, Chunk]]:
    """Return the highest-scoring chunks for a natural-language query."""

    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    directory = Path(index_dir)
    required = [CHUNKS_FILE, VECTORIZER_FILE, MATRIX_FILE]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Index is incomplete in {directory}. Missing: {', '.join(missing)}. "
            "Build the index first."
        )

    chunks = _load_chunks(directory / CHUNKS_FILE)
    vectorizer = joblib.load(directory / VECTORIZER_FILE)
    matrix = joblib.load(directory / MATRIX_FILE)

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1][: min(top_k, len(chunks))]
    return [(float(scores[index]), chunks[int(index)]) for index in ranked_indices]

