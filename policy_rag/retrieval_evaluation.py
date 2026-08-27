"""Guarded, persistent evaluation of the two frozen retrievers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from numbers import Real
from pathlib import Path
from time import perf_counter
from typing import Callable

from .evaluation import load_questions, section_matches
from .experiment_manifest import (
    build_and_write_json_exclusive,
    collect_git_info,
    collect_index_identity,
    collect_python_environment,
    file_sha256,
    new_run_identity,
    report_filename,
)
from .models import Chunk


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FINAL_QUESTION_PATH = Path(
    "data/evaluation/final_questions.json"
)
FROZEN_FINAL_QUESTION_SHA256 = (
    "ad8a474a0f5c19f4acba9d77dafeb903"
    "57cb208e2f5fe44463b4b34f60f4bd28"
)
FINAL_TFIDF_INDEX_PATH = Path("artifacts/tfidf_index")
FINAL_MINILM_INDEX_PATH = Path("artifacts/semantic_index")
FINAL_MINILM_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/retrieval_evaluation")

TFIDF_REQUIRED_FILES = frozenset(
    {
        "chunks.json",
        "matrix.joblib",
        "metadata.json",
        "vectorizer.joblib",
    }
)
MINILM_REQUIRED_FILES = frozenset(
    {
        "chunks.json",
        "embeddings.npy",
        "metadata.json",
    }
)

SearchFunction = Callable[
    [str, str | Path, int],
    list[tuple[float, Chunk]],
]


def _resolve_repository_path(path: str | Path) -> Path:
    candidate = Path(path)

    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate

    return candidate.resolve()


def _repository_relative_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(
            REPOSITORY_ROOT.resolve()
        )
    except ValueError as exc:
        raise ValueError(
            "Final retrieval paths must be inside the repository"
        ) from exc

    return relative.as_posix()


def _require_frozen_path(
    supplied_path: str | Path,
    frozen_path: Path,
    label: str,
) -> Path:
    resolved = _resolve_repository_path(supplied_path)
    expected = _resolve_repository_path(frozen_path)

    if resolved != expected:
        raise ValueError(
            f"Final retrieval evaluation requires the frozen {label} path"
        )

    return resolved


def _validate_index_identity(
    identity: dict,
    required_files: frozenset[str],
    label: str,
) -> dict:
    metadata = identity.get("metadata")
    files = identity.get("files")

    if identity.get("errors"):
        raise RuntimeError(
            f"{label} index identity could not be verified"
        )

    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"{label} index metadata must be a JSON object"
        )

    if not isinstance(files, dict) or not required_files.issubset(
        files
    ):
        raise RuntimeError(
            f"{label} index is missing required files"
        )

    if any(
        not isinstance(files[name], str) or not files[name]
        for name in required_files
    ):
        raise RuntimeError(
            f"{label} index file hashes could not be verified"
        )

    return metadata


def _load_chunks(path: Path) -> list[dict]:
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Index chunks could not be loaded: {path.name}"
        ) from exc

    if not isinstance(chunks, list) or not chunks:
        raise RuntimeError(
            "Index chunks must be a non-empty JSON list"
        )

    chunk_ids: set[str] = set()

    for position, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise RuntimeError(
                f"Index chunk {position} must be a JSON object"
            )

        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise RuntimeError(
                f"Index chunk {position} has an invalid chunk_id"
            )

        if chunk_id in chunk_ids:
            raise RuntimeError(
                "Index chunks contain duplicate chunk IDs"
            )

        chunk_ids.add(chunk_id)

    return chunks


def _canonical_chunks(chunks: list[dict]) -> str:
    return json.dumps(
        chunks,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _policy_matches(
    expected_policy: str,
    document_id: str,
) -> bool:
    if not isinstance(expected_policy, str) or not isinstance(
        document_id,
        str,
    ):
        return False

    pattern = (
        r"(?<![A-Z0-9])"
        + re.escape(expected_policy.upper())
        + r"(?![A-Z0-9])"
    )
    return re.search(pattern, document_id.upper()) is not None


def _validate_complete_ranking(
    ranking: object,
    expected_chunk_ids: set[str],
) -> list[tuple[float, Chunk]]:
    if not isinstance(ranking, list):
        raise RuntimeError(
            "Search did not return a complete ranking list"
        )

    if len(ranking) != len(expected_chunk_ids):
        raise RuntimeError(
            "Search returned an incomplete ranking"
        )

    validated: list[tuple[float, Chunk]] = []
    returned_ids: set[str] = set()

    for entry in ranking:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
        ):
            raise RuntimeError(
                "Search returned an invalid ranking entry"
            )

        score, chunk = entry

        if (
            isinstance(score, bool)
            or not isinstance(score, Real)
            or not math.isfinite(float(score))
        ):
            raise RuntimeError(
                "Search returned an invalid ranking score"
            )

        if not isinstance(chunk, Chunk):
            raise RuntimeError(
                "Search returned an invalid chunk"
            )

        if not chunk.chunk_id or chunk.chunk_id in returned_ids:
            raise RuntimeError(
                "Search returned duplicate or invalid chunk IDs"
            )

        returned_ids.add(chunk.chunk_id)
        validated.append((float(score), chunk))

    if returned_ids != expected_chunk_ids:
        raise RuntimeError(
            "Search ranking does not contain the indexed chunk set"
        )

    return validated


def _top_three_candidates(
    ranking: list[tuple[float, Chunk]],
) -> list[dict]:
    return [
        {
            "rank": rank,
            "score": score,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "section": chunk.section,
            "document_title": chunk.document_title,
        }
        for rank, (score, chunk) in enumerate(
            ranking[:3],
            start=1,
        )
    ]


def _evaluate_full_ranking(
    questions: list[dict],
    *,
    retriever: str,
    index_dir: Path,
    expected_chunk_ids: set[str],
    search_function: SearchFunction,
) -> dict:
    answerable_questions = [
        question
        for question in questions
        if question["answerable"] is True
    ]

    if not answerable_questions:
        raise ValueError("No answerable questions were found")

    details: list[dict] = []
    ranks: list[int | None] = []
    search_times: list[float] = []
    ranking_depth = len(expected_chunk_ids)

    for question in answerable_questions:
        started = perf_counter()
        raw_ranking = search_function(
            question["question"],
            index_dir,
            top_k=ranking_depth,
        )
        elapsed = perf_counter() - started
        ranking = _validate_complete_ranking(
            raw_ranking,
            expected_chunk_ids,
        )
        correct_rank = None

        for rank, (_, chunk) in enumerate(
            ranking,
            start=1,
        ):
            if _policy_matches(
                question["policy_id"],
                chunk.document_id,
            ) and section_matches(
                question["evidence_section"],
                chunk.section,
            ):
                correct_rank = rank
                break

        ranks.append(correct_rank)
        search_times.append(elapsed)
        details.append(
            {
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "target_policy": question["policy_id"],
                "target_section": question["evidence_section"],
                "correct_rank": correct_rank,
                "search_seconds": elapsed,
                "top_3_candidates": _top_three_candidates(
                    ranking
                ),
            }
        )

    evaluated_count = len(answerable_questions)

    return {
        "retriever": retriever,
        "evaluated_question_count": evaluated_count,
        "full_ranking_chunk_count": ranking_depth,
        "hit_at_1": (
            sum(rank == 1 for rank in ranks)
            / evaluated_count
        ),
        "hit_at_3": (
            sum(
                rank is not None and rank <= 3
                for rank in ranks
            )
            / evaluated_count
        ),
        "mrr": (
            sum(
                1 / rank if rank is not None else 0
                for rank in ranks
            )
            / evaluated_count
        ),
        "not_found_count": sum(
            rank is None for rank in ranks
        ),
        "mean_search_seconds": (
            sum(search_times) / evaluated_count
        ),
        "details": details,
    }


def run_final_retrieval_evaluation(
    question_path: str | Path = FROZEN_FINAL_QUESTION_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    tfidf_index_dir: str | Path = FINAL_TFIDF_INDEX_PATH,
    minilm_index_dir: str | Path = FINAL_MINILM_INDEX_PATH,
    tfidf_search_function: SearchFunction | None = None,
    minilm_search_function: SearchFunction | None = None,
) -> dict:
    """Run and persist the guarded final TF-IDF/MiniLM evaluation."""

    resolved_question_path = _require_frozen_path(
        question_path,
        FROZEN_FINAL_QUESTION_PATH,
        "question",
    )

    if not resolved_question_path.is_file():
        raise ValueError("Frozen final question file was not found")

    question_sha256 = file_sha256(resolved_question_path)

    if question_sha256 != FROZEN_FINAL_QUESTION_SHA256:
        raise ValueError(
            "Frozen final question SHA-256 does not match"
        )

    git_info = collect_git_info(REPOSITORY_ROOT)

    if git_info.get("clean") is not True:
        raise RuntimeError(
            "Final retrieval evaluation requires a clean worktree"
        )

    if not git_info.get("commit"):
        raise RuntimeError(
            "Final retrieval evaluation requires a Git commit"
        )

    resolved_tfidf_path = _require_frozen_path(
        tfidf_index_dir,
        FINAL_TFIDF_INDEX_PATH,
        "TF-IDF index",
    )
    resolved_minilm_path = _require_frozen_path(
        minilm_index_dir,
        FINAL_MINILM_INDEX_PATH,
        "MiniLM index",
    )
    tfidf_identity = collect_index_identity(
        resolved_tfidf_path,
        REPOSITORY_ROOT,
    )
    minilm_identity = collect_index_identity(
        resolved_minilm_path,
        REPOSITORY_ROOT,
    )
    tfidf_metadata = _validate_index_identity(
        tfidf_identity,
        TFIDF_REQUIRED_FILES,
        "TF-IDF",
    )
    minilm_metadata = _validate_index_identity(
        minilm_identity,
        MINILM_REQUIRED_FILES,
        "MiniLM",
    )

    if minilm_metadata.get("model_name") != FINAL_MINILM_MODEL:
        raise RuntimeError(
            "MiniLM index uses the wrong embedding model"
        )

    tfidf_chunks = _load_chunks(
        resolved_tfidf_path / "chunks.json"
    )
    minilm_chunks = _load_chunks(
        resolved_minilm_path / "chunks.json"
    )
    canonical_tfidf = _canonical_chunks(tfidf_chunks)
    canonical_minilm = _canonical_chunks(minilm_chunks)

    if canonical_tfidf != canonical_minilm:
        raise RuntimeError(
            "TF-IDF and MiniLM indexes use different chunks"
        )

    chunk_count = len(minilm_chunks)
    for label, metadata in (
        ("TF-IDF", tfidf_metadata),
        ("MiniLM", minilm_metadata),
    ):
        metadata_chunk_count = metadata.get("chunk_count")

        if (
            type(metadata_chunk_count) is not int
            or metadata_chunk_count != chunk_count
        ):
            raise RuntimeError(
                f"{label} metadata chunk_count does not match chunks.json"
            )

    questions = load_questions(resolved_question_path)
    answerable_count = sum(
        question["answerable"] is True
        for question in questions
    )
    unanswerable_count = sum(
        question["answerable"] is False
        for question in questions
    )

    if answerable_count + unanswerable_count != len(questions):
        raise ValueError(
            "Final questions must use boolean answerable values"
        )

    if answerable_count == 0:
        raise ValueError("No answerable questions were found")

    environment = collect_python_environment()
    output_path = _resolve_repository_path(output_dir)
    expected_output_path = _resolve_repository_path(
        DEFAULT_OUTPUT_DIR
    )

    if output_path != expected_output_path:
        raise ValueError(
            "Final retrieval evaluation requires the frozen output path"
        )

    output_path.mkdir(parents=True, exist_ok=True)
    created_at, run_id = new_run_identity()
    result_path = output_path / report_filename(
        "final_retrieval",
        created_at,
        run_id,
    )
    expected_chunk_ids = {
        chunk["chunk_id"] for chunk in minilm_chunks
    }
    canonical_sha256 = hashlib.sha256(
        canonical_minilm.encode("utf-8")
    ).hexdigest()

    def build_report() -> dict:
        resolved_tfidf_search = tfidf_search_function
        resolved_minilm_search = minilm_search_function

        if resolved_tfidf_search is None:
            from .indexer import search_index

            resolved_tfidf_search = search_index

        if resolved_minilm_search is None:
            from .semantic_indexer import search_semantic_index

            resolved_minilm_search = search_semantic_index

        tfidf_result = _evaluate_full_ranking(
            questions,
            retriever="tfidf",
            index_dir=resolved_tfidf_path,
            expected_chunk_ids=expected_chunk_ids,
            search_function=resolved_tfidf_search,
        )
        minilm_result = _evaluate_full_ranking(
            questions,
            retriever="minilm",
            index_dir=resolved_minilm_path,
            expected_chunk_ids=expected_chunk_ids,
            search_function=resolved_minilm_search,
        )

        return {
            "run_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "run_kind": "final_retrieval",
            "question_path": _repository_relative_path(
                resolved_question_path
            ),
            "question_sha256": question_sha256,
            "git": git_info,
            "question_counts": {
                "total": len(questions),
                "answerable": answerable_count,
                "unanswerable": unanswerable_count,
            },
            "indexes": {
                "tfidf": tfidf_identity,
                "minilm": minilm_identity,
            },
            "environment": environment,
            "chunk_consistency": {
                "matches": True,
                "chunk_count": chunk_count,
                "canonical_sha256": canonical_sha256,
            },
            "results": {
                "tfidf": tfidf_result,
                "minilm": minilm_result,
            },
            "result_path": _repository_relative_path(
                result_path
            ),
        }

    return build_and_write_json_exclusive(
        result_path,
        build_report,
    )
