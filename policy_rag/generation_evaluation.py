"""Run repeatable local RAG generation experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Callable
from uuid import uuid4

from .evaluation import load_questions, section_matches
from .experiment_manifest import (
    collect_git_info,
    collect_index_identity,
    collect_ollama_identity,
    collect_python_environment,
    file_sha256,
)


DEFAULT_QUESTION_PATH = Path(
    "data/evaluation/questions.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/generation_evaluation"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FINAL_QUESTION_PATH = Path(
    "data/evaluation/final_questions.json"
)
FROZEN_FINAL_QUESTION_SHA256 = (
    "ad8a474a0f5c19f4acba9d77dafeb903"
    "57cb208e2f5fe44463b4b34f60f4bd28"
)
FINAL_SEMANTIC_INDEX_PATH = Path(
    "artifacts/semantic_index"
)
MODELS = (
    "qwen2.5:1.5b",
    "llama3.2:3b",
)
FINAL_MODELS = MODELS


def _rate(values: list[bool]) -> float:
    if not values:
        return 0.0

    return round(sum(values) / len(values), 4)


def _is_generation_failure(result: dict) -> bool:
    """Return whether generation failed after retrieval completed."""

    actual_status = result.get(
        "actual_status",
        result.get("status", ""),
    )

    return actual_status != "error" and (
        bool(result.get("generation_error"))
        or result.get("response_mode") == "generation_failure"
    )


def _evaluate_question(
    question: dict,
    model: str,
    top_k: int,
    index_dir: str | Path | None = None,
) -> dict:
    from .rag import answer_question

    expected_status = (
        "supported"
        if question["answerable"]
        else "unsupported"
    )

    started = perf_counter()

    try:
        answer_arguments = {
            "model": model,
            "top_k": top_k,
        }

        if index_dir is not None:
            answer_arguments["index_dir"] = index_dir

        result = answer_question(
            question["question"],
            **answer_arguments,
        )
    except Exception as exc:
        return {
            "question_id": question["question_id"],
            "policy_id": question["policy_id"],
            "question": question["question"],
            "expected_answer": question["expected_answer"],
            "evidence_section": question["evidence_section"],
            "question_type": question["question_type"],
            "answerable": question["answerable"],
            "requested_model": model,
            "actual_model": None,
            "expected_status": expected_status,
            "actual_status": "error",
            "status_correct": False,
            "source_policy_correct": False,
            "source_section_correct": False,
            "answer": "",
            "response_mode": "error",
            "generation_error": "",
            "error": "",
            "model_answer": "",
            "model_evidence": "",
            "grounding_coverage": 0.0,
            "sources": [],
            "generation_metrics": {},
            "prompt_eval_duration_ns": None,
            "end_to_end_seconds": round(
                perf_counter() - started,
                4,
            ),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    elapsed = perf_counter() - started
    sources = result.get("sources", [])
    response_mode = result.get(
        "response_mode",
        "unknown",
    )
    generation_error = result.get(
        "generation_error",
        "",
    )
    generation_failed = _is_generation_failure(result)

    policy_correct = any(
        str(source.get("policy", "")).upper()
        == question["policy_id"].upper()
        for source in sources
    )

    expected_section = question[
        "evidence_section"
    ].strip()

    section_correct = any(
        str(source.get("policy", "")).upper()
        == question["policy_id"].upper()
        and section_matches(
            expected_section,
            str(source.get("section", "")),
        )
        for source in sources
    )

    if not question["answerable"]:
        policy_correct = False
        section_correct = False

    generation_metrics = result.get(
        "generation_metrics",
        {},
    )
    actual_model = result.get("actual_model")
    model_identity_matches = (
        isinstance(actual_model, str)
        and bool(actual_model)
        and actual_model == model
    )
    actual_status = result.get("status", "")
    evaluated_response_mode = response_mode
    evaluation_error = ""

    if not model_identity_matches:
        actual_status = "error"
        evaluated_response_mode = "error"
        evaluation_error = "model_identity_mismatch"
        policy_correct = False
        section_correct = False

    return {
        "question_id": question["question_id"],
        "policy_id": question["policy_id"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "evidence_section": expected_section,
        "question_type": question["question_type"],
        "answerable": question["answerable"],
        "requested_model": model,
        "actual_model": actual_model,
        "expected_status": expected_status,
        "actual_status": actual_status,
        "status_correct": (
            model_identity_matches
            and not generation_failed
            and actual_status == expected_status
        ),
        "source_policy_correct": policy_correct,
        "source_section_correct": section_correct,
        "answer": result.get("answer", ""),
        "response_mode": evaluated_response_mode,
        "generation_error": generation_error,
        "error": evaluation_error,
        "model_answer": result.get(
            "model_answer",
            "",
        ),
        "model_evidence": result.get(
            "model_evidence",
            "",
        ),
        "evidence_valid": result.get(
            "evidence_valid",
            False,
        ),
        "answer_grounded": result.get(
            "answer_grounded",
            False,
        ),
        "grounding_coverage": result.get(
            "grounding_coverage",
            0.0,
        ),
        "sources": sources,
        "generation_metrics": generation_metrics,
        "prompt_eval_duration_ns": (
            generation_metrics.get(
                "prompt_eval_duration_ns"
            )
        ),
        "end_to_end_seconds": round(elapsed, 4),
        "error_type": "",
        "error_message": "",
    }


def _summarise_results(
    model: str,
    results: list[dict],
) -> dict:
    answerable = [
        result
        for result in results
        if result["answerable"]
    ]
    unanswerable = [
        result
        for result in results
        if not result["answerable"]
    ]
    completed = [
        result
        for result in results
        if result["actual_status"] != "error"
    ]

    durations = [
        result["end_to_end_seconds"]
        for result in completed
    ]
    generation_failures = [
        _is_generation_failure(result)
        for result in results
    ]

    return {
        "model": model,
        "requested_model": model,
        "question_count": len(results),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "error_count": sum(
            result["actual_status"] == "error"
            for result in results
        ),
        "generation_failure_count": sum(
            generation_failures
        ),
        "generation_failure_rate": _rate(
            generation_failures
        ),
        "status_accuracy": _rate(
            [
                result["status_correct"]
                and not _is_generation_failure(result)
                for result in results
            ]
        ),
        "answerable_support_rate": _rate(
            [
                result["actual_status"] == "supported"
                for result in answerable
            ]
        ),
        "unanswerable_refusal_rate": _rate(
            [
                result["actual_status"] == "unsupported"
                and result.get("response_mode") == "refusal"
                and not _is_generation_failure(result)
                for result in unanswerable
            ]
        ),
        "source_policy_accuracy": _rate(
            [
                result["source_policy_correct"]
                for result in answerable
            ]
        ),
        "source_section_accuracy": _rate(
            [
                result["source_section_correct"]
                for result in answerable
            ]
        ),
        "extractive_fallback_rate": _rate(
            [
                result["response_mode"]
                == "extractive_fallback"
                for result in completed
            ]
        ),
        "mean_end_to_end_seconds": round(
            mean(durations) if durations else 0.0,
            4,
        ),
    }


def _evaluate_models(
    questions: list[dict],
    models: tuple[str, ...],
    top_k: int,
    index_dir: str | Path | None = None,
) -> list[dict]:
    model_reports = []

    for model in models:
        print(f"Evaluating model: {model}")
        results = []

        for position, question in enumerate(
            questions,
            start=1,
        ):
            print(
                f"  [{position}/{len(questions)}] "
                f"{question['question_id']}"
            )

            results.append(
                _evaluate_question(
                    question,
                    model,
                    top_k,
                    index_dir=index_dir,
                )
            )

        model_reports.append(
            {
                "model": model,
                "requested_model": model,
                "summary": _summarise_results(
                    model,
                    results,
                ),
                "results": results,
            }
        )

    return model_reports


def _new_run_identity() -> tuple[datetime, str]:
    return datetime.now(timezone.utc), uuid4().hex


def _report_filename(
    prefix: str,
    created_at: datetime,
    run_id: str,
) -> str:
    timestamp = created_at.strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    return f"{prefix}_{timestamp}_{run_id}.json"


def _write_json_exclusive(
    result_path: str | Path,
    report: dict,
) -> None:
    _build_and_write_json_exclusive(
        result_path,
        lambda: report,
    )


def _build_and_write_json_exclusive(
    result_path: str | Path,
    report_builder: Callable[[], dict],
) -> dict:
    """Reserve a report path before running its builder."""

    path = Path(result_path)
    handle = path.open(
        "x",
        encoding="utf-8",
    )

    try:
        report = report_builder()
        serialised = json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        handle.write(serialised)
        handle.close()
    except BaseException:
        try:
            handle.close()
        except BaseException:
            pass

        try:
            path.unlink()
        except OSError:
            pass

        raise

    return report


def _resolve_repository_path(
    path: str | Path,
) -> Path:
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
            "Experiment path must be inside the repository"
        ) from exc

    return relative.as_posix()


def run_generation_evaluation(
    question_path: str | Path = DEFAULT_QUESTION_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    models: tuple[str, ...] = MODELS,
    top_k: int = 1,
    limit: int | None = None,
    index_dir: str | Path | None = None,
) -> dict:
    """Run a general development generation evaluation."""

    questions = load_questions(question_path)

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        questions = questions[:limit]

    requested_models = tuple(models)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_at, run_id = _new_run_identity()
    filename = _report_filename(
        "generation_evaluation",
        created_at,
        run_id,
    )
    result_path = output_path / filename

    def build_report() -> dict:
        model_reports = _evaluate_models(
            questions,
            requested_models,
            top_k,
            index_dir=index_dir,
        )

        return {
            "run_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "run_kind": "development_generation",
            "question_path": str(question_path),
            "top_k": top_k,
            "requested_models": list(requested_models),
            "models": model_reports,
            "result_path": str(result_path),
        }

    report = _build_and_write_json_exclusive(
        result_path,
        build_report,
    )

    return report


def run_final_generation_evaluation(
    question_path: str | Path = (
        FROZEN_FINAL_QUESTION_PATH
    ),
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    models: tuple[str, ...] = FINAL_MODELS,
    top_k: int = 1,
    index_dir: str | Path = (
        FINAL_SEMANTIC_INDEX_PATH
    ),
) -> dict:
    """Run the guarded, frozen final generation evaluation."""

    requested_models = tuple(models)
    resolved_question_path = _resolve_repository_path(
        question_path
    )
    expected_question_path = _resolve_repository_path(
        FROZEN_FINAL_QUESTION_PATH
    )

    if resolved_question_path != expected_question_path:
        raise ValueError(
            "Final evaluation requires the frozen question path"
        )

    if not resolved_question_path.is_file():
        raise ValueError(
            "Frozen final question file was not found"
        )

    question_sha256 = file_sha256(
        resolved_question_path
    )

    if question_sha256 != FROZEN_FINAL_QUESTION_SHA256:
        raise ValueError(
            "Frozen final question SHA-256 does not match"
        )

    if type(top_k) is not int or top_k != 1:
        raise ValueError(
            "Final generation evaluation requires top_k=1"
        )

    if requested_models != FINAL_MODELS:
        raise ValueError(
            "Final generation evaluation requires both frozen models"
        )

    resolved_index_path = _resolve_repository_path(
        index_dir
    )
    expected_index_path = _resolve_repository_path(
        FINAL_SEMANTIC_INDEX_PATH
    )

    if resolved_index_path != expected_index_path:
        raise ValueError(
            "Final generation evaluation requires the semantic index"
        )

    git_info = collect_git_info(REPOSITORY_ROOT)

    if git_info.get("clean") is not True:
        raise RuntimeError(
            "Final generation evaluation requires a clean worktree"
        )

    if not git_info.get("commit"):
        raise RuntimeError(
            "Final generation evaluation requires a Git commit"
        )

    index_identity = collect_index_identity(
        resolved_index_path,
        REPOSITORY_ROOT,
    )
    required_index_files = {
        "metadata.json",
        "chunks.json",
        "embeddings.npy",
    }

    index_metadata = index_identity.get("metadata")

    if (
        index_identity.get("errors")
        or not isinstance(index_metadata, dict)
        or index_metadata.get("model_name")
        != "sentence-transformers/all-MiniLM-L6-v2"
        or not required_index_files.issubset(
            index_identity.get("files", {})
        )
        or any(
            value is None
            for value in index_identity.get(
                "files",
                {},
            ).values()
        )
    ):
        raise RuntimeError(
            "Semantic index identity could not be verified"
        )

    environment = collect_python_environment()
    ollama_identity = collect_ollama_identity()
    identified_models = {
        model.get("name")
        for model in ollama_identity.get("models", [])
        if isinstance(model, dict)
        and isinstance(model.get("name"), str)
        and isinstance(model.get("id"), str)
        and bool(model["id"].strip())
        and isinstance(model.get("size"), str)
        and bool(model["size"].strip())
    }

    if (
        ollama_identity.get("errors")
        or not isinstance(
            ollama_identity.get("version"),
            str,
        )
        or not ollama_identity["version"].strip()
        or not set(requested_models).issubset(
            identified_models
        )
    ):
        raise RuntimeError(
            "Ollama model identity could not be verified"
        )

    questions = load_questions(resolved_question_path)
    output_path = _resolve_repository_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_at, run_id = _new_run_identity()
    filename = _report_filename(
        "final_generation",
        created_at,
        run_id,
    )
    result_path = output_path / filename

    def build_report() -> dict:
        model_reports = _evaluate_models(
            questions,
            requested_models,
            top_k,
            index_dir=resolved_index_path,
        )

        return {
            "run_id": run_id,
            "created_at_utc": created_at.isoformat(),
            "run_kind": "final_generation",
            "question_path": _repository_relative_path(
                resolved_question_path
            ),
            "question_sha256": question_sha256,
            "git": git_info,
            "top_k": top_k,
            "requested_models": list(requested_models),
            "semantic_index": index_identity,
            "environment": environment,
            "ollama": ollama_identity,
            "models": model_reports,
            "result_path": str(result_path),
        }

    report = _build_and_write_json_exclusive(
        result_path,
        build_report,
    )

    return report
