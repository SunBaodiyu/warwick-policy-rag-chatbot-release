"""Load and validate the fixed retrieval evaluation questions."""

from __future__ import annotations

import json
import re
from pathlib import Path


REQUIRED_FIELDS = {
    "question_id",
    "policy_id",
    "question",
    "expected_answer",
    "evidence_section",
    "question_type",
    "answerable",
}

_SECTION_NUMBER_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)*)(?:\.)?"
    r"(?=\s|$|[:)\-–—])"
)


def _normalise_section(
    section: str,
) -> tuple[str, str] | None:
    """Return a comparable section kind and value."""

    if not isinstance(section, str):
        return None

    stripped = section.strip()

    if not stripped:
        return None

    number_match = _SECTION_NUMBER_PATTERN.match(
        stripped
    )

    if number_match:
        return "number", number_match.group(1)

    return "text", stripped.casefold()


def section_matches(
    expected_section: str,
    actual_section: str,
) -> bool:
    """Match complete section numbers or full text labels."""

    expected = _normalise_section(expected_section)
    actual = _normalise_section(actual_section)

    return (
        expected is not None
        and actual is not None
        and expected == actual
    )


def load_questions(path: str | Path) -> list[dict]:
    """Load evaluation questions from a JSON file."""

    question_path = Path(path)

    if not question_path.is_file():
        raise FileNotFoundError(
            f"Evaluation question file not found: {question_path}"
        )

    questions = json.loads(question_path.read_text(encoding="utf-8"))

    if not isinstance(questions, list) or not questions:
        raise ValueError("Evaluation file must contain a non-empty JSON list")

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index} must be a JSON object")

        missing_fields = REQUIRED_FIELDS - question.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Question {index} is missing fields: {missing}")

    return questions


def evaluate_retrieval(
    question_path: str | Path,
    index_dir: str | Path,
    top_k: int = 3,
    search_function=None,
) -> dict:
    """Evaluate retrieval using policy IDs and evidence sections."""

    if search_function is None:
        from .indexer import search_index

        search_function = search_index

    questions = [
        question
        for question in load_questions(question_path)
        if question["answerable"] is True
    ]

    if not questions:
        raise ValueError("No answerable questions were found")

    ranks: list[int | None] = []
    details: list[dict] = []

    for question in questions:
        retrieved = search_function(
            question["question"],
            index_dir,
            top_k=top_k,
        )

        correct_rank = None

        for rank, (_, chunk) in enumerate(retrieved, start=1):
            correct_policy = (
                question["policy_id"].upper()
                in chunk.document_id.upper()
            )
            correct_section = section_matches(
                question["evidence_section"],
                chunk.section,
            )

            if correct_policy and correct_section:
                correct_rank = rank
                break

        ranks.append(correct_rank)
        details.append(
            {
                "question_id": question["question_id"],
                "correct_rank": correct_rank,
            }
        )

    question_count = len(questions)

    return {
        "question_count": question_count,
        "hit_at_1": sum(rank == 1 for rank in ranks) / question_count,
        f"hit_at_{top_k}": sum(rank is not None for rank in ranks)
        / question_count,
        "mrr": sum(
            1 / rank if rank is not None else 0
            for rank in ranks
        )
        / question_count,
        "details": details,
    }
