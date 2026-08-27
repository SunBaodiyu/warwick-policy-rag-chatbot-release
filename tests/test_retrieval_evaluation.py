"""Tests for the guarded final retrieval evaluation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import policy_rag.retrieval_evaluation as retrieval_evaluation
from policy_rag.models import Chunk
from policy_rag.retrieval_evaluation import (
    FINAL_MINILM_INDEX_PATH,
    FINAL_TFIDF_INDEX_PATH,
    FROZEN_FINAL_QUESTION_PATH,
    FROZEN_FINAL_QUESTION_SHA256,
    run_final_retrieval_evaluation,
)


class FinalRetrievalEvaluationTests(unittest.TestCase):
    @staticmethod
    def make_questions() -> list[dict]:
        return [
            {
                "question_id": "SYNTHETIC-Q01",
                "policy_id": "IMP02",
                "question": "Where is the first synthetic rule?",
                "expected_answer": "It is in the target clause.",
                "evidence_section": "7.1",
                "question_type": "direct",
                "answerable": True,
            },
            {
                "question_id": "SYNTHETIC-Q02",
                "policy_id": "IMP03",
                "question": "Where is the second synthetic rule?",
                "expected_answer": "It is in another target clause.",
                "evidence_section": "2.1",
                "question_type": "paraphrase",
                "answerable": True,
            },
            {
                "question_id": "SYNTHETIC-Q03",
                "policy_id": "IMP08",
                "question": "Where is the absent synthetic rule?",
                "expected_answer": "It is not represented in this fixture.",
                "evidence_section": "9.9",
                "question_type": "scenario",
                "answerable": True,
            },
            {
                "question_id": "SYNTHETIC-Q04",
                "policy_id": "IMP09",
                "question": "Where is the fourth synthetic rule?",
                "expected_answer": "It is in the final target clause.",
                "evidence_section": "4.1",
                "question_type": "scope_or_condition",
                "answerable": True,
            },
            {
                "question_id": "SYNTHETIC-Q05",
                "policy_id": "IMP09",
                "question": "What detail is not stated?",
                "expected_answer": "The policies do not state it.",
                "evidence_section": "",
                "question_type": "unanswerable",
                "answerable": False,
            },
        ]

    @staticmethod
    def make_chunks() -> list[Chunk]:
        return [
            Chunk(
                chunk_id="wrong-section",
                document_id="IMP02",
                document_title="Synthetic policy two",
                source_path="raw/imp02.html",
                section="7.10 Different clause",
                text="Same policy, wrong section.",
                word_count=4,
                metadata={"position": 1},
            ),
            Chunk(
                chunk_id="wrong-policy",
                document_id="IMP03",
                document_title="Synthetic policy three",
                source_path="raw/imp03.html",
                section="7.1 Same-numbered clause",
                text="Same section number, wrong policy.",
                word_count=5,
                metadata={"position": 2},
            ),
            Chunk(
                chunk_id="correct-first",
                document_id="IMP02",
                document_title="Synthetic policy two",
                source_path="raw/imp02.html",
                section="7.1 Target clause",
                text="Correct policy and section for question one.",
                word_count=7,
                metadata={"position": 3},
            ),
            Chunk(
                chunk_id="correct-second",
                document_id="IMP03",
                document_title="Synthetic policy three",
                source_path="raw/imp03.html",
                section="2.1 Other target clause",
                text="Correct policy and section for question two.",
                word_count=7,
                metadata={"position": 4},
            ),
            Chunk(
                chunk_id="fifth-distractor",
                document_id="IMP09",
                document_title="Synthetic policy nine",
                source_path="raw/imp09.html",
                section="4.1 Final target clause",
                text="Correct policy and section for question four.",
                word_count=7,
                metadata={"position": 5},
            ),
        ]

    @classmethod
    def make_repository(cls, temp_dir: str) -> dict:
        root = Path(temp_dir)
        question_path = (
            root
            / "data"
            / "evaluation"
            / "final_questions.json"
        )
        question_path.parent.mkdir(parents=True)
        question_bytes = json.dumps(
            cls.make_questions(),
            indent=2,
        ).encode("utf-8")
        question_path.write_bytes(question_bytes)

        chunks = cls.make_chunks()
        chunk_rows = [asdict(chunk) for chunk in chunks]
        tfidf_dir = root / "artifacts" / "tfidf_index"
        minilm_dir = root / "artifacts" / "semantic_index"
        tfidf_dir.mkdir(parents=True)
        minilm_dir.mkdir(parents=True)

        tfidf_metadata = {
            "strategy": "policy",
            "document_count": 4,
            "chunk_count": len(chunks),
            "vocabulary_size": 20,
        }
        minilm_metadata = {
            "model_name": (
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "document_count": 4,
            "chunk_count": len(chunks),
            "embedding_dimension": 384,
            "chunking_strategy": "policy",
        }

        (tfidf_dir / "metadata.json").write_text(
            json.dumps(tfidf_metadata, indent=2),
            encoding="utf-8",
        )
        (tfidf_dir / "chunks.json").write_text(
            json.dumps(chunk_rows, indent=2),
            encoding="utf-8",
        )
        (tfidf_dir / "vectorizer.joblib").write_bytes(
            b"synthetic vectorizer"
        )
        (tfidf_dir / "matrix.joblib").write_bytes(
            b"synthetic sparse matrix"
        )

        # The object content and list order are identical to TF-IDF,
        # while key order and JSON formatting deliberately differ.
        reordered_rows = [
            {
                key: row[key]
                for key in reversed(tuple(row))
            }
            for row in chunk_rows
        ]
        (minilm_dir / "metadata.json").write_text(
            json.dumps(minilm_metadata, separators=(",", ":")),
            encoding="utf-8",
        )
        (minilm_dir / "chunks.json").write_text(
            json.dumps(reordered_rows, separators=(",", ":")),
            encoding="utf-8",
        )
        (minilm_dir / "embeddings.npy").write_bytes(
            b"synthetic embeddings"
        )

        return {
            "root": root,
            "question_path": question_path,
            "question_digest": hashlib.sha256(
                question_bytes
            ).hexdigest(),
            "tfidf_dir": tfidf_dir,
            "minilm_dir": minilm_dir,
            "output_dir": (
                root
                / "artifacts"
                / "retrieval_evaluation"
            ),
            "questions": cls.make_questions(),
            "chunks": chunks,
        }

    @staticmethod
    def clean_git() -> dict:
        return {
            "commit": "0123456789abcdef",
            "branch": "main",
            "clean": True,
            "errors": None,
        }

    @staticmethod
    def environment_identity() -> dict:
        return {
            "python_version": "3.11.synthetic",
            "python_implementation": "CPython",
            "platform": {
                "system": "Windows",
                "machine": "AMD64",
            },
            "platform_errors": None,
            "package_versions": {
                "numpy": "synthetic",
                "scikit-learn": "synthetic",
                "sentence-transformers": "synthetic",
            },
            "package_errors": None,
        }

    @staticmethod
    def fixed_identity() -> tuple[datetime, str]:
        return (
            datetime(
                2026,
                8,
                5,
                15,
                0,
                0,
                321,
                tzinfo=timezone.utc,
            ),
            "retrieval123",
        )

    @contextmanager
    def patched_final_environment(
        self,
        fixture: dict,
        *,
        expected_digest: str | None = None,
        git_identity: dict | None = None,
        identity: tuple[datetime, str] | None = None,
    ):
        digest = (
            fixture["question_digest"]
            if expected_digest is None
            else expected_digest
        )
        git_result = (
            self.clean_git()
            if git_identity is None
            else git_identity
        )
        run_identity = identity or self.fixed_identity()

        with patch.object(
            retrieval_evaluation,
            "REPOSITORY_ROOT",
            fixture["root"],
        ), patch.object(
            retrieval_evaluation,
            "FROZEN_FINAL_QUESTION_SHA256",
            digest,
        ), patch.object(
            retrieval_evaluation,
            "collect_git_info",
            return_value=git_result,
        ), patch.object(
            retrieval_evaluation,
            "collect_python_environment",
            return_value=self.environment_identity(),
        ), patch.object(
            retrieval_evaluation,
            "new_run_identity",
            return_value=run_identity,
        ):
            yield

    @staticmethod
    def make_rankings(fixture: dict) -> tuple[dict, dict]:
        questions = fixture["questions"]
        chunks = fixture["chunks"]
        (
            wrong_section,
            wrong_policy,
            correct_first,
            correct_second,
            fifth_distractor,
        ) = chunks

        tfidf = {
            questions[0]["question"]: [
                (0.9, wrong_section),
                (0.8, wrong_policy),
                (0.7, correct_first),
                (0.6, correct_second),
                (0.5, fifth_distractor),
            ],
            questions[1]["question"]: [
                (0.9, correct_second),
                (0.8, wrong_section),
                (0.7, wrong_policy),
                (0.6, correct_first),
                (0.5, fifth_distractor),
            ],
            questions[2]["question"]: [
                (0.9, wrong_policy),
                (0.8, wrong_section),
                (0.7, correct_second),
                (0.6, correct_first),
                (0.5, fifth_distractor),
            ],
            questions[3]["question"]: [
                (0.9, wrong_section),
                (0.8, wrong_policy),
                (0.7, correct_first),
                (0.6, correct_second),
                (0.5, fifth_distractor),
            ],
        }
        minilm = {
            questions[0]["question"]: [
                (0.95, wrong_policy),
                (0.85, correct_first),
                (0.75, wrong_section),
                (0.65, correct_second),
                (0.55, fifth_distractor),
            ],
            questions[1]["question"]: [
                (0.95, wrong_section),
                (0.85, wrong_policy),
                (0.75, correct_first),
                (0.65, fifth_distractor),
                (0.55, correct_second),
            ],
            questions[2]["question"]: [
                (0.95, correct_first),
                (0.85, correct_second),
                (0.75, wrong_policy),
                (0.65, wrong_section),
                (0.55, fifth_distractor),
            ],
            questions[3]["question"]: [
                (0.95, fifth_distractor),
                (0.85, correct_first),
                (0.75, correct_second),
                (0.65, wrong_policy),
                (0.55, wrong_section),
            ],
        }
        return tfidf, minilm

    def make_search(
        self,
        rankings: dict,
        expected_index_dir: Path,
        chunk_count: int,
    ) -> Mock:
        search = Mock()

        def synthetic_search(query, index_dir, *, top_k):
            self.assertEqual(
                Path(index_dir).resolve(),
                expected_index_dir.resolve(),
            )
            self.assertEqual(top_k, chunk_count)
            return rankings[query]

        search.side_effect = synthetic_search
        return search

    def run_success_fixture(
        self,
        fixture: dict,
        *,
        patch_clock: bool = False,
    ) -> tuple[dict, Mock, Mock]:
        tfidf_rankings, minilm_rankings = self.make_rankings(
            fixture
        )
        tfidf_search = self.make_search(
            tfidf_rankings,
            fixture["tfidf_dir"],
            len(fixture["chunks"]),
        )
        minilm_search = self.make_search(
            minilm_rankings,
            fixture["minilm_dir"],
            len(fixture["chunks"]),
        )
        clock_values = (
            0.0,
            1.0,
            10.0,
            12.0,
            20.0,
            23.0,
            30.0,
            34.0,
            40.0,
            45.0,
            50.0,
            56.0,
            60.0,
            67.0,
            70.0,
            78.0,
        )

        with self.patched_final_environment(fixture):
            if patch_clock:
                with patch.object(
                    retrieval_evaluation,
                    "perf_counter",
                    side_effect=clock_values,
                ):
                    report = run_final_retrieval_evaluation(
                        question_path=fixture["question_path"],
                        output_dir=fixture["output_dir"],
                        tfidf_index_dir=fixture["tfidf_dir"],
                        minilm_index_dir=fixture["minilm_dir"],
                        tfidf_search_function=tfidf_search,
                        minilm_search_function=minilm_search,
                    )
            else:
                report = run_final_retrieval_evaluation(
                    question_path=fixture["question_path"],
                    output_dir=fixture["output_dir"],
                    tfidf_index_dir=fixture["tfidf_dir"],
                    minilm_index_dir=fixture["minilm_dir"],
                    tfidf_search_function=tfidf_search,
                    minilm_search_function=minilm_search,
                )

        return report, tfidf_search, minilm_search

    def assert_preflight_rejected(
        self,
        fixture: dict,
        *,
        expected_exception: type[Exception],
        expected_message: str,
        question_path: Path | None = None,
        output_dir: Path | None = None,
        tfidf_index_dir: Path | None = None,
        minilm_index_dir: Path | None = None,
        expected_digest: str | None = None,
        git_identity: dict | None = None,
    ) -> None:
        tfidf_search = Mock()
        minilm_search = Mock()

        with self.patched_final_environment(
            fixture,
            expected_digest=expected_digest,
            git_identity=git_identity,
        ):
            with self.assertRaisesRegex(
                expected_exception,
                expected_message,
            ):
                run_final_retrieval_evaluation(
                    question_path=(
                        question_path or fixture["question_path"]
                    ),
                    output_dir=(
                        output_dir or fixture["output_dir"]
                    ),
                    tfidf_index_dir=(
                        tfidf_index_dir or fixture["tfidf_dir"]
                    ),
                    minilm_index_dir=(
                        minilm_index_dir or fixture["minilm_dir"]
                    ),
                    tfidf_search_function=tfidf_search,
                    minilm_search_function=minilm_search,
                )

        tfidf_search.assert_not_called()
        minilm_search.assert_not_called()
        self.assertFalse(
            fixture["output_dir"].exists()
            and any(fixture["output_dir"].glob("*.json"))
        )

    def test_frozen_retrieval_identity_constants_are_locked(
        self,
    ) -> None:
        self.assertEqual(
            FROZEN_FINAL_QUESTION_PATH.as_posix(),
            "data/evaluation/final_questions.json",
        )
        self.assertEqual(
            FROZEN_FINAL_QUESTION_SHA256,
            (
                "ad8a474a0f5c19f4acba9d77dafeb903"
                "57cb208e2f5fe44463b4b34f60f4bd28"
            ),
        )
        self.assertEqual(
            FINAL_TFIDF_INDEX_PATH.as_posix(),
            "artifacts/tfidf_index",
        )
        self.assertEqual(
            FINAL_MINILM_INDEX_PATH.as_posix(),
            "artifacts/semantic_index",
        )

    def test_final_report_calculates_ranks_and_persists_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self.make_repository(temp_dir)
            report, tfidf_search, minilm_search = (
                self.run_success_fixture(
                    fixture,
                    patch_clock=True,
                )
            )

            self.assertEqual(tfidf_search.call_count, 4)
            self.assertEqual(minilm_search.call_count, 4)
            answerable_queries = [
                question["question"]
                for question in fixture["questions"]
                if question["answerable"] is True
            ]
            self.assertEqual(
                [call.args[0] for call in tfidf_search.call_args_list],
                answerable_queries,
            )
            self.assertEqual(
                [call.args[0] for call in minilm_search.call_args_list],
                answerable_queries,
            )

            self.assertEqual(
                report["result_path"],
                (
                    "artifacts/retrieval_evaluation/"
                    "final_retrieval_"
                    "20260805T150000000321Z_retrieval123.json"
                ),
            )
            result_path = fixture["root"] / report["result_path"]
            self.assertEqual(
                result_path.name,
                (
                    "final_retrieval_"
                    "20260805T150000000321Z_retrieval123.json"
                ),
            )
            saved_report = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            self.assertEqual(saved_report, report)
            self.assertEqual(report["run_id"], "retrieval123")
            self.assertEqual(
                report["created_at_utc"],
                "2026-08-05T15:00:00.000321+00:00",
            )
            self.assertEqual(report["run_kind"], "final_retrieval")
            self.assertEqual(
                report["question_path"],
                "data/evaluation/final_questions.json",
            )
            self.assertEqual(
                report["question_sha256"],
                hashlib.sha256(
                    fixture["question_path"].read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(report["git"], self.clean_git())
            self.assertEqual(
                report["environment"],
                self.environment_identity(),
            )
            self.assertEqual(
                report["question_counts"],
                {
                    "total": 5,
                    "answerable": 4,
                    "unanswerable": 1,
                },
            )
            self.assertEqual(
                set(report["indexes"]),
                {"tfidf", "minilm"},
            )
            self.assertEqual(
                report["indexes"]["tfidf"]["path"],
                "artifacts/tfidf_index",
            )
            self.assertEqual(
                report["indexes"]["minilm"]["path"],
                "artifacts/semantic_index",
            )

            required_files = {
                "tfidf": (
                    fixture["tfidf_dir"],
                    {
                        "metadata.json",
                        "chunks.json",
                        "vectorizer.joblib",
                        "matrix.joblib",
                    },
                ),
                "minilm": (
                    fixture["minilm_dir"],
                    {
                        "metadata.json",
                        "chunks.json",
                        "embeddings.npy",
                    },
                ),
            }

            for label, (index_dir, filenames) in (
                required_files.items()
            ):
                for filename in filenames:
                    expected_hash = hashlib.sha256(
                        (index_dir / filename).read_bytes()
                    ).hexdigest()
                    self.assertEqual(
                        report["indexes"][label]["files"][filename],
                        expected_hash,
                    )

            self.assertNotEqual(
                report["indexes"]["tfidf"]["files"][
                    "chunks.json"
                ],
                report["indexes"]["minilm"]["files"][
                    "chunks.json"
                ],
            )
            self.assertTrue(report["chunk_consistency"]["matches"])
            self.assertEqual(
                report["chunk_consistency"]["chunk_count"],
                5,
            )
            canonical_chunks = json.dumps(
                [
                    asdict(chunk)
                    for chunk in fixture["chunks"]
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            self.assertEqual(
                report["chunk_consistency"]["canonical_sha256"],
                hashlib.sha256(
                    canonical_chunks.encode("utf-8")
                ).hexdigest(),
            )

            tfidf = report["results"]["tfidf"]
            minilm = report["results"]["minilm"]
            self.assertEqual(
                {
                    key: tfidf[key]
                    for key in (
                        "retriever",
                        "evaluated_question_count",
                        "full_ranking_chunk_count",
                        "hit_at_1",
                        "hit_at_3",
                        "not_found_count",
                        "mean_search_seconds",
                    )
                },
                {
                    "retriever": "tfidf",
                    "evaluated_question_count": 4,
                    "full_ranking_chunk_count": 5,
                    "hit_at_1": 1 / 4,
                    "hit_at_3": 2 / 4,
                    "not_found_count": 1,
                    "mean_search_seconds": 2.5,
                },
            )
            self.assertAlmostEqual(tfidf["mrr"], 23 / 60)
            self.assertEqual(
                [
                    detail["correct_rank"]
                    for detail in tfidf["details"]
                ],
                [3, 1, None, 5],
            )
            self.assertEqual(
                [
                    detail["search_seconds"]
                    for detail in tfidf["details"]
                ],
                [1.0, 2.0, 3.0, 4.0],
            )
            self.assertEqual(
                [
                    detail["question_type"]
                    for detail in tfidf["details"]
                ],
                [
                    "direct",
                    "paraphrase",
                    "scenario",
                    "scope_or_condition",
                ],
            )

            self.assertEqual(minilm["retriever"], "minilm")
            self.assertEqual(minilm["evaluated_question_count"], 4)
            self.assertEqual(minilm["full_ranking_chunk_count"], 5)
            self.assertEqual(minilm["hit_at_1"], 1 / 4)
            self.assertAlmostEqual(minilm["hit_at_3"], 2 / 4)
            self.assertEqual(minilm["mrr"], 17 / 40)
            self.assertEqual(minilm["not_found_count"], 1)
            self.assertEqual(minilm["mean_search_seconds"], 6.5)
            self.assertEqual(
                [
                    detail["correct_rank"]
                    for detail in minilm["details"]
                ],
                [2, 5, None, 1],
            )

            detail_fields = {
                "question_id",
                "question_type",
                "target_policy",
                "target_section",
                "correct_rank",
                "search_seconds",
                "top_3_candidates",
            }
            candidate_fields = {
                "rank",
                "score",
                "chunk_id",
                "document_id",
                "section",
                "document_title",
            }

            for retrieval_result in (tfidf, minilm):
                self.assertEqual(len(retrieval_result["details"]), 4)

                for detail in retrieval_result["details"]:
                    self.assertEqual(set(detail), detail_fields)
                    self.assertLessEqual(
                        len(detail["top_3_candidates"]),
                        3,
                    )

                    for candidate in detail["top_3_candidates"]:
                        self.assertEqual(
                            set(candidate),
                            candidate_fields,
                        )
                        self.assertNotIn("text", candidate)
                        self.assertNotIn("source_path", candidate)

            first_tfidf_candidates = tfidf["details"][0][
                "top_3_candidates"
            ]
            self.assertEqual(
                first_tfidf_candidates[0],
                {
                    "rank": 1,
                    "score": 0.9,
                    "chunk_id": "wrong-section",
                    "document_id": "IMP02",
                    "section": "7.10 Different clause",
                    "document_title": "Synthetic policy two",
                },
            )
            self.assertEqual(
                tfidf["details"][0]["target_policy"],
                "IMP02",
            )
            self.assertEqual(
                tfidf["details"][0]["target_section"],
                "7.1",
            )

    def test_preflight_rejects_wrong_question_path_and_hash(
        self,
    ) -> None:
        for case in ("path", "hash"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                fixture = self.make_repository(temp_dir)

                if case == "path":
                    alternate = (
                        fixture["root"]
                        / "data"
                        / "evaluation"
                        / "other_questions.json"
                    )
                    alternate.write_bytes(
                        fixture["question_path"].read_bytes()
                    )
                    self.assert_preflight_rejected(
                        fixture,
                        expected_exception=ValueError,
                        expected_message="frozen question path",
                        question_path=alternate,
                    )
                else:
                    self.assert_preflight_rejected(
                        fixture,
                        expected_exception=ValueError,
                        expected_message="SHA-256",
                        expected_digest="0" * 64,
                    )

    def test_preflight_rejects_git_and_wrong_frozen_paths(
        self,
    ) -> None:
        cases = (
            "dirty_git",
            "missing_commit",
            "tfidf_path",
            "minilm_path",
            "output_path",
        )

        for case in cases:
            with self.subTest(
                case=case
            ), tempfile.TemporaryDirectory() as temp_dir:
                fixture = self.make_repository(temp_dir)
                arguments = {}

                if case == "dirty_git":
                    arguments["git_identity"] = {
                        "commit": "0123456789abcdef",
                        "branch": "main",
                        "clean": False,
                        "errors": None,
                    }
                    error_type = RuntimeError
                    error_message = "clean worktree"
                elif case == "missing_commit":
                    arguments["git_identity"] = {
                        "commit": None,
                        "branch": "main",
                        "clean": True,
                        "errors": None,
                    }
                    error_type = RuntimeError
                    error_message = "Git commit"
                elif case == "tfidf_path":
                    arguments["tfidf_index_dir"] = (
                        fixture["root"] / "artifacts" / "other_tfidf"
                    )
                    error_type = ValueError
                    error_message = "TF-IDF index"
                elif case == "minilm_path":
                    arguments["minilm_index_dir"] = (
                        fixture["root"] / "artifacts" / "other_minilm"
                    )
                    error_type = ValueError
                    error_message = "MiniLM index"
                else:
                    arguments["output_dir"] = (
                        fixture["root"] / "other_reports"
                    )
                    error_type = ValueError
                    error_message = "frozen output path"

                self.assert_preflight_rejected(
                    fixture,
                    expected_exception=error_type,
                    expected_message=error_message,
                    **arguments,
                )

    def test_preflight_requires_every_index_file(
        self,
    ) -> None:
        required_files = {
            "tfidf": (
                "metadata.json",
                "chunks.json",
                "vectorizer.joblib",
                "matrix.joblib",
            ),
            "minilm": (
                "metadata.json",
                "chunks.json",
                "embeddings.npy",
            ),
        }

        for index_label, filenames in required_files.items():
            for filename in filenames:
                with self.subTest(
                    index=index_label,
                    filename=filename,
                ), tempfile.TemporaryDirectory() as temp_dir:
                    fixture = self.make_repository(temp_dir)
                    index_dir = fixture[f"{index_label}_dir"]
                    (index_dir / filename).unlink()

                    expected_message = (
                        "identity could not be verified"
                        if filename == "metadata.json"
                        else "missing required files"
                    )
                    self.assert_preflight_rejected(
                        fixture,
                        expected_exception=RuntimeError,
                        expected_message=expected_message,
                    )

    def test_preflight_validates_metadata_and_chunk_counts(
        self,
    ) -> None:
        cases = (
            ("semantic_model", "minilm"),
            ("tfidf_chunk_count", "tfidf"),
            ("minilm_chunk_count", "minilm"),
        )

        for case, index_label in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                fixture = self.make_repository(temp_dir)
                metadata_path = (
                    fixture[f"{index_label}_dir"] / "metadata.json"
                )
                metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )

                if case == "semantic_model":
                    metadata["model_name"] = "synthetic-wrong-model"
                    expected_message = "wrong embedding model"
                else:
                    metadata["chunk_count"] = len(
                        fixture["chunks"]
                    ) + 1
                    expected_message = "chunk_count"

                metadata_path.write_text(
                    json.dumps(metadata),
                    encoding="utf-8",
                )
                self.assert_preflight_rejected(
                    fixture,
                    expected_exception=RuntimeError,
                    expected_message=expected_message,
                )

    def test_preflight_rejects_chunk_content_or_order_mismatch(
        self,
    ) -> None:
        for case in ("content", "order"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                fixture = self.make_repository(temp_dir)
                minilm_chunks_path = (
                    fixture["minilm_dir"] / "chunks.json"
                )
                chunks = json.loads(
                    minilm_chunks_path.read_text(encoding="utf-8")
                )

                if case == "content":
                    chunks[0]["text"] = "Changed synthetic content."
                else:
                    chunks.reverse()

                minilm_chunks_path.write_text(
                    json.dumps(chunks),
                    encoding="utf-8",
                )
                self.assert_preflight_rejected(
                    fixture,
                    expected_exception=RuntimeError,
                    expected_message="use different chunks",
                )

    def test_invalid_full_rankings_clean_reserved_report(
        self,
    ) -> None:
        for case in ("wrong_length", "duplicate_chunk_id"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp_dir:
                fixture = self.make_repository(temp_dir)
                chunks = fixture["chunks"]

                if case == "wrong_length":
                    invalid_ranking = [
                        (0.9, chunk)
                        for chunk in chunks[:-1]
                    ]
                    expected_message = "incomplete ranking"
                else:
                    invalid_ranking = [
                        (0.9, chunks[0]),
                        (0.8, chunks[1]),
                        (0.7, chunks[2]),
                        (0.6, chunks[3]),
                        (0.5, chunks[3]),
                    ]
                    expected_message = (
                        "duplicate or invalid chunk IDs"
                    )

                tfidf_search = Mock(return_value=invalid_ranking)
                minilm_search = Mock()
                created_at, run_id = self.fixed_identity()
                expected_path = fixture["output_dir"] / (
                    "final_retrieval_"
                    "20260805T150000000321Z_retrieval123.json"
                )

                with self.patched_final_environment(
                    fixture,
                    identity=(created_at, run_id),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        expected_message,
                    ):
                        run_final_retrieval_evaluation(
                            question_path=fixture["question_path"],
                            output_dir=fixture["output_dir"],
                            tfidf_index_dir=fixture["tfidf_dir"],
                            minilm_index_dir=fixture["minilm_dir"],
                            tfidf_search_function=tfidf_search,
                            minilm_search_function=minilm_search,
                        )

                self.assertEqual(tfidf_search.call_count, 1)
                minilm_search.assert_not_called()
                self.assertFalse(expected_path.exists())

    def test_search_exception_cleans_reserved_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self.make_repository(temp_dir)
            tfidf_rankings, _ = self.make_rankings(fixture)
            tfidf_search = self.make_search(
                tfidf_rankings,
                fixture["tfidf_dir"],
                len(fixture["chunks"]),
            )
            minilm_search = Mock(
                side_effect=RuntimeError("synthetic search failure")
            )
            fixture["output_dir"].mkdir()
            sentinel_path = fixture["output_dir"] / "historical.json"
            sentinel_bytes = b'{"historical": "preserve me"}'
            sentinel_path.write_bytes(sentinel_bytes)
            expected_path = fixture["output_dir"] / (
                "final_retrieval_"
                "20260805T150000000321Z_retrieval123.json"
            )

            with self.patched_final_environment(fixture):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "synthetic search failure",
                ):
                    run_final_retrieval_evaluation(
                        question_path=fixture["question_path"],
                        output_dir=fixture["output_dir"],
                        tfidf_index_dir=fixture["tfidf_dir"],
                        minilm_index_dir=fixture["minilm_dir"],
                        tfidf_search_function=tfidf_search,
                        minilm_search_function=minilm_search,
                    )

            self.assertEqual(tfidf_search.call_count, 4)
            self.assertEqual(minilm_search.call_count, 1)
            self.assertFalse(expected_path.exists())
            self.assertEqual(
                sentinel_path.read_bytes(),
                sentinel_bytes,
            )

    def test_output_collision_stops_before_searching(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = self.make_repository(temp_dir)
            fixture["output_dir"].mkdir()
            result_path = fixture["output_dir"] / (
                "final_retrieval_"
                "20260805T150000000321Z_retrieval123.json"
            )
            historical_bytes = b'{"historical": true}'
            result_path.write_bytes(historical_bytes)
            tfidf_search = Mock()
            minilm_search = Mock()

            with self.patched_final_environment(fixture):
                with self.assertRaises(FileExistsError):
                    run_final_retrieval_evaluation(
                        question_path=fixture["question_path"],
                        output_dir=fixture["output_dir"],
                        tfidf_index_dir=fixture["tfidf_dir"],
                        minilm_index_dir=fixture["minilm_dir"],
                        tfidf_search_function=tfidf_search,
                        minilm_search_function=minilm_search,
                    )

            tfidf_search.assert_not_called()
            minilm_search.assert_not_called()
            self.assertEqual(
                result_path.read_bytes(),
                historical_bytes,
            )


if __name__ == "__main__":
    unittest.main()
