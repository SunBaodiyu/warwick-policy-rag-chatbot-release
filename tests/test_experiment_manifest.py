"""Tests for reproducible experiment identity collection."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from importlib import metadata as importlib_metadata
from pathlib import Path
from unittest.mock import patch

from policy_rag.experiment_manifest import (
    PACKAGE_DISTRIBUTIONS,
    collect_git_info,
    collect_index_identity,
    collect_ollama_identity,
    collect_python_environment,
    file_sha256,
)


class ExperimentManifestTests(unittest.TestCase):
    def test_file_sha256_matches_known_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "identity.bin"
            contents = b"synthetic identity bytes\x00\xff"
            path.write_bytes(contents)

            actual = file_sha256(path)

        expected = hashlib.sha256(contents).hexdigest()
        self.assertEqual(actual, expected)

    @patch("policy_rag.experiment_manifest.subprocess.run")
    def test_git_identity_uses_shell_false(
        self,
        mock_run,
    ) -> None:
        outputs = {
            ("git", "rev-parse", "HEAD"): "abc123\n",
            (
                "git",
                "branch",
                "--show-current",
            ): "main\n",
            (
                "git",
                "status",
                "--porcelain",
            ): "",
        }

        def completed(command, **kwargs):
            key = tuple(command)

            if key not in outputs:
                raise AssertionError(f"Unexpected command: {key}")

            return subprocess.CompletedProcess(
                command,
                0,
                stdout=outputs[key],
                stderr="",
            )

        mock_run.side_effect = completed

        result = collect_git_info("synthetic-repository")

        self.assertEqual(result["commit"], "abc123")
        self.assertEqual(result["branch"], "main")
        self.assertTrue(result["clean"])
        self.assertIsNone(result["errors"])
        self.assertEqual(mock_run.call_count, 3)

        for call in mock_run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)
            self.assertEqual(
                call.kwargs["cwd"],
                "synthetic-repository",
            )

    def test_index_identity_saves_metadata_and_file_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_dir = root / "artifacts" / "semantic_index"
            index_dir.mkdir(parents=True)
            metadata = {
                "model_name": "synthetic-minilm",
                "chunk_count": 2,
            }
            files = {
                "metadata.json": json.dumps(metadata).encode(),
                "chunks.json": b"[]",
                "embeddings.npy": b"synthetic embeddings",
            }

            for name, contents in files.items():
                (index_dir / name).write_bytes(contents)

            result = collect_index_identity(
                "artifacts/semantic_index",
                root,
            )

        self.assertEqual(
            result["path"],
            "artifacts/semantic_index",
        )
        self.assertEqual(result["metadata"], metadata)
        self.assertIsNone(result["errors"])

        for name, contents in files.items():
            self.assertEqual(
                result["files"][name],
                hashlib.sha256(contents).hexdigest(),
            )

    def test_python_environment_records_versions_and_errors(
        self,
    ) -> None:
        def package_version(package_name):
            if package_name == "torch":
                raise importlib_metadata.PackageNotFoundError(
                    package_name
                )

            return f"{package_name}-test-version"

        with patch(
            "policy_rag.experiment_manifest."
            "importlib_metadata.version",
            side_effect=package_version,
        ), patch(
            "policy_rag.experiment_manifest.platform.platform",
            return_value="Synthetic Windows",
        ), patch(
            "policy_rag.experiment_manifest.platform.system",
            return_value="Windows",
        ), patch(
            "policy_rag.experiment_manifest.platform.release",
            return_value="test-release",
        ), patch(
            "policy_rag.experiment_manifest.platform.version",
            return_value="test-version",
        ), patch(
            "policy_rag.experiment_manifest.platform.machine",
            return_value="AMD64",
        ), patch(
            "policy_rag.experiment_manifest.platform.processor",
            return_value="Synthetic CPU",
        ), patch(
            "policy_rag.experiment_manifest."
            "platform.python_implementation",
            return_value="SyntheticPython",
        ), patch(
            "policy_rag.experiment_manifest.sys.version",
            "3.11.synthetic",
        ):
            result = collect_python_environment()

        self.assertEqual(
            set(result["package_versions"]),
            set(PACKAGE_DISTRIBUTIONS),
        )
        self.assertIsNone(
            result["package_versions"]["torch"]
        )
        self.assertEqual(
            result["package_errors"]["torch"],
            "not_installed",
        )
        self.assertEqual(
            result["platform"]["system"],
            "Windows",
        )
        self.assertEqual(
            result["python_version"],
            "3.11.synthetic",
        )
        self.assertEqual(
            result["python_implementation"],
            "SyntheticPython",
        )

    @patch("policy_rag.experiment_manifest.subprocess.run")
    def test_ollama_identity_parses_models_without_shell(
        self,
        mock_run,
    ) -> None:
        outputs = {
            (
                "ollama",
                "--version",
            ): "ollama version is 0.11.4\n",
            (
                "ollama",
                "list",
            ): (
                "NAME ID SIZE MODIFIED\r\n"
                "qwen2.5:1.5b qwen-id 986 MB 2 days ago\r\n"
                "llama3.2:3b llama-id 2.0 GB 1 day ago\r\n"
            ),
        }

        def completed(command, **kwargs):
            key = tuple(command)

            if key not in outputs:
                raise AssertionError(f"Unexpected command: {key}")

            return subprocess.CompletedProcess(
                command,
                0,
                stdout=outputs[key],
                stderr="",
            )

        mock_run.side_effect = completed

        result = collect_ollama_identity()

        self.assertEqual(
            result["version"],
            "ollama version is 0.11.4",
        )
        self.assertEqual(
            result["models"],
            [
                {
                    "name": "qwen2.5:1.5b",
                    "id": "qwen-id",
                    "size": "986 MB",
                    "modified": "2 days ago",
                },
                {
                    "name": "llama3.2:3b",
                    "id": "llama-id",
                    "size": "2.0 GB",
                    "modified": "1 day ago",
                },
            ],
        )
        self.assertIsNone(result["errors"])

        for call in mock_run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)

    @patch("policy_rag.experiment_manifest.subprocess.run")
    def test_ollama_list_without_header_fails_closed(
        self,
        mock_run,
    ) -> None:
        outputs = {
            (
                "ollama",
                "--version",
            ): "ollama version synthetic\n",
            (
                "ollama",
                "list",
            ): "warning output with four or more words\n",
        }

        def completed(command, **kwargs):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=outputs[tuple(command)],
                stderr="",
            )

        mock_run.side_effect = completed

        result = collect_ollama_identity()

        self.assertEqual(result["models"], [])
        self.assertIn("list_parse", result["errors"])
        self.assertIn(
            "ollama_list_header_not_found",
            result["errors"]["list_parse"],
        )

    @patch(
        "policy_rag.experiment_manifest.subprocess.run",
        side_effect=FileNotFoundError("command unavailable"),
    )
    def test_ollama_command_failure_records_null_and_error(
        self,
        mock_run,
    ) -> None:
        result = collect_ollama_identity()

        self.assertIsNone(result["version"])
        self.assertEqual(result["models"], [])
        self.assertIn("version", result["errors"])
        self.assertIn("list", result["errors"])

        for call in mock_run.call_args_list:
            self.assertIs(call.kwargs["shell"], False)


if __name__ == "__main__":
    unittest.main()
