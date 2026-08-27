"""Tests for the local Ollama HTTP client."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from policy_rag.ollama_client import generate_local


class OllamaClientTests(unittest.TestCase):
    @patch("policy_rag.ollama_client.urlopen")
    def test_generate_local_preserves_ollama_metrics(
        self,
        mock_urlopen,
    ) -> None:
        ollama_response = {
            "model": "test-model:1b",
            "response": '{"answer": "synthetic"}',
            "total_duration": 900,
            "load_duration": 100,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 200,
            "eval_count": 10,
            "eval_duration": 500,
        }
        response = MagicMock()
        response.read.return_value = json.dumps(
            ollama_response
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = (
            response
        )
        schema = {
            "type": "object",
            "properties": {},
        }

        result = generate_local(
            "Synthetic prompt",
            model="test-model:1b",
            temperature=0.0,
            num_ctx=2048,
            num_predict=220,
            timeout=30,
            response_format=schema,
        )

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:11434/api/generate",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.get_header("Content-type"),
            "application/json",
        )
        payload = json.loads(
            request.data.decode("utf-8")
        )
        self.assertEqual(payload["model"], "test-model:1b")
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["format"], schema)
        self.assertEqual(
            payload["options"],
            {
                "temperature": 0.0,
                "num_ctx": 2048,
                "num_predict": 220,
            },
        )
        self.assertEqual(
            result["prompt_eval_duration_ns"],
            200,
        )
        self.assertEqual(
            result["actual_model"],
            "test-model:1b",
        )
        self.assertEqual(result["total_duration_ns"], 900)
        self.assertEqual(result["load_duration_ns"], 100)
        self.assertEqual(result["prompt_eval_count"], 20)
        self.assertEqual(result["eval_count"], 10)
        self.assertEqual(result["eval_duration_ns"], 500)
        self.assertEqual(
            mock_urlopen.call_args.kwargs["timeout"],
            30,
        )

    @patch("policy_rag.ollama_client.urlopen")
    def test_missing_prompt_duration_remains_explicitly_null(
        self,
        mock_urlopen,
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "model": "test-model:1b",
                "response": "synthetic response",
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = (
            response
        )

        result = generate_local(
            "Synthetic prompt",
            model="test-model:1b",
        )

        self.assertIsNone(
            result["prompt_eval_duration_ns"]
        )
        self.assertEqual(result["model"], "test-model:1b")
        self.assertEqual(
            result["actual_model"],
            "test-model:1b",
        )

    @patch("policy_rag.ollama_client.urlopen")
    def test_zero_prompt_duration_is_valid(
        self,
        mock_urlopen,
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "model": "test-model:1b",
                "response": "synthetic response",
                "prompt_eval_duration": 0,
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = (
            response
        )

        result = generate_local(
            "Synthetic prompt",
            model="test-model:1b",
        )

        self.assertEqual(result["prompt_eval_duration_ns"], 0)

    @patch("policy_rag.ollama_client.urlopen")
    def test_invalid_prompt_duration_types_are_rejected(
        self,
        mock_urlopen,
    ) -> None:
        cases = (
            ("bool_true", True),
            ("bool_false", False),
            ("string", "200"),
            ("float", 200.0),
            ("negative", -1),
            ("list", [200]),
            ("object", {"value": 200}),
            ("null", None),
        )

        for label, invalid_value in cases:
            with self.subTest(label=label):
                response = MagicMock()
                response.read.return_value = json.dumps(
                    {
                        "model": "test-model:1b",
                        "response": "synthetic response",
                        "prompt_eval_duration": invalid_value,
                    }
                ).encode("utf-8")
                mock_urlopen.return_value.__enter__.return_value = (
                    response
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    (
                        "^Ollama returned an invalid "
                        "prompt_eval_duration$"
                    ),
                ):
                    generate_local(
                        "Synthetic prompt",
                        model="test-model:1b",
                    )

    @patch("policy_rag.ollama_client.urlopen")
    def test_missing_response_model_is_not_fabricated_as_actual(
        self,
        mock_urlopen,
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"response": "synthetic response"}
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = (
            response
        )

        result = generate_local(
            "Synthetic prompt",
            model="requested-model:1b",
        )

        self.assertEqual(result["model"], "requested-model:1b")
        self.assertIsNone(result["actual_model"])


if __name__ == "__main__":
    unittest.main()
