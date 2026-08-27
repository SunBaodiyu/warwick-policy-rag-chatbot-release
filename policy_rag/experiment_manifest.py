"""Collect reproducible identities for local experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Callable
from uuid import uuid4


PACKAGE_DISTRIBUTIONS = (
    "pymupdf",
    "scikit-learn",
    "numpy",
    "sentence-transformers",
    "torch",
    "transformers",
    "streamlit",
)

_OLLAMA_MODEL_ROW = re.compile(
    r"^(?P<name>\S+)\s+"
    r"(?P<id>\S+)\s+"
    r"(?P<size>\d+(?:\.\d+)?\s+\S+)\s+"
    r"(?P<modified>.+?)\s*$"
)


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def new_run_identity() -> tuple[datetime, str]:
    """Return a UTC creation time and a unique run identifier."""

    return datetime.now(timezone.utc), uuid4().hex


def report_filename(
    prefix: str,
    created_at: datetime,
    run_id: str,
) -> str:
    """Build a report filename with UTC microseconds and a run ID."""

    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{run_id}.json"


def build_and_write_json_exclusive(
    result_path: str | Path,
    report_builder: Callable[[], dict],
) -> dict:
    """Reserve a new report path, build it, and remove partial output."""

    path = Path(result_path)
    handle = path.open("x", encoding="utf-8")

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


def _command_output(
    command: list[str],
    cwd: str | Path | None = None,
) -> tuple[str | None, str | None]:
    """Run a local identity command without invoking a shell."""

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"

    if completed.returncode != 0:
        message = completed.stderr.strip()
        return (
            None,
            message
            or f"command_failed_exit_{completed.returncode}",
        )

    return completed.stdout.strip(), None


def collect_git_info(
    repository_root: str | Path,
) -> dict:
    """Collect commit, branch, and worktree cleanliness."""

    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": [
            "git",
            "branch",
            "--show-current",
        ],
        "status": ["git", "status", "--porcelain"],
    }
    outputs: dict[str, str | None] = {}
    errors: dict[str, str] = {}

    for name, command in commands.items():
        output, error = _command_output(
            command,
            cwd=repository_root,
        )
        outputs[name] = output

        if error:
            errors[name] = error

    branch = outputs["branch"] or None

    if outputs["branch"] == "" and "branch" not in errors:
        errors["branch"] = "detached_or_unavailable"

    clean = (
        outputs["status"] == ""
        if outputs["status"] is not None
        else None
    )

    return {
        "commit": outputs["commit"],
        "branch": branch,
        "clean": clean,
        "errors": errors or None,
    }


def collect_python_environment() -> dict:
    """Collect Python, platform, and installed package versions."""

    package_versions: dict[str, str | None] = {}
    package_errors: dict[str, str] = {}

    for package_name in PACKAGE_DISTRIBUTIONS:
        try:
            package_versions[package_name] = (
                importlib_metadata.version(package_name)
            )
        except importlib_metadata.PackageNotFoundError:
            package_versions[package_name] = None
            package_errors[package_name] = "not_installed"
        except Exception as exc:
            package_versions[package_name] = None
            package_errors[package_name] = (
                f"{type(exc).__name__}: {exc}"
            )

    platform_values: dict[str, str | None] = {}
    platform_errors: dict[str, str] = {}
    getters = {
        "description": platform.platform,
        "system": platform.system,
        "release": platform.release,
        "version": platform.version,
        "machine": platform.machine,
        "processor": platform.processor,
    }

    for name, getter in getters.items():
        try:
            value = getter()
            platform_values[name] = value or None

            if not value:
                platform_errors[name] = "unavailable"
        except Exception as exc:
            platform_values[name] = None
            platform_errors[name] = (
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "python_version": sys.version,
        "python_implementation": (
            platform.python_implementation()
        ),
        "platform": platform_values,
        "platform_errors": platform_errors or None,
        "package_versions": package_versions,
        "package_errors": package_errors or None,
    }


def _relative_path(
    path: Path,
    repository_root: Path,
) -> tuple[str | None, str | None]:
    try:
        relative = path.resolve().relative_to(
            repository_root.resolve()
        )
    except ValueError:
        return None, "path_outside_repository"

    return relative.as_posix(), None


def collect_index_identity(
    index_dir: str | Path,
    repository_root: str | Path,
) -> dict:
    """Collect index metadata and file hashes."""

    root = Path(repository_root).resolve()
    supplied_directory = Path(index_dir)
    directory = (
        supplied_directory
        if supplied_directory.is_absolute()
        else root / supplied_directory
    ).resolve()
    relative_path, relative_error = _relative_path(
        directory,
        root,
    )
    errors: dict[str, str] = {}

    if relative_error:
        errors["path"] = relative_error

        return {
            "path": None,
            "metadata": None,
            "files": {},
            "errors": errors,
        }

    if not directory.is_dir():
        errors["directory"] = "index_directory_not_found"
        return {
            "path": relative_path,
            "metadata": None,
            "files": {},
            "errors": errors,
        }

    metadata = None
    metadata_path = directory / "metadata.json"

    if not metadata_path.is_file():
        errors["metadata"] = "metadata_json_not_found"
    else:
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors["metadata"] = (
                f"{type(exc).__name__}: {exc}"
            )

    file_hashes: dict[str, str | None] = {}

    for file_path in sorted(
        (
            candidate
            for candidate in directory.rglob("*")
            if candidate.is_file()
        ),
        key=lambda candidate: candidate.as_posix(),
    ):
        relative_file = file_path.relative_to(
            directory
        ).as_posix()

        try:
            file_hashes[relative_file] = file_sha256(
                file_path
            )
        except OSError as exc:
            file_hashes[relative_file] = None
            errors[f"file:{relative_file}"] = (
                f"{type(exc).__name__}: {exc}"
            )

    return {
        "path": relative_path,
        "metadata": metadata,
        "files": file_hashes,
        "errors": errors or None,
    }


def _parse_ollama_list(
    raw_output: str,
) -> tuple[list[dict], list[str]]:
    models: list[dict] = []
    errors: list[str] = []
    header_seen = False

    for line_number, line in enumerate(
        raw_output.splitlines(),
        start=1,
    ):
        stripped = line.strip()

        if not stripped:
            continue

        columns = stripped.split()

        if tuple(
            column.casefold()
            for column in columns[:4]
        ) == ("name", "id", "size", "modified"):
            header_seen = True
            continue

        if not header_seen:
            errors.append(
                f"line_{line_number}_before_header"
            )
            continue

        match = _OLLAMA_MODEL_ROW.fullmatch(stripped)

        if match is None:
            errors.append(
                f"line_{line_number}_could_not_be_parsed"
            )
            continue

        models.append(
            {
                "name": match.group("name"),
                "id": match.group("id"),
                "size": match.group("size"),
                "modified": match.group("modified"),
            }
        )

    if not header_seen:
        errors.append("ollama_list_header_not_found")

    return models, errors


def collect_ollama_identity() -> dict:
    """Collect the local Ollama version and installed model list."""

    version, version_error = _command_output(
        ["ollama", "--version"]
    )
    raw_models, list_error = _command_output(
        ["ollama", "list"]
    )
    errors: dict[str, object] = {}
    models: list[dict] = []

    if version_error:
        errors["version"] = version_error

    if list_error:
        errors["list"] = list_error
    elif raw_models is not None:
        models, parse_errors = _parse_ollama_list(
            raw_models
        )

        if parse_errors:
            errors["list_parse"] = parse_errors

    return {
        "version": version,
        "models": models,
        "errors": errors or None,
    }
