"""Structured error helpers for engine/UI/CLI boundaries."""
from __future__ import annotations

import traceback as _traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ExperimentError:
    """Structured error object passed from engine code to UI/CLI layers."""

    code: str
    message: str
    stage: str = "engine"
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentWarning:
    """Structured warning object for non-fatal experiment issues."""

    code: str
    message: str
    stage: str = "engine"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def exception_to_error(exc: BaseException, *, stage: str = "engine") -> ExperimentError:
    """Convert an exception into a serializable ExperimentError."""
    return ExperimentError(
        code=type(exc).__name__,
        message=str(exc),
        stage=stage,
        traceback=_traceback.format_exc(),
    )


def make_error_payload(exc: BaseException, *, stage: str = "engine", engine_file: Optional[str] = None) -> Dict[str, Any]:
    """Return the canonical dict shape used by Streamlit and CLI error paths."""
    err = exception_to_error(exc, stage=stage).to_dict()
    err["error"] = err.pop("code")
    if engine_file:
        err["engine_file"] = engine_file
    return err


def warning_payload(code: str, message: str, *, stage: str = "engine") -> Dict[str, Any]:
    return ExperimentWarning(code=code, message=message, stage=stage).to_dict()


def format_error_result(res: Dict[str, Any]) -> str:
    """Human-readable one-line error formatting for CLI/UI."""
    code = str(res.get("error") or res.get("code") or "UNKNOWN_ERROR")
    msg = str(res.get("message") or "Unknown error")
    stage = str(res.get("stage") or "engine")
    return f"{stage}: {code} — {msg}"
