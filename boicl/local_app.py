"""Local browser runner for BO-ICL/GPR experiments.

This module intentionally avoids web-framework dependencies. It serves a small
single-page app with Python's standard HTTP server, while the numerical work
stays inside the existing BO-ICL package.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import socket
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - the launcher installs python-dotenv
    load_dotenv = None


ACQUISITION_FUNCTIONS = [
    "upper_confidence_bound",
    "expected_improvement",
    "log_expected_improvement",
    "probability_of_improvement",
    "greedy",
    "random",
]


MODEL_PRESETS = [
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "openrouter/mistralai/mistral-7b-instruct:free",
    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]


EMBEDDING_MODEL_PRESETS = [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
]


DEFAULT_PREDICTION_SYSTEM_MESSAGE = (
    "You are a materials synthesis and phase-optimization assistant. Use the "
    "provided labeled examples to estimate the numeric objective for candidate "
    "experimental procedures. Treat the objective name in the prompt as the "
    "quantity to optimize. When the prompt asks for a numeric prediction, "
    "return only the number without units or prose."
)


DEFAULT_INVERSE_SYSTEM_MESSAGE = (
    "You are a materials synthesis and phase-optimization assistant. Use the "
    "provided examples and target objective value to propose realistic "
    "experimental procedures in the same style as the dataset. Keep proposals "
    "specific, physically plausible, and compatible with the described "
    "synthesis workflow."
)


GENERATED_PREDICTION_PROMPT_PREFIX = (
    "You are a careful experimental-design surrogate for Bayesian optimization."
)
GENERATED_INVERSE_PROMPT_PREFIX = (
    "You are a careful inverse-design assistant for Bayesian optimization."
)


DEFAULT_CONFIG = {
    "workflow_mode": "live",
    "optimizer": "gpr",
    "objective_name": "objective",
    "objective_direction": "maximize",
    "objective_scaling": "off",
    "acquisition": "upper_confidence_bound",
    "embedding_model": "text-embedding-ada-002",
    "prediction_model": "gpt-4o",
    "inverse_model": "gpt-4o",
    "prediction_system_message": DEFAULT_PREDICTION_SYSTEM_MESSAGE,
    "inverse_system_message": DEFAULT_INVERSE_SYSTEM_MESSAGE,
    "llm_samples": 3,
    "selector_k": 0,
    "inverse_filter": 16,
    "inverse_random_candidates": 0,
    "inverse_target_value": "",
    "inverse_target_multiplier": 1.2,
    "inverse_target_jitter": 0.05,
    "inverse_target_floor_value": "",
    "inverse_design_count": 3,
    "batch_size": 1,
    "iterations_per_trial": 0,
    "replicates_per_candidate": 1,
    "benchmark_iterations": 30,
    "benchmark_replicates": 5,
    "benchmark_initial_points": 1,
    "benchmark_seed": 0,
    "benchmark_starting_baseline": "none",
    "greedy_final_iteration": False,
    "random_replicates": 0,
    "ucb_lambda": 0.1,
    "score_limit": 250,
    "api_pause_seconds": 0.5,
    "api_retry_attempts": 8,
    "api_rate_limit_cooldown_seconds": 10.0,
    "n_neighbors": 5,
    "n_components": 16,
    "auto_suggest": True,
    "plot_stat_guides": "max",
}


BENCHMARK_COLORS = [
    "#0f766e",
    "#2563eb",
    "#7c3aed",
    "#b45309",
    "#c2410c",
    "#be123c",
    "#047857",
    "#4338ca",
]


SCALING_MODES = ["off", "auto", "minmax", "zscore"]
PLOT_STAT_GUIDES = ["off", "max", "paper"]
BENCHMARK_STARTING_BASELINES = ["none", "mean"]


class RunCancelled(RuntimeError):
    """Raised when the browser asks a long-running local task to stop."""


def _short_prompt_text(value: str, limit: int = 650) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _prompt_examples(candidates: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    if not candidates:
        return []
    if len(candidates) <= limit:
        selected = candidates
    else:
        indexes = np.linspace(0, len(candidates) - 1, limit, dtype=int)
        selected = [candidates[int(index)] for index in indexes]
    examples = []
    seen = set()
    for candidate in selected:
        procedure = _short_prompt_text(candidate.get("procedure", ""))
        if procedure and procedure not in seen:
            seen.add(procedure)
            examples.append(procedure)
    return examples


def _is_auto_prediction_prompt(value: Optional[str]) -> bool:
    text = (value or "").strip()
    return not text or text == DEFAULT_PREDICTION_SYSTEM_MESSAGE or text.startswith(
        GENERATED_PREDICTION_PROMPT_PREFIX
    )


def _is_auto_inverse_prompt(value: Optional[str]) -> bool:
    text = (value or "").strip()
    return not text or text == DEFAULT_INVERSE_SYSTEM_MESSAGE or text.startswith(
        GENERATED_INVERSE_PROMPT_PREFIX
    )


def _dataset_prompt_summary(
    candidates: List[Dict[str, Any]], objective_names: List[str]
) -> str:
    objectives = ", ".join(objective_names) if objective_names else "objective"
    lines = [
        f"Uploaded dataset summary: {len(candidates)} candidate procedures.",
        f"Objective columns available to the tool: {objectives}.",
        "The first uploaded column is the procedure text. Numeric labels, if present, "
        "are used by the tool only when an experiment is observed or simulated.",
    ]
    examples = _prompt_examples(candidates)
    if examples:
        lines.append("Procedure style examples from the uploaded pool:")
        lines.extend(f"{index}. {example}" for index, example in enumerate(examples, 1))
    return "\n".join(lines)


def _dataset_prediction_prompt(
    candidates: List[Dict[str, Any]], objective_names: List[str]
) -> str:
    return (
        f"{GENERATED_PREDICTION_PROMPT_PREFIX} You will receive relevant labeled "
        "examples and then one candidate experimental procedure from an uploaded "
        "dataset. Predict the numeric objective named in the prompt. Rely primarily "
        "on the examples, the procedure text, and the objective name; use general "
        "scientific knowledge only as a weak prior when examples are sparse. Do not "
        "claim access to current literature, hidden labels, or information outside "
        "the prompt. Return exactly one numeric value in the original objective "
        "units. Do not include units, JSON, ranges, uncertainty, citations, "
        "explanations, or extra text.\n\n"
        f"{_dataset_prompt_summary(candidates, objective_names)}"
    )


def _dataset_inverse_prompt(
    candidates: List[Dict[str, Any]], objective_names: List[str]
) -> str:
    return (
        f"{GENERATED_INVERSE_PROMPT_PREFIX} You will receive labeled examples and "
        "a target objective value. Generate one procedure-like design that would "
        "plausibly reach the requested target and matches the uploaded dataset "
        "style. If the workflow uses a finite candidate pool, your output is used "
        "as a retrieval query to find real candidates from that uploaded pool; do "
        "not invent measurements or labels. Stay within the apparent design space: "
        "reuse parameter names, units, syntax, reagents, ranges, and workflow steps "
        "seen in the examples unless the prompt explicitly allows otherwise. Return "
        "only the procedure text, with no explanation or formatting.\n\n"
        f"{_dataset_prompt_summary(candidates, objective_names)}"
    )


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _read_table_from_bytes(filename: str, raw: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(raw))
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(BytesIO(raw))
    if suffix == ".npy":
        return _read_npy_table(raw)
    raise ValueError("Use a CSV, Excel, or NPY file.")


def _read_npy_table(raw: bytes) -> pd.DataFrame:
    try:
        loaded = np.load(BytesIO(raw), allow_pickle=True)
    except Exception as exc:
        raise ValueError("Could not read the NPY file.") from exc
    if isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Use a .npy array file, not a .npz archive.")
    array = np.asarray(loaded)
    if array.dtype.names:
        return pd.DataFrame.from_records(array)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, dict):
            return pd.DataFrame(item)
        if isinstance(item, list):
            return pd.DataFrame(item)
        raise ValueError("The NPY file must contain a table-like array.")
    if array.ndim == 1:
        if array.dtype == object and len(array) and isinstance(array[0], dict):
            return pd.DataFrame(list(array))
        return pd.DataFrame({"procedure": array.tolist()})
    if array.ndim == 2:
        if array.shape[1] < 1:
            raise ValueError("The NPY table must include a procedure column.")
        columns = ["procedure"]
        if array.shape[1] == 2:
            columns.append("objective")
        else:
            columns.extend(
                f"objective_{index}" for index in range(1, array.shape[1])
            )
        return pd.DataFrame(array, columns=columns)
    raise ValueError("The NPY file must be a 1D or 2D table-like array.")


def _clean_procedures(df: pd.DataFrame) -> List[str]:
    if df.empty:
        raise ValueError("The dataset has no rows.")
    first_col = df.columns[0]
    procedures = [
        str(value).strip()
        for value in df[first_col].tolist()
        if pd.notna(value) and str(value).strip()
    ]
    if not procedures:
        raise ValueError("The first column did not contain any procedures.")
    return procedures


def _best_trace(
    observations: List[Dict[str, Any]],
    direction: str,
    baseline_value: Optional[float] = None,
    skip_observations: int = 0,
) -> List[Dict[str, Any]]:
    best = float(baseline_value) if baseline_value is not None else None
    trace = (
        [{"index": 1, "value": best, "best": best, "baseline": True}]
        if best is not None
        else []
    )
    plotted = 1 if best is not None else 0
    for obs in observations[max(0, int(skip_observations)) :]:
        value = obs["value"]
        if value is None:
            continue
        plotted += 1
        if best is None:
            best = value
        elif direction == "minimize":
            best = min(best, value)
        else:
            best = max(best, value)
        trace.append({"index": plotted, "value": value, "best": best})
    return trace


def _paper_random_trace(
    values: List[float],
    direction: str,
    steps: int,
    buckets: int = 100,
    baseline_value: Optional[float] = None,
) -> List[Dict[str, Any]]:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return []
    steps = max(0, int(steps))
    sign = -1.0 if direction == "minimize" else 1.0
    series = pd.Series([sign * value for value in clean_values])
    quantiles = [float(series.quantile(i / buckets)) for i in range(buckets + 1)]
    trace = []
    baseline_target = None
    if baseline_value is not None:
        baseline_target = sign * float(baseline_value)
        trace.append(
            {
                "index": 1,
                "best": float(baseline_value),
                "baseline": True,
            }
        )
    for sample_count in range(1, steps + 1):
        expected = 0.0
        for bucket in range(1, buckets + 1):
            probability = (bucket**sample_count - (bucket - 1) ** sample_count) * (
                (1 / buckets) ** sample_count
            )
            bucket_value = (quantiles[bucket - 1] + quantiles[bucket]) / 2
            if baseline_target is not None:
                bucket_value = max(baseline_target, bucket_value)
            expected += bucket_value * probability
        index = sample_count + 1 if baseline_target is not None else sample_count
        trace.append({"index": index, "best": float(sign * expected)})
    return trace


def _dataset_stats(
    values: List[float], direction: str = "maximize"
) -> List[Dict[str, Any]]:
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return []
    series = pd.Series(clean_values)
    if direction == "minimize":
        stats = [
            ("mean", float(series.mean())),
            ("25%", float(series.quantile(0.25))),
            ("5%", float(series.quantile(0.05))),
            ("1%", float(series.quantile(0.01))),
            ("min", float(series.min())),
        ]
    else:
        stats = [
            ("mean", float(series.mean())),
            ("75%", float(series.quantile(0.75))),
            ("95%", float(series.quantile(0.95))),
            ("99%", float(series.quantile(0.99))),
            ("max", float(series.max())),
        ]
    return [{"label": label, "value": value} for label, value in stats]


def _target_value(value: float, direction: str) -> float:
    return -value if direction == "minimize" else value


def _display_value(value: float, direction: str) -> float:
    return -value if direction == "minimize" else value


def _target_scaler(values: List[float], direction: str, mode: str) -> Dict[str, float]:
    mode = mode if mode in SCALING_MODES else "off"
    directed = np.array([_target_value(float(value), direction) for value in values])
    directed = directed[np.isfinite(directed)]
    if directed.size < 2:
        return {"mode": "off"}
    span = float(np.max(directed) - np.min(directed))
    if span <= 1e-12:
        return {"mode": "off"}
    if mode == "auto":
        mode = "minmax" if span > 1.0 or float(np.max(np.abs(directed))) > 1.0 else "off"
    if mode == "minmax":
        return {"mode": "minmax", "min": float(np.min(directed)), "span": span}
    if mode == "zscore":
        std = float(np.std(directed))
        if std <= 1e-12:
            return {"mode": "off"}
        return {"mode": "zscore", "mean": float(np.mean(directed)), "std": std}
    return {"mode": "off"}


def _scale_target(value: float, direction: str, scaler: Dict[str, float]) -> float:
    target = _target_value(float(value), direction)
    if scaler.get("mode") == "minmax":
        return (target - scaler["min"]) / scaler["span"]
    if scaler.get("mode") == "zscore":
        return (target - scaler["mean"]) / scaler["std"]
    return target


def _unscale_target(value: float, direction: str, scaler: Dict[str, float]) -> float:
    target = float(value)
    if scaler.get("mode") == "minmax":
        target = target * scaler["span"] + scaler["min"]
    elif scaler.get("mode") == "zscore":
        target = target * scaler["std"] + scaler["mean"]
    return _display_value(target, direction)


def _group_training_observations(
    observations: List[Dict[str, Any]], direction: str, scaling_mode: str
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for obs in observations:
        grouped.setdefault(obs["procedure"], []).append(obs)
    rows = []
    for procedure, obs_list in grouped.items():
        values = [obs["value"] for obs in obs_list if obs.get("value") is not None]
        if not values:
            continue
        rows.append({"procedure": procedure, "value": float(np.mean(values))})
    scaler = _target_scaler([row["value"] for row in rows], direction, scaling_mode)
    for row in rows:
        row["target"] = float(_scale_target(row["value"], direction, scaler))
    return rows, scaler


def _safe_cache_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "default"


def _is_uncertainty_column(column: Any) -> bool:
    name = str(column).lower()
    return any(token in name for token in ["uncert", "std", "stdev", "sigma", "error"])


def _numeric_columns(df: pd.DataFrame, columns: List[Any]) -> List[Any]:
    numeric = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric.append(column)
    return numeric


def _model_provider(model_name: str) -> str:
    if model_name.startswith("openrouter/"):
        return "openrouter"
    if model_name.startswith("claude-"):
        return "anthropic"
    return "openai"


def _required_key_name(model_name: str) -> str:
    provider = _model_provider(model_name)
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def _load_env_file(env_path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(env_path, override=True)
        return
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _clean_api_key_value(value: str, key_name: str) -> str:
    cleaned = (value or "").strip().strip("'\"")
    if "=" in cleaned:
        maybe_key, maybe_value = cleaned.split("=", 1)
        if maybe_key.strip() == key_name:
            cleaned = maybe_value.strip().strip("'\"")
    return cleaned


def _is_retryable_api_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "temporarily unavailable" in text
        or "timeout" in text
    )


def _retry_delay_seconds(
    exc: Exception,
    attempt: int,
    base: float,
    maximum: float,
    rate_limit_cooldown: float = 0.0,
) -> float:
    text = str(exc)
    match = re.search(
        r"try again in\s+([0-9]*\.?[0-9]+)\s*(ms|milliseconds|s|sec|seconds)?",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        value = float(match.group(1))
        unit = (match.group(2) or "s").lower()
        delay = value / 1000 if unit.startswith("m") else value
    else:
        delay = base * (2 ** attempt)
    lowered = text.lower()
    if rate_limit_cooldown > 0 and (
        "tokens per min" in lowered
        or "tpm" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
    ):
        delay = max(delay, rate_limit_cooldown * (attempt + 1))
    jitter = random.uniform(0.05, 0.25)
    return max(0.05, min(maximum, delay + jitter))


def _write_env_value(env_path: Path, key: str, value: str) -> None:
    value = _clean_api_key_value(value, key)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    found = False
    out = []
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    os.environ[key] = value


@dataclass
class LocalBOState:
    root: Path
    config: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    objective_names: List[str] = field(default_factory=lambda: ["objective"])
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    inverse_designs: List[Dict[str, Any]] = field(default_factory=list)
    benchmark_runs: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, str]] = field(default_factory=list)
    last_error: Optional[str] = None
    last_model_status: str = "No dataset loaded."
    campaign_id: Optional[str] = None
    campaign_name: str = ""
    progress: Dict[str, Any] = field(
        default_factory=lambda: {
            "status": "idle",
            "label": "",
            "detail": "",
            "current": 0,
            "total": 0,
            "percent": 0,
            "updated": "",
        }
    )

    def __post_init__(self) -> None:
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self._embedding_cache_model_obj = None
        self._embedding_cache_model_key = None
        self.env_path = self.root / ".env"
        self.cache_dir = self.root / ".cache"
        self.campaigns_dir = self.root / "saved_experiments"
        _load_env_file(self.env_path)

    def log(self, message: str) -> None:
        self.events.insert(0, {"time": _now(), "message": message})
        self.events = self.events[:20]

    def set_progress(
        self,
        label: str,
        current: int,
        total: int,
        detail: str = "",
        status: str = "running",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if status == "running" and self.cancel_event.is_set():
            status = "cancelling"
            detail = "Stop requested. Finishing the current API call before stopping."
        total = max(0, int(total))
        current = max(0, min(int(current), total if total else int(current)))
        percent = int(round((current / total) * 100)) if total else 0
        previous_partial = self.progress.get("partial_run")
        self.progress = {
            "status": status,
            "label": label,
            "detail": detail,
            "current": current,
            "total": total,
            "percent": percent,
            "updated": _now(),
        }
        if extra:
            self.progress.update(extra)
        elif status in {"running", "cancelling", "error", "cancelled"} and previous_partial:
            self.progress["partial_run"] = previous_partial
        message = f"{label}: {current}/{total}"
        if detail:
            message += f" - {detail}"
        print(f"[{self.progress['updated']}] {message}", flush=True)

    def finish_progress(self, label: str, detail: str = "") -> None:
        total = int(self.progress.get("total") or 0)
        self.cancel_event.clear()
        self.set_progress(label, total, total, detail=detail, status="complete")

    def fail_progress(self, label: str, detail: str = "") -> None:
        current = int(self.progress.get("current") or 0)
        total = int(self.progress.get("total") or 0)
        self.cancel_event.clear()
        self.set_progress(label, current, total, detail=detail, status="error")

    def cancel_progress(self, label: str, detail: str = "") -> None:
        current = int(self.progress.get("current") or 0)
        total = int(self.progress.get("total") or 0)
        self.set_progress(label, current, total, detail=detail, status="cancelled")
        self.cancel_event.clear()

    def request_cancel(self) -> Dict[str, Any]:
        self.cancel_event.set()
        progress = dict(self.progress)
        if progress.get("status") == "running":
            detail = "Stop requested. Waiting for the current API call to return."
            self.progress = {
                **progress,
                "status": "cancelling",
                "detail": detail,
                "updated": _now(),
            }
            print(f"[{self.progress['updated']}] {progress.get('label') or 'Run'}: {detail}", flush=True)
        return self.progress_snapshot()

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled("Run stopped by user.")

    def _sleep_with_cancel(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0, seconds)
        while time.monotonic() < deadline:
            self.check_cancelled()
            time.sleep(min(0.2, deadline - time.monotonic()))
        self.check_cancelled()

    def _api_pause(self) -> None:
        pause = float(self.config.get("api_pause_seconds") or 0)
        if pause > 0:
            self._sleep_with_cancel(pause)

    def _api_call_with_retries(self, label: str, func):
        attempts = max(1, int(self.config.get("api_retry_attempts") or 1))
        base_delay = max(0.1, float(self.config.get("api_pause_seconds") or 0.5))
        rate_limit_cooldown = max(
            0.0, float(self.config.get("api_rate_limit_cooldown_seconds") or 0.0)
        )
        max_delay = 60.0
        for attempt in range(attempts):
            self.check_cancelled()
            try:
                result = func()
                self._api_pause()
                return result
            except Exception as exc:
                if not _is_retryable_api_error(exc) or attempt >= attempts - 1:
                    raise
                delay = _retry_delay_seconds(
                    exc,
                    attempt,
                    base_delay,
                    max_delay,
                    rate_limit_cooldown,
                )
                current = int(self.progress.get("current") or 0)
                total = int(self.progress.get("total") or 0)
                self.set_progress(
                    label,
                    current,
                    total,
                    detail=(
                        f"API rate limit or transient error. Cooling down {delay:.1f}s "
                        f"before retry {attempt + 2}/{attempts}."
                    ),
                )
                self._sleep_with_cancel(delay)

    def progress_snapshot(self) -> Dict[str, Any]:
        return dict(self.progress)

    def campaign_file(self, campaign_id: str) -> Path:
        return self.campaigns_dir / campaign_id / "campaign.json"

    def list_campaigns(self) -> List[Dict[str, Any]]:
        if not self.campaigns_dir.exists():
            return []
        campaigns = []
        for path in sorted(self.campaigns_dir.glob("*/campaign.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            meta = payload.get("meta", {})
            campaigns.append(
                {
                    "id": meta.get("id") or path.parent.name,
                    "name": meta.get("name") or path.parent.name,
                    "updated": meta.get("updated") or "",
                    "candidate_count": len(payload.get("candidates", [])),
                    "observation_count": len(payload.get("observations", [])),
                }
            )
        campaigns.sort(key=lambda item: item.get("updated", ""), reverse=True)
        return campaigns

    def _unique_campaign_id(self, name: str) -> str:
        base = _safe_cache_fragment(name.lower()) or "campaign"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        candidate_id = f"{base}_{stamp}"
        counter = 2
        while self.campaign_file(candidate_id).exists():
            candidate_id = f"{base}_{stamp}_{counter}"
            counter += 1
        return candidate_id

    def _campaign_payload_locked(self) -> Dict[str, Any]:
        now = _now()
        return {
            "version": 1,
            "meta": {
                "id": self.campaign_id,
                "name": self.campaign_name,
                "saved": now,
                "updated": now,
            },
            "config": self.config,
            "objective_names": self.objective_names,
            "candidates": self.candidates,
            "observations": self.observations,
            "suggestions": self.suggestions,
            "inverse_designs": self.inverse_designs,
            "benchmark_runs": self.benchmark_runs,
            "last_model_status": self.last_model_status,
            "progress": self.progress,
            "events": self.events[:20],
        }

    def _write_campaign_locked(self) -> None:
        if not self.campaign_id:
            return
        path = self.campaign_file(self.campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._campaign_payload_locked()
        existing_created = None
        if path.exists():
            try:
                existing_created = json.loads(path.read_text(encoding="utf-8")).get(
                    "meta", {}
                ).get("created")
            except (OSError, json.JSONDecodeError):
                existing_created = None
        payload["meta"]["created"] = existing_created or payload["meta"]["saved"]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _autosave_locked(self) -> None:
        if self.campaign_id:
            self._write_campaign_locked()

    def embedding_cache_path(self) -> Path:
        fragment = _safe_cache_fragment(self.config["embedding_model"])
        return self.cache_dir / f"boicl_embeddings_{fragment}.csv"

    def embedding_cache_status(self) -> Dict[str, Any]:
        procedures = [candidate["procedure"] for candidate in self.candidates]
        total = len(procedures)
        cached = 0
        path = self.embedding_cache_path()
        if total and path.exists():
            try:
                cache = pd.read_csv(path, usecols=["x", "embedding_model"])
                model_cache = cache[cache["embedding_model"] == self.config["embedding_model"]]
                cached_texts = set(model_cache["x"].astype(str).tolist())
                cached = sum(1 for procedure in procedures if procedure in cached_texts)
            except (OSError, ValueError, KeyError, pd.errors.ParserError):
                cached = 0
        return {
            "model": self.config["embedding_model"],
            "path": str(path),
            "cached_count": cached,
            "total_count": total,
            "missing_count": max(0, total - cached),
            "ready": total > 0 and cached >= total,
        }

    def key_status(self) -> Dict[str, Any]:
        key = os.environ.get("OPENAI_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return {
            "openai_configured": bool(key),
            "openrouter_configured": bool(openrouter_key),
            "anthropic_configured": bool(anthropic_key),
            "openai_hint": f"...{key[-4:]}" if len(key) > 8 else "",
        }

    def public_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": candidate["id"],
            "row": candidate["row"],
            "procedure": candidate["procedure"],
        }

    def to_json(self) -> Dict[str, Any]:
        with self.lock:
            active_observations = self.active_observations()
            available_candidates = self.available_candidates()
            labelled_values = self.candidate_objective_values()
            plot_steps = self.plot_horizon(len(active_observations))
            random_baseline = (
                float(np.mean(labelled_values))
                if self.config.get("benchmark_starting_baseline") == "mean"
                and labelled_values
                else None
            )
            random_steps = plot_steps - 1 if random_baseline is not None else plot_steps
            return {
                "config": self.config,
                "campaign": {
                    "id": self.campaign_id,
                    "name": self.campaign_name,
                    "saved": bool(self.campaign_id),
                },
                "campaigns": self.list_campaigns(),
                "objective_names": self.objective_names,
                "key_status": self.key_status(),
                "candidates": [
                    self.public_candidate(candidate) for candidate in self.candidates[:500]
                ],
                "candidate_count": len(self.candidates),
                "label_count": len(labelled_values),
                "available_candidates": [
                    self.public_candidate(candidate)
                    for candidate in available_candidates[:500]
                ],
                "available_count": len(available_candidates),
                "observations": self.observations,
                "suggestions": self.suggestions,
                "inverse_designs": self.inverse_designs,
                "benchmark_runs": self.benchmark_runs,
                "events": self.events,
                "last_error": self.last_error,
                "last_model_status": self.last_model_status,
                "progress": self.progress,
                "best_trace": _best_trace(
                    active_observations, self.config["objective_direction"]
                ),
                "random_walk_trace": _paper_random_trace(
                    labelled_values,
                    self.config["objective_direction"],
                    random_steps,
                    baseline_value=random_baseline,
                ),
                "dataset_stats": _dataset_stats(
                    labelled_values, self.config["objective_direction"]
                ),
                "acquisition_functions": ACQUISITION_FUNCTIONS,
                "model_presets": MODEL_PRESETS,
                "embedding_model_presets": EMBEDDING_MODEL_PRESETS,
                "embedding_cache": self.embedding_cache_status(),
                "scaling_modes": SCALING_MODES,
                "plot_stat_guides": PLOT_STAT_GUIDES,
            }

    def plot_horizon(self, live_count: int = 0) -> int:
        horizon = max(
            live_count,
            int(self.config["benchmark_initial_points"])
            + int(self.config["benchmark_iterations"]),
        )
        for run in self.benchmark_runs:
            for point in run.get("summary", []):
                horizon = max(horizon, int(point.get("index", 0)))
        return horizon

    def labelled_candidates(self, objective: Optional[str] = None) -> List[Dict[str, Any]]:
        objective = objective or self.config["objective_name"]
        return [
            cand
            for cand in self.candidates
            if cand.get("objectives", {}).get(objective) is not None
        ]

    def candidate_objective_values(self, objective: Optional[str] = None) -> List[float]:
        objective = objective or self.config["objective_name"]
        return [
            float(cand["objectives"][objective])
            for cand in self.labelled_candidates(objective)
        ]

    def prediction_system_message(self) -> str:
        return (
            self.config.get("prediction_system_message")
            or DEFAULT_PREDICTION_SYSTEM_MESSAGE
        )

    def inverse_system_message(self) -> str:
        return self.config.get("inverse_system_message") or DEFAULT_INVERSE_SYSTEM_MESSAGE

    def _refresh_dataset_prompts_locked(self, force: bool = False) -> bool:
        if not self.candidates:
            if force:
                raise ValueError("Import a dataset before generating dataset prompts.")
            return False
        changed = False
        if force or _is_auto_prediction_prompt(
            self.config.get("prediction_system_message")
        ):
            self.config["prediction_system_message"] = _dataset_prediction_prompt(
                self.candidates, self.objective_names
            )
            changed = True
        if force or _is_auto_inverse_prompt(self.config.get("inverse_system_message")):
            self.config["inverse_system_message"] = _dataset_inverse_prompt(
                self.candidates, self.objective_names
            )
            changed = True
        if changed:
            self.log("Generated dataset-specific BO-ICL system messages.")
        return changed

    def regenerate_prompts(self) -> Dict[str, Any]:
        with self.lock:
            self._refresh_dataset_prompts_locked(force=True)
            self._autosave_locked()
            return self.to_json()

    def _embedding_cache_model(self):
        from boicl import AskTellGPR

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = (
            str(self.embedding_cache_path()),
            self.config["embedding_model"],
            self.config["objective_name"],
        )
        if self._embedding_cache_model_obj is not None and key == self._embedding_cache_model_key:
            return self._embedding_cache_model_obj
        self._embedding_cache_model_obj = AskTellGPR(
            cache_path=str(self.embedding_cache_path()),
            embedding_model=self.config["embedding_model"],
            n_neighbors=1,
            n_components=1,
            y_name=self.config["objective_name"],
        )
        self._embedding_cache_model_key = key
        return self._embedding_cache_model_obj

    def _cached_embeddings(
        self, texts: List[str], progress_label: str = "Preparing embeddings"
    ) -> List[List[float]]:
        if not texts:
            return []
        model = self._embedding_cache_model()
        normalized_texts = [str(text) for text in texts]
        unique_texts = list(dict.fromkeys(text for text in normalized_texts if text.strip()))
        model_cache = model._embeddings_cache[
            model._embeddings_cache["embedding_model"] == self.config["embedding_model"]
        ]
        cached_by_text = {
            str(row["x"]): row["embedding"]
            for _, row in model_cache.iterrows()
        }
        missing = [text for text in unique_texts if text not in cached_by_text]
        if missing:
            client = OpenAI()
            batch_size = 25
            self.set_progress(
                progress_label,
                0,
                len(missing),
                detail=f"Embedding new texts with {self.config['embedding_model']}",
            )
            rows = []
            for start in range(0, len(missing), batch_size):
                self.check_cancelled()
                batch = missing[start : start + batch_size]
                response = self._api_call_with_retries(
                    progress_label,
                    lambda batch=batch: client.embeddings.create(
                        input=batch,
                        model=self.config["embedding_model"],
                        encoding_format="float",
                    ),
                )
                rows.extend(
                    {
                        "x": text,
                        "embedding": data.embedding,
                        "embedding_model": self.config["embedding_model"],
                    }
                    for text, data in zip(batch, response.data)
                )
                self.set_progress(
                    progress_label,
                    min(start + len(batch), len(missing)),
                    len(missing),
                    detail=f"Cached {min(start + len(batch), len(missing))} new embedding(s)",
                )
                self.check_cancelled()
            if rows:
                model._embeddings_cache = pd.concat(
                    [model._embeddings_cache, pd.DataFrame(rows)],
                    ignore_index=True,
                )
                model.save_cache(str(self.embedding_cache_path()))
                for row in rows:
                    cached_by_text[str(row["x"])] = row["embedding"]
        embeddings = []
        for text in normalized_texts:
            if text not in cached_by_text:
                raise ValueError(f"Embedding for '{text}' was not found in the local cache.")
            embeddings.append(cached_by_text[text])
        return embeddings

    def precompute_embeddings(self) -> Dict[str, Any]:
        with self.lock:
            self.cancel_event.clear()
            if not self.candidates:
                raise ValueError("Import a dataset before preparing embeddings.")
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required to prepare embeddings.")
            procedures = [candidate["procedure"] for candidate in self.candidates]
            before = self.embedding_cache_status()["cached_count"]
            try:
                self._cached_embeddings(procedures, "Preparing dataset embeddings")
            except RunCancelled:
                self.last_model_status = "Embedding preparation stopped by user."
                self.log("Stopped embedding preparation.")
                self.cancel_progress(
                    "Preparing dataset embeddings",
                    "Stopped before all requested embeddings were prepared.",
                )
                self._autosave_locked()
                return self.to_json()
            after_status = self.embedding_cache_status()
            added = after_status["cached_count"] - before
            self.last_model_status = (
                f"Prepared embeddings for {after_status['cached_count']} of "
                f"{after_status['total_count']} candidates."
            )
            self.log(
                f"Prepared {max(0, added)} new embedding(s) using "
                f"{self.config['embedding_model']}."
            )
            self.finish_progress(
                "Preparing dataset embeddings",
                f"{after_status['cached_count']} of {after_status['total_count']} cached.",
            )
            self._autosave_locked()
            return self.to_json()

    def save_campaign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            name = (payload.get("name") or self.campaign_name or "").strip()
            if not name:
                name = f"BO campaign {time.strftime('%Y-%m-%d %H:%M')}"
            save_as = bool(payload.get("save_as"))
            if save_as or not self.campaign_id:
                self.campaign_id = self._unique_campaign_id(name)
            self.campaign_name = name
            self._write_campaign_locked()
            self.last_model_status = f"Saved campaign: {name}."
            self.log(f"Saved campaign '{name}'.")
            return self.to_json()

    def load_campaign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        campaign_id = _safe_cache_fragment(str(payload.get("id") or "").lower())
        if not campaign_id:
            raise ValueError("Choose a saved campaign to load.")
        path = self.campaign_file(campaign_id).resolve()
        root = self.campaigns_dir.resolve()
        if root not in path.parents or not path.exists():
            raise ValueError("Saved campaign was not found.")
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        with self.lock:
            self.campaign_id = meta.get("id") or campaign_id
            self.campaign_name = meta.get("name") or campaign_id
            self.config = dict(DEFAULT_CONFIG)
            self.config.update(data.get("config", {}))
            self.objective_names = list(data.get("objective_names") or ["objective"])
            self.candidates = list(data.get("candidates") or [])
            self.observations = list(data.get("observations") or [])
            self.suggestions = list(data.get("suggestions") or [])
            self.inverse_designs = list(data.get("inverse_designs") or [])
            self.benchmark_runs = list(data.get("benchmark_runs") or [])
            self.progress = data.get("progress") or self.progress
            if self.progress.get("status") == "running":
                partial = self.progress.get("partial_run")
                if partial and partial.get("replicate_observations"):
                    partial["partial"] = True
                    partial["status"] = "interrupted"
                    self._upsert_benchmark_run_locked(partial)
                self.finish_progress("Loaded campaign", "Previous run was interrupted.")
            self.events = list(data.get("events") or [])[:20]
            self.last_error = None
            self.last_model_status = (
                data.get("last_model_status")
                or f"Loaded campaign: {self.campaign_name}."
            )
            self.refresh_active_objective()
            self.log(f"Loaded campaign '{self.campaign_name}'.")
            return self.to_json()

    def delete_campaign(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        campaign_id = _safe_cache_fragment(str(payload.get("id") or "").lower())
        if not campaign_id:
            raise ValueError("Choose a saved campaign to delete.")
        path = self.campaign_file(campaign_id).resolve()
        root = self.campaigns_dir.resolve()
        if root not in path.parents or not path.exists():
            raise ValueError("Saved campaign was not found.")
        with self.lock:
            path.unlink()
            try:
                path.parent.rmdir()
            except OSError:
                pass
            if self.campaign_id == campaign_id:
                self.campaign_id = None
                self.campaign_name = ""
            self.last_model_status = "Deleted saved campaign."
            self.log("Deleted saved campaign.")
            return self.to_json()

    def active_observations(self) -> List[Dict[str, Any]]:
        objective = self.config["objective_name"]
        return [
            obs
            for obs in self.observations
            if obs.get("objectives", {}).get(objective) is not None
        ]

    def refresh_active_objective(self) -> None:
        objective = self.config["objective_name"]
        for obs in self.observations:
            value = obs.get("objectives", {}).get(objective)
            uncertainty = obs.get("uncertainties", {}).get(objective)
            obs["value"] = value
            obs["uncertainty"] = uncertainty
            obs["target"] = self.target_value(value) if value is not None else None

    def available_candidates(self) -> List[Dict[str, Any]]:
        max_replicates = max(1, int(self.config["replicates_per_candidate"]))
        counts: Dict[str, int] = {}
        for obs in self.active_observations():
            key = obs.get("candidate_id") or obs["procedure"]
            counts[key] = counts.get(key, 0) + 1
        available = []
        for cand in self.candidates:
            count = counts.get(cand["id"], counts.get(cand["procedure"], 0))
            if count < max_replicates:
                available.append(cand)
        return available

    def target_value(self, value: float) -> float:
        if value is None:
            return None
        return _target_value(value, self.config["objective_direction"])

    def display_value(self, value: float) -> float:
        if value is None:
            return None
        return _display_value(value, self.config["objective_direction"])

    def import_dataset(
        self, filename: str, raw: bytes, objective_name: Optional[str] = None
    ) -> Dict[str, Any]:
        df = _read_table_from_bytes(filename, raw)
        _clean_procedures(df)
        first_col = df.columns[0]
        data_columns = list(df.columns[1:])
        requested_objective = (objective_name or "").strip()
        value_columns = _numeric_columns(
            df, [column for column in data_columns if not _is_uncertainty_column(column)]
        )
        uncertainty_columns = _numeric_columns(
            df, [column for column in data_columns if _is_uncertainty_column(column)]
        )
        uncertainty_by_objective: Dict[str, Any] = {}
        for uncertainty_column in uncertainty_columns:
            uncertainty_name = str(uncertainty_column).lower()
            matches = [
                column
                for column in value_columns
                if str(column).lower() in uncertainty_name
            ]
            if len(matches) == 1:
                uncertainty_by_objective[str(matches[0])] = uncertainty_column
            elif len(value_columns) == 1:
                uncertainty_by_objective[str(value_columns[0])] = uncertainty_column

        with self.lock:
            self.candidates = []
            self.observations = []
            self.suggestions = []
            self.inverse_designs = []
            self.benchmark_runs = []
            self.last_error = None
            self.last_model_status = (
                "Dataset loaded. Add live observations or run an offline benchmark."
            )
            if value_columns:
                self.objective_names = [str(column) for column in value_columns]
                if self.config["objective_name"] not in self.objective_names:
                    self.config["objective_name"] = self.objective_names[0]
            else:
                if requested_objective:
                    self.config["objective_name"] = requested_objective
                self.objective_names = [self.config["objective_name"]]

            for row_index, row in df.iterrows():
                if pd.isna(row[first_col]) or not str(row[first_col]).strip():
                    continue
                procedure = str(row[first_col]).strip()
                objectives = {
                    str(column): _coerce_float(row[column]) for column in value_columns
                }
                objectives = {
                    key: value for key, value in objectives.items() if value is not None
                }
                uncertainties = {
                    objective: _coerce_float(row[uncertainty_column])
                    for objective, uncertainty_column in uncertainty_by_objective.items()
                }
                uncertainties = {
                    key: value for key, value in uncertainties.items() if value is not None
                }
                candidate = {
                    "id": f"cand-{len(self.candidates)}",
                    "row": int(row_index) + 1,
                    "procedure": procedure,
                    "objectives": {
                        key: float(value) for key, value in objectives.items()
                    },
                    "uncertainties": {
                        key: float(value) for key, value in uncertainties.items()
                    },
                }
                self.candidates.append(candidate)
            self.refresh_active_objective()
            self._refresh_dataset_prompts_locked()
            self.log(
                f"Imported {len(self.candidates)} candidates from {Path(filename).name}."
            )
            labelled = len(self.labelled_candidates())
            if labelled:
                self.config["workflow_mode"] = "offline"
                self.log(f"Loaded {labelled} hidden labels for offline benchmarks.")
            else:
                self.config["workflow_mode"] = "live"
            self._autosave_locked()
            return self.to_json()

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            for key in DEFAULT_CONFIG:
                if key not in payload:
                    continue
                value = payload[key]
                if key in {
                    "batch_size",
                    "iterations_per_trial",
                    "replicates_per_candidate",
                    "benchmark_iterations",
                    "benchmark_replicates",
                    "benchmark_initial_points",
                    "benchmark_seed",
                    "random_replicates",
                    "llm_samples",
                    "selector_k",
                    "inverse_filter",
                    "inverse_random_candidates",
                    "inverse_design_count",
                    "score_limit",
                    "api_retry_attempts",
                    "n_neighbors",
                    "n_components",
                }:
                    value = int(value)
                elif key == "ucb_lambda":
                    value = float(value)
                elif key in {
                    "inverse_target_multiplier",
                    "inverse_target_jitter",
                    "api_pause_seconds",
                    "api_rate_limit_cooldown_seconds",
                }:
                    value = float(value)
                elif key in {"auto_suggest", "greedy_final_iteration"}:
                    value = bool(value)
                self.config[key] = value
            self.config["optimizer"] = (
                "llm" if self.config["optimizer"] == "llm" else "gpr"
            )
            self.config["workflow_mode"] = (
                "offline"
                if self.config["workflow_mode"] == "offline"
                else "live"
            )
            if self.config["acquisition"] not in ACQUISITION_FUNCTIONS:
                self.config["acquisition"] = DEFAULT_CONFIG["acquisition"]
            self.config["objective_direction"] = (
                "minimize"
                if self.config["objective_direction"] == "minimize"
                else "maximize"
            )
            if self.config["objective_scaling"] not in SCALING_MODES:
                self.config["objective_scaling"] = "off"
            if self.config["plot_stat_guides"] not in PLOT_STAT_GUIDES:
                self.config["plot_stat_guides"] = "max"
            if self.config["benchmark_starting_baseline"] not in BENCHMARK_STARTING_BASELINES:
                self.config["benchmark_starting_baseline"] = "none"
            self.config["batch_size"] = max(1, min(25, int(self.config["batch_size"])))
            self.config["iterations_per_trial"] = max(
                0, int(self.config["iterations_per_trial"])
            )
            self.config["replicates_per_candidate"] = max(
                1, int(self.config["replicates_per_candidate"])
            )
            self.config["benchmark_iterations"] = max(
                1, int(self.config["benchmark_iterations"])
            )
            self.config["benchmark_replicates"] = max(
                1, min(50, int(self.config["benchmark_replicates"]))
            )
            self.config["benchmark_initial_points"] = max(
                1, int(self.config["benchmark_initial_points"])
            )
            self.config["benchmark_seed"] = int(self.config["benchmark_seed"])
            self.config["random_replicates"] = max(0, int(self.config["random_replicates"]))
            self.config["llm_samples"] = max(1, min(20, int(self.config["llm_samples"])))
            self.config["selector_k"] = max(0, int(self.config["selector_k"]))
            self.config["inverse_filter"] = max(0, int(self.config["inverse_filter"]))
            self.config["inverse_random_candidates"] = max(
                0, int(self.config["inverse_random_candidates"])
            )
            self.config["inverse_design_count"] = max(
                1, min(10, int(self.config["inverse_design_count"]))
            )
            self.config["inverse_target_jitter"] = max(
                0.0, min(1.0, float(self.config["inverse_target_jitter"]))
            )
            self.config["score_limit"] = max(1, int(self.config["score_limit"]))
            self.config["api_retry_attempts"] = max(
                1, min(20, int(self.config["api_retry_attempts"]))
            )
            self.config["api_pause_seconds"] = max(
                0.0, min(120.0, float(self.config["api_pause_seconds"]))
            )
            self.config["api_rate_limit_cooldown_seconds"] = max(
                0.0,
                min(300.0, float(self.config["api_rate_limit_cooldown_seconds"])),
            )
            self.config["n_neighbors"] = max(1, int(self.config["n_neighbors"]))
            self.config["n_components"] = max(1, int(self.config["n_components"]))
            if self.config["objective_name"] not in self.objective_names:
                self.objective_names.append(self.config["objective_name"])
            self.refresh_active_objective()
            self.log("Updated run settings.")
            self._autosave_locked()
            return self.to_json()

    def add_observation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate_id = payload.get("candidate_id") or None
        procedure = (payload.get("procedure") or "").strip()
        if candidate_id:
            match = next((cand for cand in self.candidates if cand["id"] == candidate_id), None)
            if match is None:
                raise ValueError("Selected candidate was not found.")
            procedure = match["procedure"]
        if not procedure:
            raise ValueError("Choose or enter a procedure.")
        value = _coerce_float(payload.get("value"))
        if value is None:
            raise ValueError("Objective value must be numeric.")
        uncertainty = _coerce_float(payload.get("uncertainty"))
        with self.lock:
            objective = self.config["objective_name"]
            if objective not in self.objective_names:
                self.objective_names.append(objective)
            self._add_observation_locked(
                procedure,
                {objective: value},
                {objective: uncertainty} if uncertainty is not None else {},
                candidate_id,
            )
            self.refresh_active_objective()
            self.suggestions = []
            self.last_error = None
            self.last_model_status = "Observation added. Suggestions need an update."
            self.log(f"Added objective value {value:g}.")
            self._autosave_locked()
            return self.to_json()

    def _add_observation_locked(
        self,
        procedure: str,
        objectives: Dict[str, float],
        uncertainties: Dict[str, float],
        candidate_id: Optional[str] = None,
    ) -> None:
        if candidate_id is None:
            match = next(
                (cand for cand in self.candidates if cand["procedure"] == procedure),
                None,
            )
            candidate_id = match["id"] if match else None
        objective = self.config["objective_name"]
        value = objectives.get(objective)
        uncertainty = uncertainties.get(objective)
        self.observations.append(
            {
                "id": f"obs-{len(self.observations) + 1}",
                "candidate_id": candidate_id,
                "procedure": procedure,
                "objectives": {key: float(val) for key, val in objectives.items()},
                "uncertainties": {
                    key: float(val) for key, val in uncertainties.items() if val is not None
                },
                "value": float(value) if value is not None else None,
                "target": float(self.target_value(value)) if value is not None else None,
                "uncertainty": uncertainty,
                "time": _now(),
            }
        )

    def suggest(self) -> Dict[str, Any]:
        with self.lock:
            self.cancel_event.clear()
            self.last_error = None
            self.set_progress(
                "Updating suggestions",
                0,
                1,
                detail=f"{self.config['optimizer'].upper()} / {self.config['acquisition']}",
            )
            active_observations = self.active_observations()
            iteration_cap = int(self.config["iterations_per_trial"])
            if iteration_cap and len(active_observations) >= iteration_cap:
                self.suggestions = []
                self.last_model_status = "Iteration cap reached."
                self.finish_progress("Updating suggestions", "Iteration cap reached.")
                self._autosave_locked()
                return self.to_json()
            available = self.available_candidates()
            if not available:
                self.suggestions = []
                self.last_model_status = "No unevaluated candidates remain."
                self.finish_progress("Updating suggestions", "No unevaluated candidates remain.")
                self._autosave_locked()
                return self.to_json()
            missing_key = self._missing_key_for_suggestions()
            if missing_key:
                self.suggestions = self._random_suggestions(available)
                self.last_model_status = f"{missing_key} missing. Showing random candidates."
                self.log(f"Add {missing_key} to enable {self.config['optimizer'].upper()} suggestions.")
                self.finish_progress("Updating suggestions", f"{missing_key} missing.")
                self._autosave_locked()
                return self.to_json()
            if len(active_observations) < 2:
                self.suggestions = self._random_suggestions(available)
                self.last_model_status = "Add at least 2 observations before model suggestions."
                self.finish_progress(
                    "Updating suggestions", "Showing random candidates until 2 observations."
                )
                self._autosave_locked()
                return self.to_json()
            if len(self._training_observations()) < 2:
                self.suggestions = self._random_suggestions(available)
                self.last_model_status = "Add at least 2 unique procedures before model suggestions."
                self.finish_progress(
                    "Updating suggestions", "Showing random candidates until 2 unique procedures."
                )
                self._autosave_locked()
                return self.to_json()

            try:
                self.check_cancelled()
                self.set_progress(
                    "Updating suggestions",
                    0,
                    1,
                    detail=(
                        f"Scoring up to "
                        f"{self._llm_scored_candidate_count(len(available)) if self.config['optimizer'] == 'llm' else min(self.config['score_limit'], len(available))} "
                        f"candidate(s)"
                        + (
                            f" x {self.config['llm_samples']} LLM sample(s) "
                            f"from a {min(self.config['score_limit'], len(available))}-candidate broad pool"
                            if self.config["optimizer"] == "llm"
                            else ""
                        )
                    ),
                )
                if self.config["optimizer"] == "llm":
                    self.suggestions = self._llm_suggestions(available)
                else:
                    self.suggestions = self._gpr_suggestions(available)
                self.check_cancelled()
                self.last_model_status = (
                    f"Updated {self.config['optimizer'].upper()} on {len(active_observations)} observations."
                )
                self.log(f"Updated suggestions with {self.config['acquisition']}.")
                self.finish_progress(
                    "Updating suggestions",
                    f"Generated {len(self.suggestions)} suggestion(s).",
                )
            except RunCancelled as exc:
                self.suggestions = []
                self.last_error = None
                self.last_model_status = str(exc)
                self.log("Stopped suggestion update.")
                self.cancel_progress("Updating suggestions", str(exc))
            except Exception as exc:  # pragma: no cover - needs live API/GPR deps
                self.suggestions = self._random_suggestions(available)
                self.last_error = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.last_model_status = "Model update failed. Showing random candidates."
                self.log("Model update failed; see status panel.")
                self.fail_progress("Updating suggestions", self.last_error)
            self._autosave_locked()
            return self.to_json()

    def _missing_key_for_suggestions(self) -> Optional[str]:
        if self.config["optimizer"] == "gpr":
            return None if os.environ.get("OPENAI_API_KEY") else "OPENAI_API_KEY"
        key_name = _required_key_name(self.config["prediction_model"])
        if not os.environ.get(key_name):
            return key_name
        if self.config["selector_k"] and not os.environ.get("OPENAI_API_KEY"):
            return "OPENAI_API_KEY"
        if self.config["inverse_filter"]:
            inverse_key_name = _required_key_name(self.config["inverse_model"])
            if not os.environ.get(inverse_key_name):
                return inverse_key_name
            if not os.environ.get("OPENAI_API_KEY"):
                return "OPENAI_API_KEY"
        return None

    def _random_suggestions(self, available: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch_size = min(self.config["batch_size"], len(available))
        sampled = random.sample(available, batch_size)
        return [
            {
                "candidate_id": cand["id"],
                "procedure": cand["procedure"],
                "acquisition": 0.0,
                "mean": None,
                "std": None,
                "source": "random",
            }
            for cand in sampled
        ]

    def _candidate_subset(
        self, available: List[Dict[str, Any]], rng: Optional[random.Random] = None
    ) -> List[Dict[str, Any]]:
        score_limit = min(self.config["score_limit"], len(available))
        if score_limit == len(available):
            return list(available)
        return (rng or random).sample(available, score_limit)

    def _llm_scored_candidate_count(self, available_count: int) -> int:
        broad = min(int(self.config["score_limit"]), int(available_count))
        inverse_count = int(self.config["inverse_filter"])
        if inverse_count <= 0:
            return broad
        filtered = min(inverse_count, broad)
        random_addons = min(
            int(self.config["inverse_random_candidates"]), max(0, broad - filtered)
        )
        return filtered + random_addons

    def _gpr_suggestions(
        self,
        available: List[Dict[str, Any]],
        observations: Optional[List[Dict[str, Any]]] = None,
        rng: Optional[random.Random] = None,
        k: Optional[int] = None,
        acquisition: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from boicl import AskTellGPR

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        active_observations, scaler = self._training_rows_and_scaler(observations)
        n_obs = len(active_observations)
        n_neighbors = min(self.config["n_neighbors"], max(1, n_obs - 1))
        n_components = min(self.config["n_components"], max(1, n_obs - 1))
        subset = self._candidate_subset(available, rng)
        procedures = [cand["procedure"] for cand in subset]
        self.check_cancelled()
        self._cached_embeddings(
            [obs["procedure"] for obs in active_observations] + procedures,
            "Preparing embeddings for GPR",
        )
        self.check_cancelled()
        model = AskTellGPR(
            cache_path=str(self.embedding_cache_path()),
            embedding_model=self.config["embedding_model"],
            n_neighbors=n_neighbors,
            n_components=n_components,
            y_name=self.config["objective_name"],
            y_formatter=lambda y: f"{float(y):0.6g}",
        )

        for obs in active_observations[:-1]:
            model.tell(obs["procedure"], obs["target"], train=False)
        model.tell(
            active_observations[-1]["procedure"],
            active_observations[-1]["target"],
            train=True,
        )

        raw = model.ask(
            procedures,
            aq_fxn=acquisition or self.config["acquisition"],
            k=min(k or self.config["batch_size"], len(procedures)),
            aug_random_filter=len(procedures),
            _lambda=self.config["ucb_lambda"],
        )
        self.check_cancelled()
        if hasattr(model, "save_cache"):
            model.save_cache(str(self.embedding_cache_path()))

        selected, acquisition, means = raw[:3]
        stds = raw[3] if len(raw) > 3 else [None] * len(selected)
        by_proc = {cand["procedure"]: cand for cand in subset}
        return [
            {
                "candidate_id": by_proc[procedure]["id"],
                "procedure": procedure,
                "acquisition": float(aq),
                "mean": _unscale_target(
                    float(mean), self.config["objective_direction"], scaler
                )
                if mean is not None
                else None,
                "std": float(std) if std is not None else None,
                "source": "gpr",
            }
            for procedure, aq, mean, std in zip(selected, acquisition, means, stds)
        ]

    def _build_llm_model(
        self, observations: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, Dict[str, float]]:
        from boicl import AskTellFewShotTopk

        selector_k = int(self.config["selector_k"]) or None
        active_observations, scaler = self._training_rows_and_scaler(observations)
        model = AskTellFewShotTopk(
            model=self.config["prediction_model"],
            inverse_model=self.config["inverse_model"],
            k=int(self.config["llm_samples"]),
            selector_k=selector_k,
            y_name=self.config["objective_name"],
            x_name="procedure",
            y_formatter=lambda y: f"{float(y):0.6g}",
        )
        for obs in active_observations:
            model.tell(obs["procedure"], obs["target"])
        return model, scaler

    def _llm_suggestions(
        self,
        available: List[Dict[str, Any]],
        observations: Optional[List[Dict[str, Any]]] = None,
        rng: Optional[random.Random] = None,
        k: Optional[int] = None,
        acquisition: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        model, scaler = self._build_llm_model(observations)
        subset = self._candidate_subset(available, rng)
        procedures = [cand["procedure"] for cand in subset]
        inverse_text = None
        target_observations = (
            observations if observations is not None else self.active_observations()
        )
        if self.config["inverse_filter"] and procedures:
            self.check_cancelled()
            inverse_target = self._inverse_target_display_value(
                target_observations, rng=rng
            )
            inverse_text = self._api_call_with_retries(
                "Generating inverse-filter query",
                lambda: self._generate_inverse_text(
                    model,
                    scaler,
                    target_observations,
                    target_value=inverse_target,
                ),
            )
            self.check_cancelled()
            filtered = self._cached_approx_sample(
                procedures,
                inverse_text,
                min(int(self.config["inverse_filter"]), len(procedures)),
            )
            remaining = [procedure for procedure in procedures if procedure not in filtered]
            random_count = min(int(self.config["inverse_random_candidates"]), len(remaining))
            procedures = filtered + (rng or random).sample(remaining, random_count)
            self.inverse_designs.insert(
                0,
                {
                    "procedure": inverse_text,
                    "target": inverse_target,
                    "model": self.config["inverse_model"],
                    "time": _now(),
                    "source": "inverse_filter",
                },
            )
            self.inverse_designs = self.inverse_designs[:20]
        if not procedures:
            return self._random_suggestions(available)

        self.check_cancelled()
        raw = self._api_call_with_retries(
            "Scoring LLM candidate shortlist",
            lambda: model.ask(
                procedures,
                aq_fxn=acquisition or self.config["acquisition"],
                k=min(k or self.config["batch_size"], len(procedures)),
                inv_filter=0,
                aug_random_filter=len(procedures),
                _lambda=self.config["ucb_lambda"],
                system_message=self.prediction_system_message(),
                inv_system_message=self.inverse_system_message(),
            ),
        )
        self.check_cancelled()
        selected, acquisition, means = raw[:3]
        by_proc = {cand["procedure"]: cand for cand in subset}
        return [
            {
                "candidate_id": by_proc[procedure]["id"],
                "procedure": procedure,
                "acquisition": float(aq),
                "mean": _unscale_target(
                    float(mean), self.config["objective_direction"], scaler
                )
                if mean is not None
                else None,
                "std": None,
                "source": "llm",
                "prediction_model": self.config["prediction_model"],
                "inverse_model": self.config["inverse_model"],
                "inverse_seed": inverse_text,
            }
            for procedure, aq, mean in zip(selected, acquisition, means)
            if procedure in by_proc
        ]

    def _cached_approx_sample(
        self, procedures: List[str], query: str, k: int, lambda_mult: float = 0.5
    ) -> List[str]:
        if k <= 0 or not procedures:
            return []
        embeddings = np.array(
            self._cached_embeddings(
                [*procedures, query],
                "Embedding inverse-filter query",
            ),
            dtype=float,
        )
        candidate_vectors = embeddings[: len(procedures)]
        query_vector = embeddings[-1]
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denom = np.maximum(candidate_norms * query_norm, 1e-12)
        scores = (candidate_vectors @ query_vector) / denom
        if k >= len(procedures):
            return [procedures[index] for index in np.argsort(scores)[::-1]]

        selected: List[int] = []
        remaining = set(range(len(procedures)))
        while remaining and len(selected) < k:
            if not selected:
                chosen = max(remaining, key=lambda index: scores[index])
            else:
                selected_vectors = candidate_vectors[selected]
                selected_norms = np.maximum(candidate_norms[selected], 1e-12)

                def mmr_score(index: int) -> float:
                    similarities = (
                        selected_vectors @ candidate_vectors[index]
                    ) / np.maximum(selected_norms * candidate_norms[index], 1e-12)
                    diversity_penalty = float(np.max(similarities))
                    return (
                        lambda_mult * float(scores[index])
                        - (1.0 - lambda_mult) * diversity_penalty
                    )

                chosen = max(remaining, key=mmr_score)
            selected.append(chosen)
            remaining.remove(chosen)
        return [procedures[index] for index in selected]

    def _inverse_target_display_value(
        self,
        observations: Optional[List[Dict[str, Any]]] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        configured = _coerce_float(self.config.get("inverse_target_value"))
        if configured is not None:
            return configured
        observations = observations if observations is not None else self.active_observations()
        if not observations:
            raise ValueError(
                "Inverse design needs live observations or an explicit inverse target. "
                "For a fully labeled dataset, use Offline Benchmark > Run & Append "
                "instead of Generate Proposals."
            )
        values = [obs["value"] for obs in observations if obs["value"] is not None]
        best = min(values) if self.config["objective_direction"] == "minimize" else max(values)
        multiplier = float(self.config["inverse_target_multiplier"])
        jitter = float(self.config.get("inverse_target_jitter") or 0.0)
        if jitter > 0:
            sampler = rng if rng is not None else random
            multiplier = max(0.0, sampler.normalvariate(multiplier, jitter))
        target = best * multiplier
        floor_value = _coerce_float(self.config.get("inverse_target_floor_value"))
        if floor_value is not None:
            if self.config["objective_direction"] == "minimize":
                target = min(target, floor_value)
            else:
                target = max(target, floor_value)
        return target

    def _inverse_target_model_value(
        self,
        scaler: Optional[Dict[str, float]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        target_value: Optional[float] = None,
    ) -> float:
        if scaler is None:
            _, scaler = self._training_rows_and_scaler(observations)
        return _scale_target(
            target_value
            if target_value is not None
            else self._inverse_target_display_value(observations),
            self.config["objective_direction"],
            scaler,
        )

    def _generate_inverse_text(
        self,
        model,
        scaler: Optional[Dict[str, float]] = None,
        observations: Optional[List[Dict[str, Any]]] = None,
        target_value: Optional[float] = None,
    ) -> str:
        return model.inv_predict(
            self._inverse_target_model_value(scaler, observations, target_value),
            system_message=self.inverse_system_message(),
        )

    def generate_inverse_designs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        target = payload.get("target_value")
        count = payload.get("count")
        with self.lock:
            self.cancel_event.clear()
            if target not in (None, ""):
                self.config["inverse_target_value"] = str(target)
            if count not in (None, ""):
                self.config["inverse_design_count"] = max(1, min(10, int(count)))
            key_name = _required_key_name(self.config["inverse_model"])
            if not os.environ.get(key_name):
                raise ValueError(f"{key_name} is required for inverse design.")
            if self.config["selector_k"] and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for selector examples.")
            if len(self._training_observations()) < 1:
                raise ValueError("Add at least one labeled procedure before inverse design.")
            model, scaler = self._build_llm_model()
            generated = []
            design_count = int(self.config["inverse_design_count"])
            self.set_progress(
                "Generating inverse designs",
                0,
                design_count,
                detail=f"{self.config['inverse_model']}",
            )
            try:
                for index in range(design_count):
                    self.check_cancelled()
                    inverse_target = self._inverse_target_display_value()
                    generated.append(
                        {
                            "procedure": self._api_call_with_retries(
                                "Generating inverse designs",
                                lambda: self._generate_inverse_text(
                                    model,
                                    scaler,
                                    target_value=inverse_target,
                                ),
                            ),
                            "target": inverse_target,
                            "model": self.config["inverse_model"],
                            "time": _now(),
                            "source": "manual_inverse_design",
                        }
                    )
                    self.set_progress(
                        "Generating inverse designs",
                        index + 1,
                        design_count,
                        detail=f"Generated {index + 1}/{design_count}",
                    )
                    self.check_cancelled()
            except RunCancelled as exc:
                self.last_model_status = str(exc)
                self.log("Stopped inverse design generation.")
                self.cancel_progress("Generating inverse designs", str(exc))
                self._autosave_locked()
                return self.to_json()
            self.inverse_designs = generated + self.inverse_designs
            self.inverse_designs = self.inverse_designs[:20]
            self.log(f"Generated {len(generated)} inverse design proposal(s).")
            self.finish_progress(
                "Generating inverse designs",
                f"Generated {len(generated)} proposal(s).",
            )
            self._autosave_locked()
            return self.to_json()

    def _observation_from_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        objective = self.config["objective_name"]
        value = float(candidate["objectives"][objective])
        uncertainty = candidate.get("uncertainties", {}).get(objective)
        return {
            "id": "",
            "candidate_id": candidate["id"],
            "procedure": candidate["procedure"],
            "objectives": {objective: value},
            "uncertainties": {objective: uncertainty} if uncertainty is not None else {},
            "value": value,
            "target": self.target_value(value),
            "uncertainty": uncertainty,
            "time": _now(),
        }

    def _missing_key_for_benchmark(self) -> Optional[str]:
        if (
            self.config["acquisition"] == "random"
            and not self.config["greedy_final_iteration"]
        ):
            return None
        return self._missing_key_for_suggestions()

    def _benchmark_name(self) -> str:
        if self.config["optimizer"] == "llm":
            model = self.config["prediction_model"]
        else:
            model = self.config["embedding_model"]
        return (
            f"{self.config['optimizer'].upper()} "
            f"{self.config['acquisition'].replace('_', ' ')} "
            f"({model})"
            + (" + greedy final" if self.config["greedy_final_iteration"] else "")
        )

    def _benchmark_config_snapshot(self) -> Dict[str, Any]:
        return {
            key: self.config[key]
            for key in [
                "optimizer",
                "objective_name",
                "objective_direction",
                "objective_scaling",
                "acquisition",
                "embedding_model",
                "prediction_model",
                "inverse_model",
                "llm_samples",
                "selector_k",
                "inverse_filter",
                "inverse_random_candidates",
                "inverse_target_multiplier",
                "inverse_target_jitter",
                "inverse_target_floor_value",
                "ucb_lambda",
                "score_limit",
                "api_pause_seconds",
                "api_retry_attempts",
                "api_rate_limit_cooldown_seconds",
                "n_neighbors",
                "n_components",
                "benchmark_iterations",
                "benchmark_replicates",
                "benchmark_initial_points",
                "benchmark_seed",
                "benchmark_starting_baseline",
                "greedy_final_iteration",
            ]
        }

    def _benchmark_next_candidate(
        self,
        available: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        rng: random.Random,
        acquisition: Optional[str] = None,
    ) -> Dict[str, Any]:
        acquisition = acquisition or self.config["acquisition"]
        training_rows, _ = self._training_rows_and_scaler(observations)
        if acquisition == "random" or len(training_rows) < 2:
            return rng.choice(available)
        if self.config["optimizer"] == "llm":
            suggestions = self._llm_suggestions(
                available, observations, rng=rng, k=1, acquisition=acquisition
            )
        else:
            suggestions = self._gpr_suggestions(
                available, observations, rng=rng, k=1, acquisition=acquisition
            )
        if not suggestions:
            return rng.choice(available)
        candidate_id = suggestions[0]["candidate_id"]
        return next(
            (candidate for candidate in available if candidate["id"] == candidate_id),
            rng.choice(available),
        )

    def _benchmark_baseline_value(self, labelled: List[Dict[str, Any]]) -> Optional[float]:
        if self.config.get("benchmark_starting_baseline") != "mean":
            return None
        objective = self.config["objective_name"]
        values = [
            float(candidate["objectives"][objective])
            for candidate in labelled
            if candidate.get("objectives", {}).get(objective) is not None
        ]
        return float(np.mean(values)) if values else None

    def _summarize_replicate_traces(
        self, replicate_traces: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if not replicate_traces:
            return []
        indexes = sorted(
            {
                int(point["index"])
                for trace in replicate_traces
                for point in trace
                if point.get("index") is not None
            }
        )
        summary = []
        for index in indexes:
            values = [
                point["best"]
                for trace in replicate_traces
                for point in trace
                if int(point.get("index", -1)) == index
            ]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values)) if len(values) > 1 else 0.0
            summary.append(
                {
                    "index": index,
                    "mean": mean,
                    "std": std,
                    "lower": mean - std,
                    "upper": mean + std,
                    "count": len(values),
                }
            )
        return summary

    def _upsert_benchmark_run_locked(
        self, run: Dict[str, Any], index: Optional[int] = None
    ) -> None:
        if index is not None:
            self.benchmark_runs[index] = run
            return
        run_id = run.get("id")
        for existing_index, existing in enumerate(self.benchmark_runs):
            if run_id and existing.get("id") == run_id:
                self.benchmark_runs[existing_index] = run
                return
        self.benchmark_runs.append(run)

    def run_benchmark(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.cancel_event.clear()
            self.last_error = None
            resume_id = str(payload.get("resume_id") or "").strip()
            resume_index = None
            resume_run = None
            if resume_id:
                for index, run in enumerate(self.benchmark_runs):
                    if run.get("id") == resume_id:
                        resume_index = index
                        resume_run = run
                        break
                if resume_run is None:
                    raise ValueError("The benchmark run to resume was not found.")
                if not resume_run.get("partial"):
                    raise ValueError("Only stopped or interrupted benchmark runs can be resumed.")
                if not resume_run.get("replicate_observations"):
                    raise ValueError(
                        "This benchmark was stopped before candidate-level rows were saved."
                    )
                saved_config = {
                    key: value
                    for key, value in (resume_run.get("config") or {}).items()
                    if key in self.config
                }
                self.config.update(saved_config)
                if "api_pause_seconds" in payload:
                    self.config["api_pause_seconds"] = float(payload["api_pause_seconds"])
                if "api_retry_attempts" in payload:
                    self.config["api_retry_attempts"] = int(payload["api_retry_attempts"])

            labelled = self.labelled_candidates()
            if not labelled:
                raise ValueError(
                    "Import a dataset with numeric labels before running an offline benchmark."
                )
            initial_points = min(
                int(self.config["benchmark_initial_points"]), len(labelled)
            )
            iterations = int(self.config["benchmark_iterations"])
            replicates = int(self.config["benchmark_replicates"])
            if len(labelled) <= initial_points:
                raise ValueError("Need more labelled candidates than initial random points.")
            missing_key = self._missing_key_for_benchmark()
            if missing_key:
                raise ValueError(f"{missing_key} is required for this benchmark config.")

            name = (
                (payload.get("name") or "").strip()
                or (resume_run or {}).get("name")
                or self._benchmark_name()
            )
            seed = int(self.config["benchmark_seed"])
            baseline_value = self._benchmark_baseline_value(labelled)
            trace_skip_count = initial_points if baseline_value is not None else 0
            replicate_observations: List[List[Dict[str, Any]]] = [
                [dict(obs) for obs in obs_list]
                for obs_list in ((resume_run or {}).get("replicate_observations") or [])
            ]
            replicate_traces: List[List[Dict[str, Any]]] = [
                _best_trace(
                    obs_list,
                    self.config["objective_direction"],
                    baseline_value=baseline_value,
                    skip_observations=trace_skip_count,
                )
                for obs_list in replicate_observations
                if obs_list
            ]
            total_steps = max(1, replicates * iterations)
            completed_steps = sum(
                max(0, min(iterations, len(obs_list) - initial_points))
                for obs_list in replicate_observations[:replicates]
            )
            color = (resume_run or {}).get("color") or BENCHMARK_COLORS[
                len(self.benchmark_runs) % len(BENCHMARK_COLORS)
            ]
            run_id = (resume_run or {}).get("id") or f"benchmark-{len(self.benchmark_runs) + 1}"
            partial_run = {
                "id": run_id,
                "name": name,
                "color": color,
                "time": _now(),
                "config": self._benchmark_config_snapshot(),
                "replicate_observations": replicate_observations,
                "replicate_traces": replicate_traces,
                "summary": self._summarize_replicate_traces(replicate_traces),
                "partial": True,
                "status": "running",
            }

            def update_partial_run(
                current_replicate: Optional[int] = None,
                current_observations: Optional[List[Dict[str, Any]]] = None,
            ) -> None:
                if current_replicate is not None and current_observations is not None:
                    while len(replicate_observations) <= current_replicate:
                        replicate_observations.append([])
                    replicate_observations[current_replicate] = [
                        dict(obs) for obs in current_observations
                    ]
                obs_sets = [[dict(obs) for obs in obs_list] for obs_list in replicate_observations]
                obs_sets = [obs_list for obs_list in obs_sets if obs_list]
                traces = [
                    _best_trace(
                        obs_list,
                        self.config["objective_direction"],
                        baseline_value=baseline_value,
                        skip_observations=trace_skip_count,
                    )
                    for obs_list in obs_sets
                ]
                partial_run["time"] = _now()
                partial_run["replicate_observations"] = obs_sets
                partial_run["replicate_traces"] = traces
                partial_run["summary"] = self._summarize_replicate_traces(traces)

            self.set_progress(
                f"Running benchmark: {name}",
                completed_steps,
                total_steps,
                detail=(
                    f"{replicates} replicate(s), {iterations} BO iteration(s), "
                    f"{self.config['optimizer'].upper()}"
                    + (
                        "; final BO step uses greedy acquisition"
                        if self.config["greedy_final_iteration"]
                        else ""
                    )
                    + (
                        f"; scoring up to {self._llm_scored_candidate_count(len(labelled))} "
                        f"candidate(s) x "
                        f"{self.config['llm_samples']} samples per BO step"
                        if self.config["optimizer"] == "llm"
                        else ""
                    )
                ),
                extra={"partial_run": partial_run},
            )
            try:
                for replicate in range(replicates):
                    self.check_cancelled()
                    rng = random.Random(seed + replicate)
                    shuffled = list(labelled)
                    rng.shuffle(shuffled)
                    observations = (
                        [dict(obs) for obs in replicate_observations[replicate]]
                        if replicate < len(replicate_observations)
                        else []
                    )
                    selected_ids = {
                        obs["candidate_id"]
                        for obs in observations
                        if obs.get("candidate_id")
                    }
                    for candidate in shuffled:
                        if len(observations) >= initial_points:
                            break
                        if candidate["id"] in selected_ids:
                            continue
                        observations.append(self._observation_from_candidate(candidate))
                        selected_ids.add(candidate["id"])
                    update_partial_run(replicate, observations)
                    start_iteration = max(0, len(observations) - initial_points)
                    for iteration in range(start_iteration, iterations):
                        self.check_cancelled()
                        available = [
                            candidate
                            for candidate in labelled
                            if candidate["id"] not in selected_ids
                        ]
                        if not available:
                            break
                        acquisition_override = (
                            "greedy"
                            if self.config["greedy_final_iteration"]
                            and iteration == iterations - 1
                            else None
                        )
                        candidate = self._benchmark_next_candidate(
                            available, observations, rng, acquisition=acquisition_override
                        )
                        self.check_cancelled()
                        observations.append(self._observation_from_candidate(candidate))
                        selected_ids.add(candidate["id"])
                        completed_steps += 1
                        update_partial_run(replicate, observations)
                        self.set_progress(
                            f"Running benchmark: {name}",
                            completed_steps,
                            total_steps,
                            detail=(
                                f"Replicate {replicate + 1}/{replicates}, "
                                f"iteration {iteration + 1}/{iterations}, "
                                f"{(acquisition_override or self.config['acquisition']).replace('_', ' ')}"
                            ),
                            extra={"partial_run": partial_run},
                        )
                    while len(replicate_observations) <= replicate:
                        replicate_observations.append([])
                    replicate_observations[replicate] = observations
                    replicate_traces = [
                        _best_trace(
                            obs_list,
                            self.config["objective_direction"],
                            baseline_value=baseline_value,
                            skip_observations=trace_skip_count,
                        )
                        for obs_list in replicate_observations
                        if obs_list
                    ]
                    update_partial_run()
            except RunCancelled as exc:
                self.last_error = None
                partial_run["partial"] = True
                partial_run["status"] = "stopped"
                partial_run["time"] = _now()
                update_partial_run()
                self._upsert_benchmark_run_locked(partial_run, resume_index)
                self.last_model_status = (
                    "Benchmark stopped by user. The partial run was saved and can be resumed."
                )
                self.log(f"Stopped benchmark '{name}'.")
                self.cancel_progress(
                    f"Running benchmark: {name}",
                    f"{str(exc)} Partial run saved.",
                )
                self.progress.pop("partial_run", None)
                self._autosave_locked()
                return self.to_json()
            except Exception as exc:  # pragma: no cover - needs live API/provider
                self.last_error = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                partial_run["partial"] = True
                partial_run["status"] = "error"
                partial_run["error"] = self.last_error
                partial_run["time"] = _now()
                update_partial_run()
                self._upsert_benchmark_run_locked(partial_run, resume_index)
                self.last_model_status = (
                    "Benchmark stopped after an API/model error. The partial trace "
                    "was saved; resume it after adjusting rate-limit or model settings."
                )
                self.log(f"Benchmark '{name}' stopped after an API/model error.")
                self.fail_progress(f"Running benchmark: {name}", self.last_error)
                self.progress.pop("partial_run", None)
                self._autosave_locked()
                return self.to_json()

            run = {
                "id": run_id,
                "name": name,
                "color": color,
                "time": _now(),
                "config": self._benchmark_config_snapshot(),
                "replicate_observations": replicate_observations,
                "replicate_traces": replicate_traces,
                "summary": self._summarize_replicate_traces(replicate_traces),
                "partial": False,
                "status": "complete",
            }
            self._upsert_benchmark_run_locked(run, resume_index)
            self.last_model_status = f"Added benchmark run: {name}."
            self.log(
                f"Ran benchmark '{name}' with {replicates} replicate(s), "
                f"{initial_points} initial point(s), and {iterations} BO iteration(s)."
            )
            self.finish_progress(
                f"Running benchmark: {name}",
                f"Appended {len(replicate_traces)} replicate trace(s).",
            )
            self._autosave_locked()
            return self.to_json()

    def clear_benchmarks(self) -> Dict[str, Any]:
        with self.lock:
            self.benchmark_runs = []
            self.last_model_status = "Cleared offline benchmark runs."
            self.log("Cleared offline benchmark runs.")
            self._autosave_locked()
            return self.to_json()

    def _training_rows_and_scaler(
        self, observations: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        return _group_training_observations(
            observations if observations is not None else self.active_observations(),
            self.config["objective_direction"],
            self.config["objective_scaling"],
        )

    def _training_observations(self) -> List[Dict[str, Any]]:
        rows, _ = self._training_rows_and_scaler()
        return rows

    def reset_run(self) -> Dict[str, Any]:
        with self.lock:
            self.observations = []
            self.suggestions = []
            self.last_error = None
            self.last_model_status = "Run reset. Dataset is still loaded."
            self.log("Cleared observations and suggestions.")
            self._autosave_locked()
            return self.to_json()

    def export_observations_csv(self) -> str:
        with self.lock:
            out = StringIO()
            objective_fields = []
            for name in self.objective_names:
                objective_fields.extend([name, f"{name}_uncertainty"])
            writer = csv.DictWriter(
                out,
                fieldnames=[
                    "source",
                    "run_id",
                    "run_name",
                    "run_status",
                    "replicate",
                    "experiment_count",
                    "procedure",
                    *objective_fields,
                    "time",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            for index, obs in enumerate(self.observations, start=1):
                row = {
                    "source": "live",
                    "run_id": "",
                    "run_name": "",
                    "run_status": "",
                    "replicate": "",
                    "experiment_count": index,
                    "procedure": obs["procedure"],
                    "time": obs["time"],
                }
                for name in self.objective_names:
                    row[name] = obs.get("objectives", {}).get(name, "")
                    row[f"{name}_uncertainty"] = obs.get("uncertainties", {}).get(
                        name, ""
                    )
                writer.writerow(row)
            for run in self.benchmark_runs:
                for replicate_index, observations in enumerate(
                    run.get("replicate_observations") or [], start=1
                ):
                    for obs_index, obs in enumerate(observations, start=1):
                        row = {
                            "source": "offline_benchmark",
                            "run_id": run.get("id", ""),
                            "run_name": run.get("name", ""),
                            "run_status": run.get("status", ""),
                            "replicate": replicate_index,
                            "experiment_count": obs_index,
                            "procedure": obs["procedure"],
                            "time": obs.get("time", ""),
                        }
                        for name in self.objective_names:
                            row[name] = obs.get("objectives", {}).get(name, "")
                            row[f"{name}_uncertainty"] = obs.get(
                                "uncertainties", {}
                            ).get(name, "")
                        writer.writerow(row)
            return out.getvalue()


class LocalAppHandler(BaseHTTPRequestHandler):
    state: LocalBOState

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.command == "GET" and self.path.startswith("/api/progress"):
            return
        sys.stdout.write("[%s] %s\n" % (_now(), fmt % args))

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, payload: str, content_type: str = "text/plain") -> None:
        raw = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _read_raw(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def _handle_error(self, exc: Exception, status: int = 400) -> None:
        self.state.last_error = str(exc)
        if self.state.progress.get("status") == "running":
            self.state.fail_progress(self.state.progress.get("label") or "Request", str(exc))
        self._send_json({"error": str(exc), "state": self.state.to_json()}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(INDEX_HTML, "text/html")
        elif parsed.path == "/pool-builder":
            self._send_text(POOL_BUILDER_HTML, "text/html")
        elif parsed.path == "/guide":
            self._send_text(USER_GUIDE_HTML, "text/html")
        elif parsed.path == "/api/state":
            self._send_json(self.state.to_json())
        elif parsed.path == "/api/progress":
            self._send_json({"progress": self.state.progress_snapshot()})
        elif parsed.path == "/api/campaigns":
            self._send_json({"campaigns": self.state.list_campaigns()})
        elif parsed.path == "/api/export-observations.csv":
            payload = self.state.export_observations_csv()
            raw = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition", "attachment; filename=boicl_observations.csv"
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/save-key":
                payload = self._read_json()
                key = _clean_api_key_value(
                    payload.get("openai_api_key") or "", "OPENAI_API_KEY"
                )
                openrouter_key = _clean_api_key_value(
                    payload.get("openrouter_api_key") or "", "OPENROUTER_API_KEY"
                )
                anthropic_key = _clean_api_key_value(
                    payload.get("anthropic_api_key") or "", "ANTHROPIC_API_KEY"
                )
                if key:
                    _write_env_value(self.state.env_path, "OPENAI_API_KEY", key)
                    self.state.log("Saved OPENAI_API_KEY to local .env.")
                if openrouter_key:
                    _write_env_value(
                        self.state.env_path, "OPENROUTER_API_KEY", openrouter_key
                    )
                    self.state.log("Saved OPENROUTER_API_KEY to local .env.")
                if anthropic_key:
                    _write_env_value(
                        self.state.env_path, "ANTHROPIC_API_KEY", anthropic_key
                    )
                    self.state.log("Saved ANTHROPIC_API_KEY to local .env.")
                self._send_json(self.state.to_json())
            elif parsed.path == "/api/import-dataset":
                query = parse_qs(parsed.query)
                filename = query.get("filename", ["dataset.csv"])[0]
                objective_name = query.get("objective_name", [""])[0]
                self._send_json(
                    self.state.import_dataset(
                        filename, self._read_raw(), objective_name=objective_name
                    )
                )
            elif parsed.path == "/api/save-campaign":
                self._send_json(self.state.save_campaign(self._read_json()))
            elif parsed.path == "/api/load-campaign":
                self._send_json(self.state.load_campaign(self._read_json()))
            elif parsed.path == "/api/delete-campaign":
                self._send_json(self.state.delete_campaign(self._read_json()))
            elif parsed.path == "/api/config":
                self._send_json(self.state.update_config(self._read_json()))
            elif parsed.path == "/api/regenerate-prompts":
                self._send_json(self.state.regenerate_prompts())
            elif parsed.path == "/api/cancel":
                self._send_json({"progress": self.state.request_cancel()})
            elif parsed.path == "/api/precompute-embeddings":
                self._send_json(self.state.precompute_embeddings())
            elif parsed.path == "/api/observe":
                self._send_json(self.state.add_observation(self._read_json()))
            elif parsed.path == "/api/suggest":
                self._send_json(self.state.suggest())
            elif parsed.path == "/api/inverse-design":
                self._send_json(self.state.generate_inverse_designs(self._read_json()))
            elif parsed.path == "/api/run-benchmark":
                self._send_json(self.state.run_benchmark(self._read_json()))
            elif parsed.path == "/api/clear-benchmarks":
                self._send_json(self.state.clear_benchmarks())
            elif parsed.path == "/api/reset":
                self._send_json(self.state.reset_run())
            else:
                self.send_error(404)
        except Exception as exc:
            self._handle_error(exc)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BO-ICL Local Runner</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --warn: #b45309;
      --bad: #b42318;
      --good: #027a48;
      --shadow: 0 8px 24px rgba(18, 31, 53, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      font-size: 20px;
      line-height: 1.2;
      margin: 0;
      font-weight: 720;
      letter-spacing: 0;
    }
    h2 {
      font-size: 15px;
      margin: 0 0 14px;
      letter-spacing: 0;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 7px;
      padding: 9px 12px;
      cursor: pointer;
      min-height: 38px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.secondary {
      background: #eff6ff;
      border-color: #bfdbfe;
      color: #1d4ed8;
    }
    button.danger {
      background: #fef3f2;
      border-color: #fecdca;
      color: var(--bad);
    }
    .button-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      text-decoration: none;
      font-size: 13px;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 9px 10px;
      color: var(--text);
    }
    textarea {
      min-height: 96px;
      resize: vertical;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 680;
      color: #344054;
      margin: 0 0 6px;
    }
    label.has-help {
      cursor: help;
    }
    label.has-help::after {
      content: " ?";
      color: var(--accent-2);
      font-weight: 760;
    }
    .shell {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      padding: 18px 24px 28px;
    }
    .stack {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 28px;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #f9fafb;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .chip.good { color: var(--good); border-color: #abefc6; background: #ecfdf3; }
    .chip.warn { color: var(--warn); border-color: #fedf89; background: #fffaeb; }
    .chip.bad { color: var(--bad); border-color: #fecdca; background: #fef3f2; }
    .muted {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .field {
      margin-bottom: 12px;
    }
    .switchline {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: #344054;
    }
    .switchline input { width: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid #edf0f5;
      padding: 9px 8px;
    }
    th {
      color: #475467;
      font-size: 12px;
      font-weight: 720;
      background: #fbfcfe;
    }
    td.procedure {
      max-width: 560px;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }
    .plot {
      width: 100%;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .plot text {
      fill: #667085;
      font-size: 12px;
    }
    .notice {
      border-left: 3px solid var(--warn);
      background: #fffaeb;
      color: #7a2e0e;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.4;
    }
    .error {
      border-left-color: var(--bad);
      background: #fef3f2;
      color: var(--bad);
    }
    .scroll {
      max-height: 360px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      padding: 18px;
      text-align: center;
      background: #fbfcfe;
      font-size: 13px;
    }
    .progressbox {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      margin-bottom: 12px;
      background: #fbfcfe;
    }
    .progressbar {
      width: 100%;
      height: 9px;
      border-radius: 999px;
      background: #e4e7ec;
      overflow: hidden;
      margin: 8px 0 4px;
    }
    .progressbar > div {
      height: 100%;
      width: 0%;
      background: var(--accent);
      transition: width 180ms ease;
    }
    .hidden {
      display: none !important;
    }
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .shell { grid-template-columns: 1fr; padding: 14px; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>BO-ICL Local Runner</h1>
      <div class="muted" id="modelStatus">Starting...</div>
    </div>
    <div class="toolbar">
      <span class="chip" id="keyStatus">Key status</span>
      <span class="chip" id="campaignStatus">Unsaved campaign</span>
      <span class="chip" id="workflowStatus">Workflow</span>
      <span class="chip" id="datasetStatus">No dataset</span>
      <span class="chip" id="embeddingStatus">Embeddings</span>
      <span class="chip" id="observationStatus">0 observations</span>
      <button id="openPoolBuilderTop" title="Open or focus the reusable local procedure-pool builder window.">Pool Builder</button>
      <a class="button-link" href="/guide" target="_blank" rel="noopener" title="Open the local user guide in a new browser tab.">User Guide</a>
      <button class="secondary" id="suggestTop">Update Suggestions</button>
    </div>
  </header>

  <main class="shell">
    <aside class="stack">
      <section class="panel">
        <h2>Secrets</h2>
        <div class="field">
          <label for="openaiKey">OpenAI API key</label>
          <input id="openaiKey" type="password" autocomplete="off" placeholder="sk-...">
        </div>
        <div class="field">
          <label for="openrouterKey">OpenRouter API key</label>
          <input id="openrouterKey" type="password" autocomplete="off" placeholder="optional">
        </div>
        <div class="field">
          <label for="anthropicKey">Anthropic API key</label>
          <input id="anthropicKey" type="password" autocomplete="off" placeholder="optional">
        </div>
        <button class="primary" id="saveKey">Save Locally</button>
      </section>

      <section class="panel">
        <h2>Campaign</h2>
        <div class="field">
          <label for="campaignName">Campaign name</label>
          <input id="campaignName" placeholder="new campaign">
        </div>
        <div class="field">
          <label for="savedCampaign">Saved campaigns</label>
          <select id="savedCampaign"></select>
        </div>
        <div class="toolbar">
          <button class="primary" id="saveCampaign">Save</button>
          <button id="saveAsCampaign">Save As New</button>
          <button id="loadCampaign">Load</button>
          <button id="deleteCampaign">Delete</button>
        </div>
      </section>

      <section class="panel">
        <h2>Dataset</h2>
        <div class="field">
          <label for="datasetFile">CSV, Excel, or NPY file</label>
          <input id="datasetFile" type="file" accept=".csv,.txt,.xlsx,.xls,.npy">
        </div>
        <div class="toolbar">
          <button id="importDataset">Import Dataset</button>
          <button id="prepareEmbeddings">Prepare Embeddings</button>
          <button id="openPoolBuilderDataset">Build Pool</button>
        </div>
        <div class="muted" id="embeddingDetail" style="margin-top: 8px;"></div>
      </section>

      <section class="panel">
        <h2>Settings</h2>
        <div class="field">
          <label for="workflowMode">Workflow mode</label>
          <select id="workflowMode">
            <option value="offline">Automatic benchmark: full labeled dataset</option>
            <option value="live">Live campaign: add results manually</option>
          </select>
        </div>
        <div class="field">
          <label for="optimizer">Suggestion engine</label>
          <select id="optimizer">
            <option value="gpr">GPR with embeddings</option>
            <option value="llm">BO-ICL LLM</option>
          </select>
        </div>
        <div class="field">
          <label for="objectiveName">Objective name</label>
          <input id="objectiveName" value="objective" list="objectiveOptions">
          <datalist id="objectiveOptions"></datalist>
          <datalist id="modelOptions"></datalist>
          <datalist id="embeddingModelOptions"></datalist>
        </div>
        <div class="row">
          <div class="field">
            <label for="objectiveDirection">Goal</label>
            <select id="objectiveDirection">
              <option value="maximize">Maximize</option>
              <option value="minimize">Minimize</option>
            </select>
          </div>
          <div class="field">
            <label for="acquisition">Acquisition</label>
            <select id="acquisition"></select>
          </div>
        </div>
        <div class="muted" style="margin-top: -4px; margin-bottom: 12px;">
          Tungsten phase campaigns: use an XRD phase percentage from 0-100 as the live objective, for example alpha phase (%) for Im-3m or beta phase (%) for Pm-3n.
        </div>
        <div class="field">
          <label for="objectiveScaling">Target scaling</label>
          <select id="objectiveScaling">
            <option value="off">Off</option>
            <option value="auto">Auto range</option>
            <option value="minmax">Min-max</option>
            <option value="zscore">Z-score</option>
          </select>
        </div>
        <div class="field">
          <label for="plotStatGuides">Plot guides</label>
          <select id="plotStatGuides">
            <option value="max">Best only</option>
            <option value="paper">Paper stats</option>
            <option value="off">Off</option>
          </select>
        </div>
        <div class="row">
          <div class="field">
            <label for="embeddingModel">Embedding model</label>
            <input id="embeddingModel" value="text-embedding-ada-002" list="embeddingModelOptions">
          </div>
          <div class="field">
            <label for="predictionModel">Prediction LLM</label>
            <input id="predictionModel" value="gpt-4o" list="modelOptions">
          </div>
        </div>
        <div class="field">
          <label for="inverseModel">Inverse design LLM</label>
          <input id="inverseModel" value="gpt-4o" list="modelOptions">
        </div>
        <div class="field">
          <label for="predictionSystemMessage">Prediction system message</label>
          <textarea id="predictionSystemMessage"></textarea>
        </div>
        <div class="field">
          <label for="inverseSystemMessage">Inverse design system message</label>
          <textarea id="inverseSystemMessage"></textarea>
        </div>
        <div class="toolbar">
          <button id="regeneratePrompts">Use Dataset Prompts</button>
        </div>
        <div class="row">
          <div class="field">
            <label for="batchSize">Batch size</label>
            <input id="batchSize" type="number" min="1" max="25" value="1">
          </div>
          <div class="field">
            <label for="ucbLambda">UCB lambda</label>
            <input id="ucbLambda" type="number" step="0.1" value="0.1">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="llmSamples">LLM samples</label>
            <input id="llmSamples" type="number" min="1" max="20" value="3">
          </div>
          <div class="field">
            <label for="selectorK">Selector examples</label>
            <input id="selectorK" type="number" min="0" value="0">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="inverseFilter">LLM shortlist</label>
            <input id="inverseFilter" type="number" min="0" value="16">
          </div>
          <div class="field">
            <label for="inverseRandomCandidates">Random add-ons</label>
            <input id="inverseRandomCandidates" type="number" min="0" value="0">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="inverseTargetValue">Inverse target</label>
            <input id="inverseTargetValue" type="number" step="any" placeholder="auto">
          </div>
          <div class="field">
            <label for="inverseTargetMultiplier">Auto target multiplier</label>
            <input id="inverseTargetMultiplier" type="number" step="0.05" value="1.2">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="inverseTargetJitter">Auto target jitter</label>
            <input id="inverseTargetJitter" type="number" min="0" max="1" step="0.01" value="0.05">
          </div>
          <div class="field">
            <label for="inverseTargetFloorValue">Auto target floor</label>
            <input id="inverseTargetFloorValue" type="number" step="any" placeholder="optional">
          </div>
        </div>
        <div class="field">
            <label for="inverseDesignCount">Inverse proposals</label>
            <input id="inverseDesignCount" type="number" min="1" max="10" value="3">
        </div>
        <div class="row">
          <div class="field">
            <label for="iterationsPerTrial">Iteration cap</label>
            <input id="iterationsPerTrial" type="number" min="0" value="0">
          </div>
          <div class="field">
            <label for="replicatesPerCandidate">Replicates</label>
            <input id="replicatesPerCandidate" type="number" min="1" value="1">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="scoreLimit">Broad pool</label>
            <input id="scoreLimit" type="number" min="1" value="250">
          </div>
          <div class="field">
            <label for="nNeighbors">Neighbors</label>
            <input id="nNeighbors" type="number" min="1" value="5">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="apiPauseSeconds">API pause (s)</label>
            <input id="apiPauseSeconds" type="number" min="0" step="0.1" value="0.5">
          </div>
          <div class="field">
            <label for="apiRetryAttempts">429 retries</label>
            <input id="apiRetryAttempts" type="number" min="1" max="20" value="8">
          </div>
        </div>
        <div class="field">
          <label for="apiRateLimitCooldownSeconds">429 cooldown (s)</label>
          <input id="apiRateLimitCooldownSeconds" type="number" min="0" max="300" step="1" value="10">
        </div>
        <div class="field">
          <label class="switchline">
          <input id="autoSuggest" type="checkbox" checked>
          Auto update
        </label>
        </div>
        <div class="toolbar" style="margin-top: 12px;">
          <button id="saveConfig">Apply</button>
          <button id="resetRun">Reset Run</button>
        </div>
      </section>

      <section class="panel" id="offlineBenchmarkPanel">
        <h2>Offline Benchmark</h2>
        <div class="field">
          <label for="benchmarkName">Run label</label>
          <input id="benchmarkName" placeholder="auto">
        </div>
        <div class="row">
          <div class="field">
            <label for="benchmarkInitialPoints">Initial random</label>
            <input id="benchmarkInitialPoints" type="number" min="1" value="1">
          </div>
          <div class="field">
            <label for="benchmarkIterations">BO iterations</label>
            <input id="benchmarkIterations" type="number" min="1" value="30">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="benchmarkReplicates">Workflow replicates</label>
            <input id="benchmarkReplicates" type="number" min="1" max="50" value="5">
          </div>
          <div class="field">
            <label for="benchmarkSeed">Seed</label>
            <input id="benchmarkSeed" type="number" value="0">
          </div>
        </div>
        <div class="field">
          <label for="benchmarkStartingBaseline">Starting baseline</label>
          <select id="benchmarkStartingBaseline">
            <option value="none">Observed initial points only</option>
            <option value="mean">Dataset mean incumbent</option>
          </select>
        </div>
        <div class="field">
          <label class="switchline" for="greedyFinalIteration">
            <input id="greedyFinalIteration" type="checkbox">
            Greedy for final iteration
          </label>
          <div class="hint">Use the selected acquisition for the run, then switch to greedy for the last BO choice in each replicate.</div>
        </div>
        <div class="toolbar">
          <button class="primary" id="runBenchmark">Run & Append</button>
          <button id="resumeBenchmark">Resume Last</button>
          <button id="clearBenchmarks">Clear Runs</button>
        </div>
      </section>
    </aside>

    <section class="stack">
      <section class="panel">
        <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
          <h2 style="margin:0;">Best So Far</h2>
          <a class="muted" href="/api/export-observations.csv">Export observations</a>
        </div>
        <div id="workflowBanner" class="notice" style="margin-bottom: 12px;"></div>
        <div id="progressPanel" class="progressbox hidden">
          <div class="toolbar" style="justify-content: space-between;">
            <strong id="progressLabel">Working</strong>
            <div class="toolbar">
              <span class="muted" id="progressCount">0%</span>
              <button class="danger" id="stopRun">Stop</button>
            </div>
          </div>
          <div class="progressbar"><div id="progressBar"></div></div>
          <div class="muted" id="progressDetail"></div>
        </div>
        <div id="plot" class="plot"></div>
        <div id="benchmarkRuns" style="margin-top: 12px;"></div>
      </section>

      <section class="panel" id="liveResultPanel">
        <h2>Add Result</h2>
        <div class="field">
          <label for="candidateSelect">Candidate</label>
          <select id="candidateSelect"></select>
        </div>
        <div class="muted" id="candidatePreview" style="margin: -6px 0 12px;"></div>
        <div class="field">
          <label for="manualProcedure">Procedure</label>
          <textarea id="manualProcedure"></textarea>
        </div>
        <div class="row">
          <div class="field">
            <label for="objectiveValue">Objective value</label>
            <input id="objectiveValue" type="number" step="any">
          </div>
          <div class="field">
            <label for="objectiveUncertainty">Uncertainty</label>
            <input id="objectiveUncertainty" type="number" step="any" placeholder="optional">
          </div>
        </div>
        <div class="muted" style="margin-bottom: 12px;">For the tungsten workflow, enter the XRD-calculated phase percentage on a 0-100 scale. Example: enter 73.5 for 73.5%, not 0.735.</div>
        <button class="primary" id="addObservation">Add Observation</button>
      </section>

      <section class="panel" id="suggestionsPanel">
        <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
          <h2 style="margin:0;">Suggestions</h2>
          <button id="suggest">Update Suggestions</button>
        </div>
        <div id="suggestions"></div>
      </section>

      <section class="panel" id="inverseDesignPanel">
        <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
          <h2 style="margin:0;">Inverse Design</h2>
          <button id="inverseDesign">Generate Proposals</button>
        </div>
        <div id="inverseDesigns"></div>
      </section>

      <section class="panel">
        <h2>Observations</h2>
        <div id="observations"></div>
      </section>

      <section id="messages" class="stack"></section>
    </section>
  </main>

  <script>
    window.name = 'boicl_runner';
    let state = null;
    let busy = false;
    let progressTimer = null;
    let transientNotice = '';
    const poolChannel = 'BroadcastChannel' in window ? new BroadcastChannel('boicl-local-runner') : null;

    const $ = (id) => document.getElementById(id);

    const HELP_TEXT = {
      openaiKey: 'Stored only in the local .env file. Required for OpenAI LLMs and embedding-based GPR.',
      openrouterKey: 'Stored only in the local .env file. Required only for model names that start with openrouter/.',
      anthropicKey: 'Stored only in the local .env file. Required only for Claude model names.',
      campaignName: 'Local campaign name. Save once, then future changes autosave into saved_experiments.',
      savedCampaign: 'Previously saved local campaigns that can be loaded without re-uploading the dataset.',
      workflowMode: 'Choose automatic benchmark for fully labeled datasets; choose live campaign when labels arrive from experiments over time.',
      datasetFile: 'Accepted formats: CSV, TXT, XLS, XLSX, and NPY. The first column must be procedure text; later numeric columns are objectives.',
      optimizer: 'Choose GPR with embeddings for the GP baseline, or BO-ICL LLM for in-context LLM predictions over the uploaded pool.',
      objectiveName: 'The numeric label column to optimize. If multiple objective columns were uploaded, choose one here.',
      objectiveDirection: 'Maximize for yields/selectivity/scores; minimize for losses, errors, or costs.',
      acquisition: 'Candidate ranking rule. The paper notebook default sweep included upper confidence bound, greedy, random, and random mean baseline.',
      objectiveScaling: 'Off keeps labels in their original units for model fitting. Auto/min-max/z-score scale only the model target; plots stay in original units.',
      plotStatGuides: 'Controls full-dataset dashed reference lines. Best only is cleaner; Paper stats adds mean and percentile guides.',
      embeddingModel: 'OpenAI embedding model used to featurize procedures for GPR and nearest-neighbor inverse filtering.',
      predictionModel: 'LLM used by BO-ICL to predict objective values and acquisition scores for candidate procedures.',
      inverseModel: 'LLM used only when generating inverse-design text or inverse-filter candidates.',
      predictionSystemMessage: 'Dataset-aware instruction prepended to BO-ICL prediction prompts. It is generated from procedure style examples without hidden labels.',
      inverseSystemMessage: 'Dataset-aware instruction prepended to inverse-design prompts. It constrains proposals to the uploaded procedure style.',
      batchSize: 'Number of candidates suggested at once in live experimentation. The paper BO loop used 1.',
      ucbLambda: 'Exploration weight for upper confidence bound. The paper notebook default was 0.1.',
      llmSamples: 'Number of LLM prediction samples per shortlisted candidate for BO-ICL uncertainty estimates.',
      selectorK: 'Number of nearest labeled examples to include in prompts. 0 uses the normal few-shot history.',
      inverseFilter: 'LLM-mode shortlist size retrieved with inverse-design text plus embeddings before completions are requested. 0 disables this and scores the broad pool.',
      inverseRandomCandidates: 'Extra random candidates mixed with the LLM shortlist before completions are requested.',
      inverseTargetValue: 'Manual target for inverse design. Leave blank to use the current best value times the multiplier.',
      inverseTargetMultiplier: 'Mean multiplier used for automatic inverse-design targets when no explicit target is entered. The paper-style default is 1.2.',
      inverseTargetJitter: 'Standard deviation for the random multiplier used by automatic inverse-design targets. 0.05 means target = current best x Normal(1.2, 0.05); set 0 for deterministic.',
      inverseTargetFloorValue: 'Optional minimum automatic inverse-design target for sparse-zero maximization campaigns. For alpha phase percent, enter a whole percent such as 5 or 10. Manual inverse target overrides this.',
      inverseDesignCount: 'Number of free-form inverse-design proposals to generate.',
      iterationsPerTrial: 'Live-mode stopping point after this many active-objective observations. 0 means no cap.',
      replicatesPerCandidate: 'How many live measurements are allowed for the same candidate before it leaves the available pool.',
      scoreLimit: 'Broad candidate pool sampled before scoring. GPR scores this many directly; LLM mode narrows it to the shortlist first when LLM shortlist is enabled.',
      apiPauseSeconds: 'Small delay after provider API calls. Increase this when rate limits appear.',
      apiRetryAttempts: 'Number of automatic retries for 429/rate-limit or transient provider errors.',
      apiRateLimitCooldownSeconds: 'Extra cooldown after a 429/rate-limit error before retrying. Increase this for OpenAI TPM limits; it is separate from the normal API pause after successful calls.',
      nNeighbors: 'GPR embedding neighbor count used by the local GP featurization pipeline.',
      autoSuggest: 'When checked, adding a live result immediately refreshes suggestions.',
      benchmarkName: 'Optional label for this appended offline benchmark curve.',
      benchmarkInitialPoints: 'Number of random starting examples. The paper default was 1; 2 is also common for small pools.',
      benchmarkIterations: 'Number of BO choices after initialization. The paper notebook default was 30.',
      benchmarkReplicates: 'Number of repeated runs for the same workflow. The paper notebook default was 5.',
      benchmarkSeed: 'Starting random seed for reproducible benchmark replicates.',
      benchmarkStartingBaseline: 'Plot-only incumbent value at experiment count 0. Dataset mean starts each benchmark curve from the mean label without adding a fake labeled procedure to the LLM context.',
      greedyFinalIteration: 'When checked, only the final BO choice in each replicate switches to greedy acquisition. Earlier choices use the selected acquisition function.',
      candidateSelect: 'Choose a suggested or uploaded pool candidate for live result entry.',
      manualProcedure: 'Procedure text for a live observation. This fills automatically when you select a candidate.',
      objectiveValue: 'Measured objective value in original units. For tungsten phase optimization, enter whole percent units from 0 to 100, for example 73.5 for 73.5%, not 0.735.',
      objectiveUncertainty: 'Optional measurement uncertainty or standard deviation in original units.'
    };

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toLocaleString(undefined, { maximumSignificantDigits: digits });
    }

    function applyHelpText() {
      Object.entries(HELP_TEXT).forEach(([id, text]) => {
        const element = $(id);
        if (element) {
          element.title = text;
          element.setAttribute('aria-label', text);
        }
        const label = document.querySelector(`label[for="${id}"]`);
        if (label) {
          label.title = text;
          label.classList.add('has-help');
        }
      });
      const buttonHelp = {
        saveKey: 'Save pasted keys into the local ignored .env file.',
        saveCampaign: 'Save or overwrite the current local campaign.',
        saveAsCampaign: 'Create a separate saved campaign copy with the current state.',
        loadCampaign: 'Reload the selected saved campaign and continue where it left off.',
        deleteCampaign: 'Delete the selected local campaign save folder.',
        openPoolBuilderTop: 'Open or focus one reusable Pool Builder window.',
        openPoolBuilderDataset: 'Open or focus one reusable Pool Builder window.',
        importDataset: 'Load candidates and hidden labels from the selected file.',
        prepareEmbeddings: 'Generate and save local embeddings for the current dataset and embedding model.',
        regeneratePrompts: 'Replace both system messages with prompts generated from the uploaded dataset structure and procedure examples.',
        saveConfig: 'Apply the current settings without running a suggestion.',
        resetRun: 'Clear live observations and suggestions while keeping the uploaded dataset.',
        runBenchmark: 'Run the current configuration against hidden labels and append it to the plot.',
        resumeBenchmark: 'Continue the most recent stopped or interrupted offline benchmark from its saved procedure sequence.',
        clearBenchmarks: 'Remove appended offline benchmark curves from the plot.',
        stopRun: 'Ask the current long-running task to stop after the current API call returns.',
        addObservation: 'Record a live measurement for the selected or typed procedure.',
        suggest: 'Update live-mode candidate suggestions.',
        suggestTop: 'Update live-mode candidate suggestions.',
        inverseDesign: 'Generate free-form inverse-design proposals from labeled examples.'
      };
      Object.entries(buttonHelp).forEach(([id, text]) => {
        const element = $(id);
        if (element) element.title = text;
      });
    }

    async function request(path, options = {}) {
      setBusy(true);
      startProgressPolling();
      try {
        const response = await fetch(path, options);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Request failed');
        state = payload.state || payload;
        state.live_benchmark_run = (state.progress || {}).partial_run || null;
        render();
        return state;
      } catch (error) {
        renderError(error.message);
      } finally {
        setBusy(false);
        stopProgressPolling();
        if (state) state.live_benchmark_run = (state.progress || {}).partial_run || null;
        if (state) renderProgress(state.progress || {});
      }
    }

    function setBusy(next) {
      busy = next;
      document.querySelectorAll('button').forEach((button) => {
        if (button.id === 'stopRun') return;
        button.disabled = busy;
      });
    }

    function openPoolBuilder() {
      const popup = window.open('/pool-builder', 'boicl_pool_builder', 'width=1180,height=820');
      if (popup) popup.focus();
    }

    async function refresh() {
      const response = await fetch('/api/state');
      state = await response.json();
      render();
    }

    async function pollProgress() {
      try {
        const response = await fetch('/api/progress', { cache: 'no-store' });
        const payload = await response.json();
        const progress = payload.progress || {};
        renderProgress(progress);
        if (state && progress.partial_run) {
          state.live_benchmark_run = progress.partial_run;
          renderPlot();
          renderBenchmarkRuns();
        }
      } catch (_) {
        // Progress polling is best-effort; the main request still owns errors.
      }
    }

    function startProgressPolling() {
      stopProgressPolling();
      pollProgress();
      progressTimer = window.setInterval(pollProgress, 2000);
    }

    function stopProgressPolling() {
      if (progressTimer) {
        window.clearInterval(progressTimer);
        progressTimer = null;
      }
    }

    function payloadConfig() {
      return {
        workflow_mode: $('workflowMode').value,
        optimizer: $('optimizer').value,
        objective_name: $('objectiveName').value,
        objective_direction: $('objectiveDirection').value,
        objective_scaling: $('objectiveScaling').value,
        plot_stat_guides: $('plotStatGuides').value,
        acquisition: $('acquisition').value,
        embedding_model: $('embeddingModel').value,
        prediction_model: $('predictionModel').value,
        inverse_model: $('inverseModel').value,
        prediction_system_message: $('predictionSystemMessage').value,
        inverse_system_message: $('inverseSystemMessage').value,
        llm_samples: Number($('llmSamples').value || 3),
        selector_k: Number($('selectorK').value || 0),
        inverse_filter: Number($('inverseFilter').value || 16),
        inverse_random_candidates: Number($('inverseRandomCandidates').value || 0),
        inverse_target_value: $('inverseTargetValue').value,
        inverse_target_multiplier: Number($('inverseTargetMultiplier').value || 1.2),
        inverse_target_jitter: Number($('inverseTargetJitter').value || 0),
        inverse_target_floor_value: $('inverseTargetFloorValue').value,
        inverse_design_count: Number($('inverseDesignCount').value || 3),
        batch_size: Number($('batchSize').value || 1),
        iterations_per_trial: Number($('iterationsPerTrial').value || 0),
        replicates_per_candidate: Number($('replicatesPerCandidate').value || 1),
        benchmark_iterations: Number($('benchmarkIterations').value || 30),
        benchmark_replicates: Number($('benchmarkReplicates').value || 5),
        benchmark_initial_points: Number($('benchmarkInitialPoints').value || 1),
        benchmark_seed: Number($('benchmarkSeed').value || 0),
        benchmark_starting_baseline: $('benchmarkStartingBaseline').value,
        greedy_final_iteration: $('greedyFinalIteration').checked,
        ucb_lambda: Number($('ucbLambda').value || 0.1),
        score_limit: Number($('scoreLimit').value || 250),
        api_pause_seconds: Number($('apiPauseSeconds').value || 0),
        api_retry_attempts: Number($('apiRetryAttempts').value || 8),
        api_rate_limit_cooldown_seconds: Number($('apiRateLimitCooldownSeconds').value || 0),
        n_neighbors: Number($('nNeighbors').value || 5),
        auto_suggest: $('autoSuggest').checked
      };
    }

    function latestPartialBenchmarkRun() {
      const runs = state ? (state.benchmark_runs || []) : [];
      for (let idx = runs.length - 1; idx >= 0; idx -= 1) {
        if (runs[idx].partial) return runs[idx];
      }
      return null;
    }

    function render() {
      if (!state) return;
      const key = state.key_status.openai_configured;
      const keyCount = [state.key_status.openai_configured, state.key_status.openrouter_configured, state.key_status.anthropic_configured].filter(Boolean).length;
      $('keyStatus').textContent = keyCount ? `${keyCount} API key${keyCount > 1 ? 's' : ''} set` : 'API keys missing';
      $('keyStatus').className = `chip ${keyCount ? 'good' : 'warn'}`;
      const campaign = state.campaign || {};
      $('campaignStatus').textContent = campaign.saved ? `Saved: ${campaign.name || campaign.id}` : 'Unsaved campaign';
      $('campaignStatus').className = `chip ${campaign.saved ? 'good' : 'warn'}`;
      const workflow = state.config.workflow_mode === 'offline' ? 'Automatic benchmark' : 'Live campaign';
      $('workflowStatus').textContent = workflow;
      $('workflowStatus').className = `chip ${state.config.workflow_mode === 'offline' ? 'good' : 'warn'}`;
      $('datasetStatus').textContent = `${state.candidate_count} candidates, ${state.label_count || 0} labels`;
      $('datasetStatus').className = `chip ${state.candidate_count ? 'good' : 'warn'}`;
      const cache = state.embedding_cache || {};
      $('embeddingStatus').textContent = `${cache.cached_count || 0}/${cache.total_count || 0} embedded`;
      $('embeddingStatus').className = `chip ${cache.ready ? 'good' : (cache.total_count ? 'warn' : '')}`;
      $('embeddingDetail').textContent = cache.total_count
        ? `${cache.cached_count || 0} of ${cache.total_count || 0} candidates embedded with ${cache.model || ''}.`
        : 'Import a dataset before preparing embeddings.';
      $('observationStatus').textContent = `${state.observations.length} observations`;
      $('observationStatus').className = `chip ${state.observations.length ? 'good' : 'warn'}`;
      $('modelStatus').textContent = state.last_model_status || '';

      renderCampaigns();
      renderConfig();
      renderWorkflowMode();
      renderProgress(state.progress || {});
      renderCandidateSelect();
      renderPlot();
      renderBenchmarkRuns();
      renderSuggestions();
      renderInverseDesigns();
      renderObservations();
      renderMessages();
    }

    function renderCampaigns() {
      const campaign = state.campaign || {};
      $('campaignName').value = campaign.name || '';
      const campaigns = state.campaigns || [];
      const options = ['<option value="">Choose saved campaign</option>'].concat(
        campaigns.map((item) => {
          const label = `${item.name} (${item.observation_count || 0} obs, ${item.candidate_count || 0} candidates)`;
          return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
        })
      );
      $('savedCampaign').innerHTML = options.join('');
      if (campaign.id) $('savedCampaign').value = campaign.id;
    }

    function renderConfig() {
      const config = state.config;
      $('workflowMode').value = config.workflow_mode;
      $('optimizer').value = config.optimizer;
      $('objectiveName').value = config.objective_name;
      $('objectiveOptions').innerHTML = (state.objective_names || [])
        .map((name) => `<option value="${escapeHtml(name)}"></option>`)
        .join('');
      $('modelOptions').innerHTML = (state.model_presets || [])
        .map((name) => `<option value="${escapeHtml(name)}"></option>`)
        .join('');
      $('embeddingModelOptions').innerHTML = (state.embedding_model_presets || [])
        .map((name) => `<option value="${escapeHtml(name)}"></option>`)
        .join('');
      $('objectiveDirection').value = config.objective_direction;
      $('objectiveScaling').value = config.objective_scaling;
      $('plotStatGuides').value = config.plot_stat_guides || 'max';
      $('embeddingModel').value = config.embedding_model;
      $('predictionModel').value = config.prediction_model;
      $('inverseModel').value = config.inverse_model;
      $('predictionSystemMessage').value = config.prediction_system_message;
      $('inverseSystemMessage').value = config.inverse_system_message;
      $('llmSamples').value = config.llm_samples;
      $('selectorK').value = config.selector_k;
      $('inverseFilter').value = config.inverse_filter;
      $('inverseRandomCandidates').value = config.inverse_random_candidates;
      $('inverseTargetValue').value = config.inverse_target_value;
      $('inverseTargetMultiplier').value = config.inverse_target_multiplier;
      $('inverseTargetJitter').value = config.inverse_target_jitter;
      $('inverseTargetFloorValue').value = config.inverse_target_floor_value || '';
      $('inverseDesignCount').value = config.inverse_design_count;
      $('batchSize').value = config.batch_size;
      $('iterationsPerTrial').value = config.iterations_per_trial;
      $('replicatesPerCandidate').value = config.replicates_per_candidate;
      $('benchmarkIterations').value = config.benchmark_iterations;
      $('benchmarkReplicates').value = config.benchmark_replicates;
      $('benchmarkInitialPoints').value = config.benchmark_initial_points;
      $('benchmarkSeed').value = config.benchmark_seed;
      $('benchmarkStartingBaseline').value = config.benchmark_starting_baseline || 'none';
      $('greedyFinalIteration').checked = Boolean(config.greedy_final_iteration);
      $('ucbLambda').value = config.ucb_lambda;
      $('scoreLimit').value = config.score_limit;
      $('apiPauseSeconds').value = config.api_pause_seconds;
      $('apiRetryAttempts').value = config.api_retry_attempts;
      $('apiRateLimitCooldownSeconds').value = config.api_rate_limit_cooldown_seconds;
      $('nNeighbors').value = config.n_neighbors;
      $('autoSuggest').checked = Boolean(config.auto_suggest);
      const partialRun = latestPartialBenchmarkRun();
      $('resumeBenchmark').disabled = busy || !partialRun;

      const current = $('acquisition').value || config.acquisition;
      $('acquisition').innerHTML = state.acquisition_functions
        .map((name) => `<option value="${name}">${name.replaceAll('_', ' ')}</option>`)
        .join('');
      $('acquisition').value = current;
    }

    function renderWorkflowMode() {
      const isOffline = state.config.workflow_mode === 'offline';
      $('offlineBenchmarkPanel').classList.toggle('hidden', !isOffline);
      $('liveResultPanel').classList.toggle('hidden', isOffline);
      $('suggestionsPanel').classList.toggle('hidden', isOffline);
      $('inverseDesignPanel').classList.toggle('hidden', isOffline);
      $('suggestTop').classList.toggle('hidden', isOffline);
      if (isOffline) {
        $('workflowBanner').textContent = 'Automatic benchmark mode: use Run & Append. Uploaded labels are hidden from the model until each simulated experiment is selected. Do not use Add Result or Generate Proposals for this workflow.';
      } else {
        $('workflowBanner').textContent = 'Live campaign mode: use Update Suggestions to choose the next procedure, run the experiment offline, then enter the result with Add Observation.';
      }
    }

    function renderProgress(progress) {
      const status = progress.status || 'idle';
      const active = ['running', 'cancelling', 'cancelled', 'complete', 'error'].includes(status);
      $('progressPanel').classList.toggle('hidden', !active);
      if (!active) return;
      const percent = Number(progress.percent || 0);
      $('progressLabel').textContent = progress.label || 'Working';
      $('progressCount').textContent = progress.total
        ? `${progress.current || 0}/${progress.total} (${percent}%)`
        : `${percent}%`;
      $('progressBar').style.width = `${Math.max(0, Math.min(100, percent))}%`;
      $('progressDetail').textContent = [progress.detail, progress.updated]
        .filter(Boolean)
        .join(' - ');
      $('progressPanel').classList.toggle('error', status === 'error');
      $('stopRun').disabled = status !== 'running';
      $('stopRun').classList.toggle('hidden', status !== 'running' && status !== 'cancelling');
    }

    function candidateProcedureSummary(procedure) {
      const text = String(procedure || '');
      const temp = text.match(/ramp to\s+([0-9.]+)\s*C/i);
      const ramp = text.match(/at\s+([0-9.]+)\s*C\/min/i);
      const dwell = text.match(/soak for\s+([0-9.]+)\s*h/i);
      const parts = [];
      if (temp) parts.push(`T=${temp[1]} C`);
      if (ramp) parts.push(`ramp=${ramp[1]} C/min`);
      if (dwell) parts.push(`dwell=${dwell[1]} h`);
      return parts.length ? parts.join(' | ') : '';
    }

    function candidateOptionLabel(item, prefix = '') {
      const summary = candidateProcedureSummary(item.procedure);
      const row = item.row ? `#${item.row}` : '';
      const lead = [prefix, row, summary].filter(Boolean).join(' ');
      const tail = String(item.procedure || '').replace(/\s+/g, ' ').slice(0, 90);
      return lead ? `${lead} - ${tail}` : tail;
    }

    function updateCandidatePreview() {
      const id = $('candidateSelect').value;
      const cand = (state.candidates || []).find((item) => item.id === id);
      const sug = (state.suggestions || []).find((item) => item.candidate_id === id);
      const item = cand || sug;
      if (!item) {
        $('candidatePreview').textContent = '';
        return;
      }
      const summary = candidateProcedureSummary(item.procedure);
      $('candidatePreview').textContent = summary
        ? `${summary}. Full procedure is shown below.`
        : 'Full selected procedure is shown below.';
    }

    function renderCandidateSelect() {
      const suggestions = state.suggestions || [];
      const candidates = (state.available_candidates || []).slice(0, 500);
      const options = ['<option value="">Manual procedure</option>'];
      suggestions.forEach((sug) => {
        options.push(`<option value="${escapeHtml(sug.candidate_id)}">${escapeHtml(candidateOptionLabel(sug, 'Suggested'))}</option>`);
      });
      candidates.forEach((cand) => {
        if (!suggestions.some((sug) => sug.candidate_id === cand.id)) {
          options.push(`<option value="${escapeHtml(cand.id)}">${escapeHtml(candidateOptionLabel(cand))}</option>`);
        }
      });
      $('candidateSelect').innerHTML = options.join('');
      updateCandidatePreview();
    }

    function renderPlot() {
      const host = $('plot');
      const trace = state.best_trace || [];
      const randomTrace = state.random_walk_trace || [];
      const benchmarkRuns = (state.benchmark_runs || []).concat(
        state.live_benchmark_run ? [state.live_benchmark_run] : []
      );
      const statMode = (state.config || {}).plot_stat_guides || 'max';
      const rawDatasetStats = state.dataset_stats || [];
      const datasetStats = statMode === 'paper'
        ? rawDatasetStats
        : (statMode === 'max'
          ? rawDatasetStats.filter((item) => ['max', 'min'].includes(item.label))
          : []);
      if (!trace.length && !benchmarkRuns.length && !randomTrace.length && !datasetStats.length) {
        host.innerHTML = '<div class="empty" style="margin: 18px;">No observations, random baseline, or benchmark runs yet</div>';
        return;
      }
      const obs = state.observations || [];
      const width = Math.max(560, host.clientWidth || 760);
      const height = 330;
      const pad = { left: 56, right: 76, top: 26, bottom: 46 };
      const activeObs = obs.filter((item) => item.value !== null && item.value !== undefined);
      const xIndexes = [
        ...trace.map((item) => Number(item.index)),
        ...randomTrace.map((item) => Number(item.index)),
        ...benchmarkRuns.flatMap((run) => (run.summary || []).map((item) => Number(item.index)))
      ].filter((value) => Number.isFinite(value) && value >= 1);
      const minIndex = 1;
      const maxIndex = Math.max(1, ...xIndexes);
      const values = activeObs.flatMap((item) => {
        const unc = Number(item.uncertainty || 0);
        return [item.value - unc, item.value + unc, item.value];
      }).concat(
        trace.map((item) => item.best),
        randomTrace.map((item) => item.best),
        benchmarkRuns.flatMap((run) => (run.summary || []).flatMap((item) => [item.lower, item.upper, item.mean])),
        datasetStats.map((item) => item.value)
      ).filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value)));
      if (!values.length) {
        host.innerHTML = '<div class="empty" style="margin: 18px;">No plottable values yet</div>';
        return;
      }
      const rawMinY = Math.min(...values);
      const rawMaxY = Math.max(...values);
      let minY = rawMinY;
      let maxY = rawMaxY;
      if (minY === maxY) {
        if (minY >= 0) {
          minY = 0;
          maxY = maxY > 0 ? maxY * 1.08 : 1;
        } else {
          minY -= 1;
          maxY += 1;
        }
      } else {
        const yMargin = (maxY - minY) * 0.08;
        if (rawMinY >= 0) {
          minY = 0;
          maxY += yMargin;
        } else {
          minY -= yMargin;
          maxY += yMargin;
        }
      }
      const x = (i) => pad.left + ((i - minIndex) / Math.max(1, maxIndex - minIndex)) * (width - pad.left - pad.right);
      const y = (value) => pad.top + (1 - ((value - minY) / (maxY - minY))) * (height - pad.top - pad.bottom);
      const pathFor = (items, valueKey) => items.length > 1
        ? items.map((item, idx) => `${idx ? 'L' : 'M'} ${x(item.index).toFixed(1)} ${y(item[valueKey]).toFixed(1)}`).join(' ')
        : '';
      const bestPath = pathFor(trace, 'best');
      const randomPath = pathFor(randomTrace, 'best');
      const randomMarkers = randomTrace.map((item) => (
        `<circle cx="${x(item.index).toFixed(1)}" cy="${y(item.best).toFixed(1)}" r="3" fill="#fff" stroke="#667085" stroke-width="1.5" />`
      )).join('');
      const statLines = datasetStats.map((item, idx) => {
        const yy = y(item.value);
        return `<line x1="${pad.left}" x2="${width - pad.right}" y1="${yy}" y2="${yy}" stroke="#98a2b3" stroke-width="1" stroke-dasharray="4 5" opacity="${idx === 0 ? 0.7 : 0.55}" />
          <text x="${width - pad.right + 8}" y="${yy + 4}">${escapeHtml(item.label)}</text>`;
      }).join('');
      const runLayers = benchmarkRuns.map((run) => {
        const points = run.summary || [];
        if (!points.length) return '';
        const color = run.color || '#7c3aed';
        const meanPath = pathFor(points, 'mean');
        const band = points.length > 1
          ? `M ${points.map((item) => `${x(item.index).toFixed(1)} ${y(item.upper).toFixed(1)}`).join(' L ')} L ${[...points].reverse().map((item) => `${x(item.index).toFixed(1)} ${y(item.lower).toFixed(1)}`).join(' L ')} Z`
          : '';
        const markers = points.map((item) => (
          `<circle cx="${x(item.index).toFixed(1)}" cy="${y(item.mean).toFixed(1)}" r="3.5" fill="#fff" stroke="${color}" stroke-width="2" />`
        )).join('');
        return `${band ? `<path d="${band}" fill="${color}" opacity="0.14" />` : ''}
          ${meanPath ? `<path d="${meanPath}" fill="none" stroke="${color}" stroke-width="3" />` : ''}
          ${markers}`;
      }).join('');
      const points = activeObs.map((item, idx) => {
        const cx = x(idx + 1);
        const cy = y(item.value);
        const unc = Number(item.uncertainty || 0);
        const err = unc ? `<line x1="${cx}" x2="${cx}" y1="${y(item.value - unc)}" y2="${y(item.value + unc)}" stroke="#b45309" stroke-width="1.5" />` : '';
        return `${err}<circle cx="${cx}" cy="${cy}" r="4" fill="#2563eb"><title>${escapeHtml(item.procedure)}: ${fmt(item.value)}</title></circle>`;
      }).join('');
      const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => {
        const value = minY + (maxY - minY) * t;
        const yy = y(value);
        return `<line x1="${pad.left}" x2="${width - pad.right}" y1="${yy}" y2="${yy}" stroke="#edf0f5" />
          <text x="10" y="${yy + 4}">${fmt(value, 3)}</text>`;
      }).join('');
      const zeroLine = minY <= 0 && maxY >= 0
        ? `<line x1="${pad.left}" x2="${width - pad.right}" y1="${y(0)}" y2="${y(0)}" stroke="#9aa4b2" stroke-width="1.2" />`
        : '';
      const xSpan = Math.max(1, maxIndex - minIndex);
      const xStep = xSpan <= 12 ? 1 : (xSpan <= 40 ? 5 : Math.ceil(xSpan / 8));
      const firstRegularTick = Math.ceil(minIndex / xStep) * xStep;
      const xTickValues = Array.from(new Set([
        minIndex,
        ...Array.from(
          { length: Math.floor((maxIndex - firstRegularTick) / xStep) + 1 },
          (_, idx) => firstRegularTick + idx * xStep
        ),
        maxIndex
      ])).filter((value) => value >= minIndex && value <= maxIndex).sort((a, b) => a - b);
      const xTicks = xTickValues.map((value) => {
        const xx = x(value);
        return `<line x1="${xx}" x2="${xx}" y1="${height - pad.bottom}" y2="${height - pad.bottom + 5}" stroke="#98a2b3" />
          <text x="${xx}" y="${height - pad.bottom + 20}" text-anchor="middle">${value}</text>`;
      }).join('');
      const legendItems = [
        bestPath ? { label: 'live best', color: '#0f766e', dash: '' } : null,
        randomPath ? { label: 'random mean', color: '#667085', dash: '5 5' } : null,
        ...benchmarkRuns.map((run) => ({ label: run.name, color: run.color || '#7c3aed', dash: '' }))
      ].filter(Boolean).slice(0, 6);
      const legend = legendItems.map((item, idx) => {
        const xx = pad.left + idx * 112;
        return `<line x1="${xx}" x2="${xx + 22}" y1="16" y2="16" stroke="${item.color}" stroke-width="3" stroke-dasharray="${item.dash}" />
          <text x="${xx + 28}" y="20">${escapeHtml(item.label).slice(0, 16)}</text>`;
      }).join('');
      host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">
        ${legend}
        ${ticks}
        ${zeroLine}
        ${statLines}
        <line x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" stroke="#cfd6e2" />
        <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="#cfd6e2" />
        ${xTicks}
        ${randomPath ? `<path d="${randomPath}" fill="none" stroke="#667085" stroke-width="2" stroke-dasharray="6 5" />` : ''}
        ${randomMarkers}
        ${runLayers}
        ${bestPath ? `<path d="${bestPath}" fill="none" stroke="#0f766e" stroke-width="3" />` : ''}
        ${points}
        <text x="${width / 2 - 44}" y="${height - 12}">experiment count</text>
      </svg>`;
    }

    function renderBenchmarkRuns() {
      const runs = (state.benchmark_runs || []).concat(
        state.live_benchmark_run ? [state.live_benchmark_run] : []
      );
      if (!runs.length) {
        $('benchmarkRuns').innerHTML = '<div class="muted">No offline benchmark runs appended.</div>';
        return;
      }
      $('benchmarkRuns').innerHTML = `<div class="scroll"><table>
        <thead><tr><th>Run</th><th>Status</th><th>Config</th><th>Last best</th><th>Spread</th></tr></thead>
        <tbody>${runs.map((run) => {
          const last = (run.summary || []).slice(-1)[0] || {};
          const cfg = run.config || {};
          const count = (run.replicate_observations || []).reduce((total, items) => total + items.length, 0);
          const status = run.partial ? (run.status || 'partial') : 'complete';
          return `<tr>
            <td><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${escapeHtml(run.color || '#7c3aed')};"></span> ${escapeHtml(run.name)}</td>
            <td>${escapeHtml(status)}${count ? ` (${count} rows)` : ''}</td>
            <td>${escapeHtml(cfg.optimizer || '')} / ${escapeHtml((cfg.acquisition || '').replaceAll('_', ' '))}</td>
            <td>${fmt(last.mean)}</td>
            <td>${fmt(last.std)}</td>
          </tr>`;
        }).join('')}</tbody>
      </table></div>`;
    }

    function renderSuggestions() {
      const suggestions = state.suggestions || [];
      if (!suggestions.length) {
        $('suggestions').innerHTML = '<div class="empty">No suggestions</div>';
        return;
      }
      $('suggestions').innerHTML = `<div class="scroll"><table>
        <thead><tr><th>Procedure</th><th>Acq</th><th>Mean</th><th>Std</th><th></th></tr></thead>
        <tbody>${suggestions.map((sug) => `<tr>
          <td class="procedure">${escapeHtml(sug.procedure)}</td>
          <td>${fmt(sug.acquisition)}</td>
          <td>${fmt(sug.mean)}</td>
          <td>${fmt(sug.std)}</td>
          <td><button data-use="${escapeHtml(sug.candidate_id)}">Use</button></td>
        </tr>`).join('')}</tbody>
      </table></div>`;
      document.querySelectorAll('[data-use]').forEach((button) => {
        button.addEventListener('click', () => {
          $('candidateSelect').value = button.dataset.use;
          const sug = suggestions.find((item) => item.candidate_id === button.dataset.use);
          $('manualProcedure').value = sug ? sug.procedure : '';
          updateCandidatePreview();
          $('objectiveValue').focus();
        });
      });
    }

    function renderInverseDesigns() {
      const designs = state.inverse_designs || [];
      if (!designs.length) {
        $('inverseDesigns').innerHTML = '<div class="empty">No inverse designs</div>';
        return;
      }
      $('inverseDesigns').innerHTML = `<div class="scroll"><table>
        <thead><tr><th>Proposal</th><th>Target</th><th>Model</th><th></th></tr></thead>
        <tbody>${designs.map((design, idx) => `<tr>
          <td class="procedure">${escapeHtml(design.procedure)}</td>
          <td>${fmt(design.target)}</td>
          <td>${escapeHtml(design.model || '')}</td>
          <td><button data-inverse-use="${idx}">Use</button></td>
        </tr>`).join('')}</tbody>
      </table></div>`;
      document.querySelectorAll('[data-inverse-use]').forEach((button) => {
        button.addEventListener('click', () => {
          const design = designs[Number(button.dataset.inverseUse)];
          $('candidateSelect').value = '';
          $('manualProcedure').value = design ? design.procedure : '';
          updateCandidatePreview();
          $('objectiveValue').focus();
        });
      });
    }

    function renderObservations() {
      const observations = state.observations || [];
      if (!observations.length) {
        $('observations').innerHTML = '<div class="empty">No observations</div>';
        return;
      }
      $('observations').innerHTML = `<div class="scroll"><table>
        <thead><tr><th>#</th><th>Procedure</th><th>Value</th><th>Unc.</th><th>Time</th></tr></thead>
        <tbody>${observations.map((obs, idx) => `<tr>
          <td>${idx + 1}</td>
          <td class="procedure">${escapeHtml(obs.procedure)}</td>
          <td>${fmt(obs.value)}</td>
          <td>${fmt(obs.uncertainty)}</td>
          <td>${escapeHtml(obs.time)}</td>
        </tr>`).join('')}</tbody>
      </table></div>`;
    }

    function renderMessages() {
      const messages = [];
      if (transientNotice) {
        messages.push(`<div class="notice">${escapeHtml(transientNotice)}</div>`);
      }
      if (state.last_error) {
        messages.push(`<div class="notice error">${escapeHtml(state.last_error)}</div>`);
      }
      (state.events || []).slice(0, 4).forEach((event) => {
        messages.push(`<div class="notice"><strong>${escapeHtml(event.time)}</strong> ${escapeHtml(event.message)}</div>`);
      });
      $('messages').innerHTML = messages.join('');
    }

    function renderError(message) {
      $('messages').innerHTML = `<div class="notice error">${escapeHtml(message)}</div>` + $('messages').innerHTML;
    }

    function renderNotice(message) {
      transientNotice = message;
      renderMessages();
    }

    async function handlePoolImported(data = {}) {
      await refresh();
      const count = data.candidate_count || (state ? state.candidate_count : 0);
      const objective = data.objective_name ? ` Objective: ${data.objective_name}.` : '';
      const saved = data.saved ? ' It was autosaved into the current campaign.' : ' Save the campaign to keep it for later.';
      renderNotice(`Pool Builder imported ${count} candidate(s) into this runner.${objective}${saved}`);
      window.focus();
    }

    window.addEventListener('message', (event) => {
      if (event.origin !== window.location.origin) return;
      if (!event.data || event.data.type !== 'boicl-pool-imported') return;
      handlePoolImported(event.data);
    });
    if (poolChannel) {
      poolChannel.addEventListener('message', (event) => {
        if (!event.data || event.data.type !== 'boicl-pool-imported') return;
        handlePoolImported(event.data);
      });
    }

    async function stopCurrentRun() {
      try {
        $('stopRun').disabled = true;
        const response = await fetch('/api/cancel', { method: 'POST' });
        const payload = await response.json();
        renderProgress(payload.progress || {});
      } catch (error) {
        renderError(error.message);
      }
    }

    async function updateSuggestions() {
      await request('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadConfig())
      });
      await request('/api/suggest', { method: 'POST' });
    }

    $('saveKey').addEventListener('click', async () => {
      await request('/api/save-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          openai_api_key: $('openaiKey').value,
          openrouter_api_key: $('openrouterKey').value,
          anthropic_api_key: $('anthropicKey').value
        })
      });
      $('openaiKey').value = '';
      $('openrouterKey').value = '';
      $('anthropicKey').value = '';
    });

    $('saveCampaign').addEventListener('click', async () => {
      await request('/api/save-campaign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: $('campaignName').value, save_as: false })
      });
    });

    $('saveAsCampaign').addEventListener('click', async () => {
      await request('/api/save-campaign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: $('campaignName').value, save_as: true })
      });
    });

    $('loadCampaign').addEventListener('click', async () => {
      if (!$('savedCampaign').value) return renderError('Choose a saved campaign.');
      await request('/api/load-campaign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: $('savedCampaign').value })
      });
    });

    $('deleteCampaign').addEventListener('click', async () => {
      if (!$('savedCampaign').value) return renderError('Choose a saved campaign.');
      if (!window.confirm('Delete this saved campaign from disk?')) return;
      await request('/api/delete-campaign', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: $('savedCampaign').value })
      });
    });

    $('importDataset').addEventListener('click', async () => {
      const file = $('datasetFile').files[0];
      if (!file) return renderError('Choose a CSV, Excel, or NPY file.');
      await request(`/api/import-dataset?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        body: await file.arrayBuffer()
      });
    });

    $('prepareEmbeddings').addEventListener('click', async () => {
      await request('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadConfig())
      });
      await request('/api/precompute-embeddings', { method: 'POST' });
    });

    $('regeneratePrompts').addEventListener('click', async () => {
      await request('/api/regenerate-prompts', { method: 'POST' });
    });

    $('saveConfig').addEventListener('click', async () => {
      await request('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadConfig())
      });
    });

    $('addObservation').addEventListener('click', async () => {
      await request('/api/observe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          candidate_id: $('candidateSelect').value,
          procedure: $('manualProcedure').value,
          value: $('objectiveValue').value,
          uncertainty: $('objectiveUncertainty').value
        })
      });
      $('objectiveValue').value = '';
      $('objectiveUncertainty').value = '';
      if (state && state.config.auto_suggest) await updateSuggestions();
    });

    $('candidateSelect').addEventListener('change', () => {
      const id = $('candidateSelect').value;
      const cand = (state.candidates || []).find((item) => item.id === id);
      const sug = (state.suggestions || []).find((item) => item.candidate_id === id);
      $('manualProcedure').value = cand ? cand.procedure : (sug ? sug.procedure : '');
      updateCandidatePreview();
    });

    $('suggest').addEventListener('click', updateSuggestions);
    $('suggestTop').addEventListener('click', updateSuggestions);
    $('openPoolBuilderTop').addEventListener('click', openPoolBuilder);
    $('openPoolBuilderDataset').addEventListener('click', openPoolBuilder);
    $('stopRun').addEventListener('click', stopCurrentRun);
    $('workflowMode').addEventListener('change', () => {
      if (!state) return;
      state.config.workflow_mode = $('workflowMode').value;
      renderWorkflowMode();
    });
    $('inverseDesign').addEventListener('click', async () => {
      await request('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadConfig())
      });
      await request('/api/inverse-design', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_value: $('inverseTargetValue').value,
          count: $('inverseDesignCount').value
        })
      });
    });
    $('resetRun').addEventListener('click', () => request('/api/reset', { method: 'POST' }));
    $('runBenchmark').addEventListener('click', async () => {
      await request('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payloadConfig())
      });
      await request('/api/run-benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: $('benchmarkName').value })
      });
    });
    $('resumeBenchmark').addEventListener('click', async () => {
      const run = latestPartialBenchmarkRun();
      if (!run) return;
      await request('/api/run-benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payloadConfig(),
          resume_id: run.id,
          name: run.name
        })
      });
    });
    $('clearBenchmarks').addEventListener('click', () => request('/api/clear-benchmarks', { method: 'POST' }));

    applyHelpText();
    refresh();
  </script>
</body>
</html>
"""


POOL_BUILDER_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BO-ICL Pool Builder</title>
  <style>
    :root {
      --text: #182230;
      --muted: #667085;
      --line: #d9dee7;
      --panel: #fff;
      --bg: #f6f7f8;
      --accent: #0f766e;
      --accent2: #1d4ed8;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 20px; line-height: 1.2; margin: 0; letter-spacing: 0; }
    h2 { font-size: 15px; margin: 0 0 14px; letter-spacing: 0; }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 430px) 1fr;
      gap: 16px;
      padding: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(18, 31, 53, 0.06);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .range-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      align-items: end;
    }
    .range-grid span {
      display: block;
      font-size: 11px;
      font-weight: 720;
      color: var(--muted);
      margin: 0 0 4px;
    }
    .field { margin-bottom: 12px; }
    label {
      display: block;
      font-size: 12px;
      font-weight: 720;
      color: #344054;
      margin-bottom: 5px;
    }
    input, select, button {
      font: inherit;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      padding: 9px 10px;
      color: var(--text);
    }
    button, a.button-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: #fff;
      color: var(--text);
      padding: 9px 12px;
      cursor: pointer;
      text-decoration: none;
      font-size: 13px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button.secondary {
      background: #eff6ff;
      border-color: #bfdbfe;
      color: var(--accent2);
    }
    .toolbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .muted {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .status {
      border-left: 3px solid var(--accent);
      background: #f0fdfa;
      color: #134e48;
      padding: 10px 12px;
      border-radius: 6px;
      font-size: 13px;
      line-height: 1.4;
      margin-top: 12px;
    }
    .status.error {
      border-left-color: var(--bad);
      background: #fef3f2;
      color: var(--bad);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid #edf0f5;
      text-align: left;
      vertical-align: top;
      padding: 8px;
    }
    th {
      color: #475467;
      background: #fbfcfe;
      font-size: 12px;
    }
    td.procedure {
      max-width: 760px;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }
    .scroll {
      max-height: calc(100vh - 180px);
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 5px 9px;
      border-radius: 999px;
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: var(--accent2);
      font-size: 12px;
      white-space: nowrap;
    }
    @media (max-width: 940px) {
      header { align-items: flex-start; flex-direction: column; }
      main { grid-template-columns: 1fr; }
      .row, .range-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Pool Builder</h1>
      <div class="muted">WO3/SiO2 reduction template</div>
    </div>
    <div class="toolbar">
      <span class="pill" id="countPill">0 procedures</span>
      <button id="focusRunner">Open/Focus Runner</button>
    </div>
  </header>

  <main>
    <section class="panel">
      <h2>Template</h2>
      <div class="field">
        <label for="systemName">System</label>
        <input id="systemName" value="WO3/SiO2">
      </div>
      <div class="row">
        <div class="field">
          <label for="gasName">Gas</label>
          <input id="gasName" value="pure H2">
        </div>
        <div class="field">
          <label for="flowRate">Flow (mL/min)</label>
          <input id="flowRate" type="number" step="any" value="40">
        </div>
        <div class="field">
          <label for="stabilizeTime">Stabilize (min)</label>
          <input id="stabilizeTime" type="number" step="any" value="30">
        </div>
      </div>
      <div class="field">
        <label for="sampleTube">Sample holder</label>
        <input id="sampleTube" value="quartz sample tube">
      </div>
      <div class="field">
        <label for="coolingText">Cooling</label>
        <input id="coolingText" value="passively cool to ambient temperature">
      </div>
      <div class="field">
        <label for="phaseObjective">Live objective</label>
        <select id="phaseObjective" title="This sets the main runner objective name. Enter the XRD-calculated percentage for this phase in Add Result during live experiments.">
          <option value="alpha phase (%)">Alpha phase (%) - Im-3m</option>
          <option value="beta phase (%)">Beta phase (%) - Pm-3n</option>
        </select>
      </div>

      <h2>Variables</h2>
      <div class="field">
        <label>Ramp rate (C/min)</label>
        <div class="range-grid">
          <label><span>Minimum</span><input id="rampMin" type="number" step="any" value="2" aria-label="Ramp minimum"></label>
          <label><span>Maximum</span><input id="rampMax" type="number" step="any" value="20" aria-label="Ramp maximum"></label>
          <label><span>Step</span><input id="rampStep" type="number" step="any" value="2" aria-label="Ramp step"></label>
        </div>
      </div>
      <div class="field">
        <label>Maximum temperature (C)</label>
        <div class="range-grid">
          <label><span>Minimum</span><input id="tempMin" type="number" step="any" value="350" aria-label="Temperature minimum"></label>
          <label><span>Maximum</span><input id="tempMax" type="number" step="any" value="700" aria-label="Temperature maximum"></label>
          <label><span>Step</span><input id="tempStep" type="number" step="any" value="50" aria-label="Temperature step"></label>
        </div>
      </div>
      <div class="field">
        <label>Dwell time (h)</label>
        <div class="range-grid">
          <label><span>Minimum</span><input id="dwellMin" type="number" step="any" value="1" aria-label="Dwell minimum"></label>
          <label><span>Maximum</span><input id="dwellMax" type="number" step="any" value="8" aria-label="Dwell maximum"></label>
          <label><span>Step</span><input id="dwellStep" type="number" step="any" value="1" aria-label="Dwell step"></label>
        </div>
      </div>
      <div class="row">
        <div class="field">
          <label for="poolSize">Pool size</label>
          <input id="poolSize" type="text" value="0" readonly title="Calculated automatically from the number of ramp-rate, temperature, and dwell-time combinations.">
        </div>
        <div class="field">
          <label for="previewCount">Preview rows</label>
          <input id="previewCount" type="number" min="5" step="1" value="20">
        </div>
      </div>
      <div class="toolbar">
        <button class="secondary" id="previewPool">Preview</button>
        <button class="primary" id="importPool">Import to Runner</button>
        <button id="downloadPool">Download CSV</button>
      </div>
      <div class="status" id="statusBox">Ready.</div>
    </section>

    <section class="panel">
      <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
        <h2 style="margin:0;">Preview</h2>
        <span class="muted" id="combinationNote"></span>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Procedure</th>
            </tr>
          </thead>
          <tbody id="previewRows"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    window.name = 'boicl_pool_builder';
    const poolChannel = 'BroadcastChannel' in window ? new BroadcastChannel('boicl-local-runner') : null;
    const $ = (id) => document.getElementById(id);
    let currentRows = [];

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function numericValue(id) {
      const value = Number($(id).value);
      if (!Number.isFinite(value)) throw new Error(`${id} must be numeric.`);
      return value;
    }

    function formatNumber(value) {
      if (Math.abs(value - Math.round(value)) < 1e-9) return String(Math.round(value));
      return Number(value.toFixed(4)).toString();
    }

    function rangeValues(min, max, step, label) {
      if (step <= 0) throw new Error(`${label} step must be greater than zero.`);
      if (max < min) throw new Error(`${label} maximum must be greater than or equal to minimum.`);
      const values = [];
      const maxCount = 10000;
      for (let index = 0; index < maxCount; index += 1) {
        const value = min + index * step;
        if (value > max + step * 1e-9) break;
        values.push(Number(value.toFixed(8)));
      }
      if (!values.length) throw new Error(`${label} range produced no values.`);
      return values;
    }

    function procedureText(row) {
      const system = $('systemName').value.trim() || 'WO3/SiO2';
      const gas = $('gasName').value.trim() || 'pure H2';
      const holder = $('sampleTube').value.trim() || 'quartz sample tube';
      const flow = formatNumber(numericValue('flowRate'));
      const stabilize = formatNumber(numericValue('stabilizeTime'));
      const cooling = $('coolingText').value.trim() || 'passively cool to ambient temperature';
      return `Reduction experiment of ${system}: load the catalyst into a ${holder} and flow ${gas} at ${flow} mL/min. Stabilize under ${gas} flow for ${stabilize} min, ramp to ${formatNumber(row.temperature)} C at ${formatNumber(row.ramp)} C/min, soak for ${formatNumber(row.dwell)} h, then ${cooling}.`;
    }

    function buildRows() {
      const ramps = rangeValues(numericValue('rampMin'), numericValue('rampMax'), numericValue('rampStep'), 'Ramp rate');
      const temps = rangeValues(numericValue('tempMin'), numericValue('tempMax'), numericValue('tempStep'), 'Temperature');
      const dwells = rangeValues(numericValue('dwellMin'), numericValue('dwellMax'), numericValue('dwellStep'), 'Dwell time');
      const allRows = [];
      for (const temp of temps) {
        for (const ramp of ramps) {
          for (const dwell of dwells) {
            allRows.push({ ramp, temperature: temp, dwell });
          }
        }
      }
      return {
        rows: allRows.map((row) => ({ ...row, procedure: procedureText(row) })),
        total: allRows.length,
      };
    }

    function toCsv(rows) {
      const escape = (value) => `"${String(value).replaceAll('"', '""')}"`;
      return ['procedure'].concat(rows.map((row) => escape(row.procedure))).join('\n') + '\n';
    }

    function setStatus(message, isError = false) {
      $('statusBox').textContent = message;
      $('statusBox').classList.toggle('error', isError);
    }

    function focusRunner() {
      if (window.opener && !window.opener.closed) {
        window.opener.focus();
        return;
      }
      const runner = window.open('/', 'boicl_runner');
      if (runner) runner.focus();
    }

    function renderPreview() {
      try {
        const result = buildRows();
        currentRows = result.rows;
        const previewCount = Math.max(5, Math.trunc(numericValue('previewCount')));
        $('countPill').textContent = `${currentRows.length} procedures`;
        $('poolSize').value = currentRows.length.toLocaleString();
        $('combinationNote').textContent = `${result.total} total combinations`;
        $('previewRows').innerHTML = currentRows.slice(0, previewCount).map((row, index) => `
          <tr>
            <td>${index + 1}</td>
            <td class="procedure">${escapeHtml(row.procedure)}</td>
          </tr>
        `).join('');
        setStatus(`Generated ${currentRows.length} procedure-only row(s). Numeric variables are embedded in the procedure text.`);
      } catch (error) {
        currentRows = [];
        $('countPill').textContent = '0 procedures';
        $('poolSize').value = '0';
        $('combinationNote').textContent = '';
        $('previewRows').innerHTML = '';
        setStatus(error.message, true);
      }
    }

    async function importPool() {
      renderPreview();
      if (!currentRows.length) return;
      try {
        const csv = toCsv(currentRows);
        const objective = encodeURIComponent($('phaseObjective').value);
        const response = await fetch(`/api/import-dataset?filename=wo3_sio2_reduction_pool.csv&objective_name=${objective}`, {
          method: 'POST',
          headers: { 'Content-Type': 'text/csv; charset=utf-8' },
          body: csv,
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Import failed.');
        const message = {
          type: 'boicl-pool-imported',
          candidate_count: payload.candidate_count,
          objective_name: (payload.config || {}).objective_name,
          saved: Boolean((payload.campaign || {}).saved),
        };
        if (poolChannel) {
          poolChannel.postMessage(message);
        } else if (window.opener) {
          window.opener.postMessage(message, window.location.origin);
        }
        setStatus(
          `Imported ${payload.candidate_count} candidate(s) into the runner. `
          + `${message.saved ? 'The current campaign was autosaved.' : 'Save the campaign in the runner to keep it for later.'} `
          + `Use Open/Focus Runner to continue.`
        );
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    function downloadPool() {
      renderPreview();
      if (!currentRows.length) return;
      const blob = new Blob([toCsv(currentRows)], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'wo3_sio2_reduction_pool.csv';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    }

    [
      'systemName', 'gasName', 'flowRate', 'stabilizeTime', 'sampleTube',
      'coolingText', 'rampMin', 'rampMax', 'rampStep', 'tempMin', 'tempMax',
      'tempStep', 'dwellMin', 'dwellMax', 'dwellStep', 'previewCount',
      'phaseObjective'
    ].forEach((id) => {
      $(id).addEventListener('input', renderPreview);
      $(id).addEventListener('change', renderPreview);
    });
    $('previewPool').addEventListener('click', renderPreview);
    $('importPool').addEventListener('click', importPool);
    $('downloadPool').addEventListener('click', downloadPool);
    $('focusRunner').addEventListener('click', focusRunner);
    renderPreview();
  </script>
</body>
</html>
"""


USER_GUIDE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BO-ICL Local Runner Guide</title>
  <style>
    body {
      margin: 0;
      font-family: Inter, Segoe UI, system-ui, -apple-system, sans-serif;
      color: #17202a;
      background: #f6f7f8;
      line-height: 1.55;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 28px 22px 44px;
    }
    section {
      background: #fff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 18px 20px;
      margin: 0 0 16px;
      box-shadow: 0 8px 24px rgba(18, 31, 53, 0.08);
    }
    h1 { font-size: 26px; margin: 0 0 10px; letter-spacing: 0; }
    h2 { font-size: 18px; margin: 0 0 8px; letter-spacing: 0; }
    h3 { font-size: 15px; margin: 16px 0 6px; letter-spacing: 0; }
    code {
      background: #eef2f6;
      border-radius: 5px;
      padding: 2px 5px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      margin-top: 8px;
    }
    th, td {
      border-bottom: 1px solid #edf0f5;
      text-align: left;
      vertical-align: top;
      padding: 8px;
    }
    th { color: #475467; background: #fbfcfe; }
    .muted { color: #667085; }
    .callout {
      border-left: 3px solid #0f766e;
      background: #f0fdfa;
      padding: 10px 12px;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <main>
    <h1>BO-ICL Local Runner Guide</h1>
    <p class="muted">Use this local browser tool for Bayesian-optimization active learning over a finite pool of procedures.</p>

    <section>
      <h2>Saving Long-Running Campaigns</h2>
      <p>Live experimental campaigns can run for days or weeks. Save the campaign once before you start, then the app autosaves later changes into a local folder under <code>saved_experiments/</code>. That folder is ignored by Git so lab data and API context stay on this computer.</p>
      <ol>
        <li>Enter a <code>Campaign name</code> and click <code>Save</code>.</li>
        <li>Import the dataset, choose settings, add observations, and update suggestions as usual.</li>
        <li>Close the browser whenever needed. The saved campaign contains the uploaded pool, labels if present, observations, suggestions, settings, inverse designs, and benchmark runs.</li>
        <li>Later, start the local runner, choose the campaign in <code>Saved campaigns</code>, and click <code>Load</code>.</li>
      </ol>
      <p>Use <code>Save As New</code> to branch a campaign into a separate local copy before trying a different strategy.</p>
    </section>

    <section>
      <h2>Embedding Cache</h2>
      <p>Embeddings are saved locally under <code>.cache/</code>, one cache per embedding model. After importing a dataset, click <code>Prepare Embeddings</code> once to embed the full candidate pool. Later GPR runs, inverse-filter steps, repeated benchmark configurations, and app restarts reuse those saved embeddings instead of embedding the same procedures again.</p>
      <p>If you change the embedding model, prepare embeddings again for that model. Existing cache files remain available for the previous model.</p>
      <p>Long-running actions show a progress panel in the browser and print progress lines in the terminal window. Browser controls may be disabled while an action is running, but the progress panel should continue updating.</p>
    </section>

    <section>
      <h2>Accepted Dataset Formats</h2>
      <p>The first column is always interpreted as the procedure text. Later numeric columns are treated as objective labels. Columns with names containing <code>uncert</code>, <code>std</code>, <code>stdev</code>, <code>sigma</code>, or <code>error</code> are treated as uncertainty columns.</p>
      <table>
        <thead><tr><th>Format</th><th>Expected shape</th></tr></thead>
        <tbody>
          <tr><td><code>.csv</code>, <code>.txt</code></td><td>Comma-separated table. First column procedures, later numeric objective columns.</td></tr>
          <tr><td><code>.xlsx</code>, <code>.xls</code></td><td>First sheet is read as a table using the same first-column procedure rule.</td></tr>
          <tr><td><code>.npy</code></td><td>1D array for procedure-only pools, 2D array with first column procedures and later objective columns, or structured array with named fields.</td></tr>
        </tbody>
      </table>
      <h3>Example CSV</h3>
      <pre><code>procedure,C2 yield,C2 yield uncertainty,selectivity
"Procedure A",12.4,0.3,71.0
"Procedure B",10.1,0.4,77.5
"Procedure C",,,</code></pre>
    </section>

    <section>
      <h2>Procedure Pool Builder</h2>
      <p>Open <code>Pool Builder</code> to generate an unlabeled live-experiment pool for the WO3/SiO2 reduction template. The app reuses one Pool Builder window instead of opening a fresh tab every time. The first version varies ramp rate, maximum temperature, and dwell time while keeping gas, flow rate, stabilization time, holder, and cooling text fixed.</p>
      <p>The builder imports a procedure-only CSV so the generated numeric variables are not mistaken for objective labels. They are embedded directly inside each procedure string, which keeps the main runner in live-campaign mode.</p>
      <p>After import, the builder shows an import confirmation and the runner refreshes with a matching notice. If the campaign is not saved yet, click <code>Save</code> in the runner to keep the generated pool for later.</p>
      <p><code>Pool size</code> is calculated from the minimum, maximum, and step settings for the three variables. The live objective selector sets the main runner to maximize either <code>alpha phase (%)</code> for Im-3m or <code>beta phase (%)</code> for Pm-3n; enter the XRD-calculated percentage from 0 to 100 with <code>Add Result</code>. Use whole percent units: <code>73.5</code> means 73.5%, not 0.735.</p>
    </section>

    <section>
      <h2>Dataset-Aware System Messages</h2>
      <p>When a dataset is imported, the app generates BO-ICL prediction and inverse-design system messages from the uploaded procedure style. These prompts include the number of candidates, objective column names, and a few procedure examples, but they do not include hidden label values or label statistics.</p>
      <p>Use <code>Use Dataset Prompts</code> whenever you switch to a new dataset or want to replace an old OCM-specific prompt. You can still edit either prompt manually for a campaign-specific constraint, such as requiring a named phase, a fixed instrument, or a permitted parameter range.</p>
    </section>

    <section>
      <h2>Paper-Style Offline Benchmark</h2>
      <p>Use this when you already have labels for the full pool and want to test BO performance without revealing labels to the model until each simulated experiment is selected.</p>
      <ol>
        <li>Import a labeled dataset.</li>
        <li>Set <code>Workflow mode</code> to <code>Automatic benchmark: full labeled dataset</code>. The app switches to this automatically when imported labels are detected.</li>
        <li>Choose the active objective and whether to maximize or minimize it.</li>
        <li>Choose the suggestion engine, model, acquisition function, and target scaling.</li>
        <li>Set <code>Initial random</code>, <code>BO iterations</code>, <code>Workflow replicates</code>, and <code>Seed</code>. <code>Seed</code> is the reproducible random-number seed, not an objective-value starting point. Use <code>Starting baseline</code> when you want the first plotted marker at x=1 to be the dataset-mean incumbent; scored experiments then advance from the next x position. Optionally enable <code>Greedy for final iteration</code> so only the last BO choice in each replicate switches to greedy exploitation.</li>
        <li>Click <code>Run & Append</code>. Change settings and click it again to compare another configuration.</li>
      </ol>
      <div class="callout">Paper-style numerical defaults are <code>Initial random = 1</code>, <code>Batch size = 1</code>, <code>BO iterations = 30</code>, <code>Workflow replicates = 5</code>, and <code>UCB lambda = 0.1</code>. Current model defaults use supported modern model IDs rather than retired paper-era model names.</div>
      <p>For BO-ICL LLM runs on large pools, <code>Broad pool</code> is the wider random pool considered first, and <code>LLM shortlist</code> retrieves the smaller inverse-design/embedding shortlist that is actually scored by the LLM. The inverse-design target is based on the current replicate's labeled history: by default it uses current best x <code>Normal(1.2, 0.05)</code>, matching the paper-style stochastic target. If sparse observations are often zero, set <code>Auto target floor</code> to a meaningful minimum aspirational value so the inverse query does not stay anchored at zero. The shortlist uses cached embeddings with MMR/cosine similarity, then optional random add-ons. The default <code>Broad pool = 250</code>, <code>LLM shortlist = 16</code>, <code>Random add-ons = 0</code>, and <code>LLM samples = 3</code> scores at most 16 candidates per BO step.</p>
      <p>LLM runtime scales with <code>(LLM shortlist + Random add-ons) x LLM samples x BO iterations x Workflow replicates</code> when the shortlist is enabled. If <code>LLM shortlist = 0</code>, runtime falls back to <code>Broad pool x LLM samples</code>. Rate-limit errors are retried automatically; increase <code>429 cooldown (s)</code>, increase <code>API pause (s)</code>, or lower the shortlist/samples if 429s keep appearing. Use the <code>Stop</code> button in the progress panel to cancel after the current API call returns.</p>
      <p>The plot shows the mean best-so-far trajectory and a +/- 1 standard deviation band. The dashed random baseline is the paper notebook's random-mean quantile expectation. <code>Plot guides</code> defaults to the best labelled value only; switch it to <code>Paper stats</code> to add the mean and percentile guide lines.</p>
    </section>

    <section>
      <h2>Live Experimentation</h2>
      <p>Use this when you have a pool of procedures but labels are produced by experiments over time.</p>
      <ol>
        <li>Import a procedure pool. Labels can be blank or absent.</li>
        <li>Set <code>Workflow mode</code> to <code>Live campaign: add results manually</code>. The app switches to this automatically when no labels are detected.</li>
        <li>Add one or two initial measured results with <code>Add Observation</code>.</li>
        <li>Click <code>Update Suggestions</code> to choose the next candidate from the uploaded pool.</li>
        <li>Run the physical experiment offline, then enter the measured value and optional uncertainty.</li>
        <li>Repeat until the iteration cap is reached or you decide to stop.</li>
      </ol>
      <p>Objective values should be entered in original units. Scaling is only used internally for fitting if enabled, and the plot remains in original units.</p>
    </section>

    <section>
      <h2>Settings Reference</h2>
      <table>
        <thead><tr><th>Setting</th><th>Meaning</th></tr></thead>
        <tbody>
          <tr><td>Suggestion engine</td><td><code>GPR with embeddings</code> uses OpenAI embeddings plus a Gaussian process. <code>BO-ICL LLM</code> uses the selected LLM for in-context predictions.</td></tr>
          <tr><td>Acquisition</td><td>Rule for ranking the next experiment. UCB balances mean and uncertainty; expected improvement favors likely gains; greedy uses predicted best; random is a control.</td></tr>
          <tr><td>Target scaling</td><td>Off by default. Auto/min-max/z-score can help GPR numerics when bounded labels are not already near unit scale.</td></tr>
          <tr><td>Broad pool</td><td>Caps the wider candidate pool considered first. GPR scores this many directly; BO-ICL LLM narrows this pool to the LLM shortlist before requesting completions.</td></tr>
          <tr><td>LLM shortlist</td><td>Number of candidates retrieved by inverse-design text plus embeddings before LLM scoring. Set to 0 only when you intentionally want to score the full broad pool with the LLM.</td></tr>
          <tr><td>Auto target jitter</td><td>Stochastic spread around the automatic inverse-design target multiplier. The default <code>0.05</code> gives current best x <code>Normal(1.2, 0.05)</code>; set it to <code>0</code> for deterministic targets.</td></tr>
          <tr><td>Auto target floor</td><td>Optional minimum automatic inverse-design target for sparse-zero maximization campaigns. For phase percentages, use whole percent units such as <code>5</code> or <code>10</code>. Manual inverse target overrides it.</td></tr>
          <tr><td>Greedy for final iteration</td><td>Offline benchmark option that keeps the selected acquisition for earlier BO choices, then uses greedy acquisition for the final BO choice in each replicate.</td></tr>
          <tr><td>API pause / 429 cooldown / 429 retries</td><td><code>API pause</code> spaces out successful calls. <code>429 cooldown</code> waits after a rate-limit error before retrying. Retries controls how many recovery attempts are allowed before the partial run is saved for resume.</td></tr>
          <tr><td>Replicates</td><td>Live-mode repeated measurements allowed for the same candidate before it is removed from the available pool.</td></tr>
          <tr><td>Workflow replicates</td><td>Offline benchmark repeated runs of the whole BO workflow for averaging and spread bands.</td></tr>
          <tr><td>Starting baseline</td><td>Offline benchmark plot option. <code>Dataset mean incumbent</code> draws the mean incumbent as the first marker at x=1 and initializes best-so-far from that value, without adding a fake labeled procedure to the model context.</td></tr>
          <tr><td>API keys</td><td>Keys are written only to the local ignored <code>.env</code> file. OpenAI keys are required for embeddings and OpenAI LLMs; OpenRouter and Anthropic keys are only needed for those model families.</td></tr>
          <tr><td>System messages</td><td>Generated from the uploaded dataset by default. They constrain the LLM to numeric predictions or procedure-style inverse designs without exposing hidden labels.</td></tr>
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


def _find_port(preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("Could not find an open local port.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the local BO-ICL browser app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    root = Path.cwd()
    port = _find_port(args.port)
    LocalAppHandler.state = LocalBOState(root)
    server = ThreadingHTTPServer((args.host, port), LocalAppHandler)
    url = f"http://{args.host}:{port}"
    print(f"BO-ICL local runner is available at {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BO-ICL local runner.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
