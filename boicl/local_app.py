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
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

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


DEFAULT_CONFIG = {
    "objective_name": "objective",
    "objective_direction": "maximize",
    "acquisition": "upper_confidence_bound",
    "batch_size": 1,
    "ucb_lambda": 0.5,
    "score_limit": 250,
    "n_neighbors": 5,
    "n_components": 16,
    "auto_suggest": True,
}


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
    raise ValueError("Use a CSV or Excel file.")


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


def _best_trace(observations: List[Dict[str, Any]], direction: str) -> List[Dict[str, Any]]:
    best = None
    trace = []
    for index, obs in enumerate(observations, start=1):
        value = obs["value"]
        if best is None:
            best = value
        elif direction == "minimize":
            best = min(best, value)
        else:
            best = max(best, value)
        trace.append({"index": index, "value": value, "best": best})
    return trace


def _load_env_file(env_path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(env_path, override=False)
        return
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _write_env_value(env_path: Path, key: str, value: str) -> None:
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
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, str]] = field(default_factory=list)
    last_error: Optional[str] = None
    last_model_status: str = "No dataset loaded."

    def __post_init__(self) -> None:
        self.lock = threading.RLock()
        self.env_path = self.root / ".env"
        self.cache_dir = self.root / ".cache"
        self.embedding_cache_path = self.cache_dir / "boicl_embeddings.csv"
        _load_env_file(self.env_path)

    def log(self, message: str) -> None:
        self.events.insert(0, {"time": _now(), "message": message})
        self.events = self.events[:20]

    def key_status(self) -> Dict[str, Any]:
        key = os.environ.get("OPENAI_API_KEY", "")
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        return {
            "openai_configured": bool(key),
            "openrouter_configured": bool(openrouter_key),
            "openai_hint": f"...{key[-4:]}" if len(key) > 8 else "",
        }

    def to_json(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "config": self.config,
                "key_status": self.key_status(),
                "candidates": self.candidates[:500],
                "candidate_count": len(self.candidates),
                "available_count": len(self.available_candidates()),
                "observations": self.observations,
                "suggestions": self.suggestions,
                "events": self.events,
                "last_error": self.last_error,
                "last_model_status": self.last_model_status,
                "best_trace": _best_trace(
                    self.observations, self.config["objective_direction"]
                ),
                "acquisition_functions": ACQUISITION_FUNCTIONS,
            }

    def available_candidates(self) -> List[Dict[str, Any]]:
        evaluated_ids = {obs.get("candidate_id") for obs in self.observations}
        evaluated_procs = {obs["procedure"] for obs in self.observations}
        return [
            cand
            for cand in self.candidates
            if cand["id"] not in evaluated_ids and cand["procedure"] not in evaluated_procs
        ]

    def target_value(self, value: float) -> float:
        return -value if self.config["objective_direction"] == "minimize" else value

    def display_value(self, value: float) -> float:
        return -value if self.config["objective_direction"] == "minimize" else value

    def import_dataset(self, filename: str, raw: bytes) -> Dict[str, Any]:
        df = _read_table_from_bytes(filename, raw)
        _clean_procedures(df)
        first_col = df.columns[0]
        with self.lock:
            self.candidates = []
            self.observations = []
            self.suggestions = []
            self.last_error = None
            self.last_model_status = "Dataset loaded. Add observations to train GPR."

            value_col = df.columns[1] if len(df.columns) > 1 else None
            unc_col = df.columns[2] if len(df.columns) > 2 else None
            for row_index, row in df.iterrows():
                if pd.isna(row[first_col]) or not str(row[first_col]).strip():
                    continue
                procedure = str(row[first_col]).strip()
                candidate = {
                    "id": f"cand-{len(self.candidates)}",
                    "row": int(row_index) + 1,
                    "procedure": procedure,
                }
                self.candidates.append(candidate)
                if value_col is not None:
                    value = _coerce_float(row[value_col])
                    if value is None:
                        continue
                    uncertainty = _coerce_float(row[unc_col]) if unc_col is not None else None
                    self._add_observation_locked(
                        procedure,
                        value,
                        uncertainty,
                        candidate_id=candidate["id"],
                    )
            self.log(
                f"Imported {len(self.candidates)} candidates from {Path(filename).name}."
            )
            if self.observations:
                self.log(f"Loaded {len(self.observations)} existing objective values.")
            return self.to_json()

    def update_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            for key in DEFAULT_CONFIG:
                if key not in payload:
                    continue
                value = payload[key]
                if key in {
                    "batch_size",
                    "score_limit",
                    "n_neighbors",
                    "n_components",
                }:
                    value = int(value)
                elif key == "ucb_lambda":
                    value = float(value)
                elif key == "auto_suggest":
                    value = bool(value)
                self.config[key] = value
            if self.config["acquisition"] not in ACQUISITION_FUNCTIONS:
                self.config["acquisition"] = DEFAULT_CONFIG["acquisition"]
            self.config["objective_direction"] = (
                "minimize"
                if self.config["objective_direction"] == "minimize"
                else "maximize"
            )
            self.config["batch_size"] = max(1, min(25, int(self.config["batch_size"])))
            self.config["score_limit"] = max(1, int(self.config["score_limit"]))
            self.config["n_neighbors"] = max(1, int(self.config["n_neighbors"]))
            self.config["n_components"] = max(1, int(self.config["n_components"]))
            self.log("Updated run settings.")
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
            self._add_observation_locked(procedure, value, uncertainty, candidate_id)
            self.suggestions = []
            self.last_error = None
            self.last_model_status = "Observation added. Suggestions need an update."
            self.log(f"Added objective value {value:g}.")
            return self.to_json()

    def _add_observation_locked(
        self,
        procedure: str,
        value: float,
        uncertainty: Optional[float],
        candidate_id: Optional[str] = None,
    ) -> None:
        if candidate_id is None:
            match = next(
                (cand for cand in self.candidates if cand["procedure"] == procedure),
                None,
            )
            candidate_id = match["id"] if match else None
        self.observations.append(
            {
                "id": f"obs-{len(self.observations) + 1}",
                "candidate_id": candidate_id,
                "procedure": procedure,
                "value": float(value),
                "target": float(self.target_value(value)),
                "uncertainty": uncertainty,
                "time": _now(),
            }
        )

    def suggest(self) -> Dict[str, Any]:
        with self.lock:
            self.last_error = None
            available = self.available_candidates()
            if not available:
                self.suggestions = []
                self.last_model_status = "No unevaluated candidates remain."
                return self.to_json()
            if not os.environ.get("OPENAI_API_KEY"):
                self.suggestions = self._random_suggestions(available)
                self.last_model_status = "OpenAI API key missing. Showing random candidates."
                self.log("Add OPENAI_API_KEY to enable embeddings and GPR.")
                return self.to_json()
            if len(self.observations) < 2:
                self.suggestions = self._random_suggestions(available)
                self.last_model_status = "Add at least 2 observations before GPR suggestions."
                return self.to_json()

            try:
                self.suggestions = self._gpr_suggestions(available)
                self.last_model_status = (
                    f"Updated GPR on {len(self.observations)} observations."
                )
                self.log(f"Updated suggestions with {self.config['acquisition']}.")
            except Exception as exc:  # pragma: no cover - needs live API/GPR deps
                self.suggestions = self._random_suggestions(available)
                self.last_error = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                self.last_model_status = "GPR update failed. Showing random candidates."
                self.log("GPR update failed; see status panel.")
            return self.to_json()

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

    def _candidate_subset(self, available: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        score_limit = min(self.config["score_limit"], len(available))
        if score_limit == len(available):
            return list(available)
        return random.sample(available, score_limit)

    def _gpr_suggestions(self, available: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from boicl import AskTellGPR

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        n_obs = len(self.observations)
        n_neighbors = min(self.config["n_neighbors"], max(1, n_obs - 1))
        n_components = min(self.config["n_components"], max(1, n_obs - 1))
        model = AskTellGPR(
            cache_path=str(self.embedding_cache_path),
            n_neighbors=n_neighbors,
            n_components=n_components,
            y_name=self.config["objective_name"],
            y_formatter=lambda y: f"{float(y):0.6g}",
        )

        for obs in self.observations[:-1]:
            model.tell(obs["procedure"], obs["target"], train=False)
        model.tell(self.observations[-1]["procedure"], self.observations[-1]["target"], train=True)

        subset = self._candidate_subset(available)
        procedures = [cand["procedure"] for cand in subset]
        raw = model.ask(
            procedures,
            aq_fxn=self.config["acquisition"],
            k=min(self.config["batch_size"], len(procedures)),
            aug_random_filter=len(procedures),
            _lambda=self.config["ucb_lambda"],
        )
        if hasattr(model, "save_cache"):
            model.save_cache(str(self.embedding_cache_path))

        selected, acquisition, means = raw[:3]
        stds = raw[3] if len(raw) > 3 else [None] * len(selected)
        by_proc = {cand["procedure"]: cand for cand in subset}
        return [
            {
                "candidate_id": by_proc[procedure]["id"],
                "procedure": procedure,
                "acquisition": float(aq),
                "mean": self.display_value(float(mean)) if mean is not None else None,
                "std": float(std) if std is not None else None,
                "source": "gpr",
            }
            for procedure, aq, mean, std in zip(selected, acquisition, means, stds)
        ]

    def reset_run(self) -> Dict[str, Any]:
        with self.lock:
            self.observations = []
            self.suggestions = []
            self.last_error = None
            self.last_model_status = "Run reset. Dataset is still loaded."
            self.log("Cleared observations and suggestions.")
            return self.to_json()

    def export_observations_csv(self) -> str:
        with self.lock:
            out = StringIO()
            writer = csv.DictWriter(
                out,
                fieldnames=["procedure", "value", "uncertainty", "time"],
                lineterminator="\n",
            )
            writer.writeheader()
            for obs in self.observations:
                writer.writerow(
                    {
                        "procedure": obs["procedure"],
                        "value": obs["value"],
                        "uncertainty": "" if obs["uncertainty"] is None else obs["uncertainty"],
                        "time": obs["time"],
                    }
                )
            return out.getvalue()


class LocalAppHandler(BaseHTTPRequestHandler):
    state: LocalBOState

    def log_message(self, fmt: str, *args: Any) -> None:
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
        self._send_json({"error": str(exc), "state": self.state.to_json()}, status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(INDEX_HTML, "text/html")
        elif parsed.path == "/api/state":
            self._send_json(self.state.to_json())
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
                key = (payload.get("openai_api_key") or "").strip()
                openrouter_key = (payload.get("openrouter_api_key") or "").strip()
                if key:
                    _write_env_value(self.state.env_path, "OPENAI_API_KEY", key)
                    self.state.log("Saved OPENAI_API_KEY to local .env.")
                if openrouter_key:
                    _write_env_value(
                        self.state.env_path, "OPENROUTER_API_KEY", openrouter_key
                    )
                    self.state.log("Saved OPENROUTER_API_KEY to local .env.")
                self._send_json(self.state.to_json())
            elif parsed.path == "/api/import-dataset":
                filename = parse_qs(parsed.query).get("filename", ["dataset.csv"])[0]
                self._send_json(self.state.import_dataset(filename, self._read_raw()))
            elif parsed.path == "/api/config":
                self._send_json(self.state.update_config(self._read_json()))
            elif parsed.path == "/api/observe":
                self._send_json(self.state.add_observation(self._read_json()))
            elif parsed.path == "/api/suggest":
                self._send_json(self.state.suggest())
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
      <span class="chip" id="datasetStatus">No dataset</span>
      <span class="chip" id="observationStatus">0 observations</span>
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
        <button class="primary" id="saveKey">Save Locally</button>
      </section>

      <section class="panel">
        <h2>Dataset</h2>
        <div class="field">
          <label for="datasetFile">CSV or Excel file</label>
          <input id="datasetFile" type="file" accept=".csv,.txt,.xlsx,.xls">
        </div>
        <button id="importDataset">Import Dataset</button>
      </section>

      <section class="panel">
        <h2>Settings</h2>
        <div class="field">
          <label for="objectiveName">Objective name</label>
          <input id="objectiveName" value="objective">
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
        <div class="row">
          <div class="field">
            <label for="batchSize">Batch size</label>
            <input id="batchSize" type="number" min="1" max="25" value="1">
          </div>
          <div class="field">
            <label for="ucbLambda">UCB lambda</label>
            <input id="ucbLambda" type="number" step="0.1" value="0.5">
          </div>
        </div>
        <div class="row">
          <div class="field">
            <label for="scoreLimit">Score limit</label>
            <input id="scoreLimit" type="number" min="1" value="250">
          </div>
          <div class="field">
            <label for="nNeighbors">Neighbors</label>
            <input id="nNeighbors" type="number" min="1" value="5">
          </div>
        </div>
        <label class="switchline">
          <input id="autoSuggest" type="checkbox" checked>
          Auto update
        </label>
        <div class="toolbar" style="margin-top: 12px;">
          <button id="saveConfig">Apply</button>
          <button id="resetRun">Reset Run</button>
        </div>
      </section>
    </aside>

    <section class="stack">
      <section class="panel">
        <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
          <h2 style="margin:0;">Best So Far</h2>
          <a class="muted" href="/api/export-observations.csv">Export observations</a>
        </div>
        <div id="plot" class="plot"></div>
      </section>

      <section class="panel">
        <h2>Add Result</h2>
        <div class="field">
          <label for="candidateSelect">Candidate</label>
          <select id="candidateSelect"></select>
        </div>
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
        <button class="primary" id="addObservation">Add Observation</button>
      </section>

      <section class="panel">
        <div class="toolbar" style="justify-content: space-between; margin-bottom: 12px;">
          <h2 style="margin:0;">Suggestions</h2>
          <button id="suggest">Update Suggestions</button>
        </div>
        <div id="suggestions"></div>
      </section>

      <section class="panel">
        <h2>Observations</h2>
        <div id="observations"></div>
      </section>

      <section id="messages" class="stack"></section>
    </section>
  </main>

  <script>
    let state = null;
    let busy = false;

    const $ = (id) => document.getElementById(id);

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return '';
      return Number(value).toLocaleString(undefined, { maximumSignificantDigits: digits });
    }

    async function request(path, options = {}) {
      setBusy(true);
      try {
        const response = await fetch(path, options);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Request failed');
        state = payload.state || payload;
        render();
        return state;
      } catch (error) {
        renderError(error.message);
      } finally {
        setBusy(false);
      }
    }

    function setBusy(next) {
      busy = next;
      document.querySelectorAll('button').forEach((button) => button.disabled = busy);
    }

    async function refresh() {
      const response = await fetch('/api/state');
      state = await response.json();
      render();
    }

    function payloadConfig() {
      return {
        objective_name: $('objectiveName').value,
        objective_direction: $('objectiveDirection').value,
        acquisition: $('acquisition').value,
        batch_size: Number($('batchSize').value || 1),
        ucb_lambda: Number($('ucbLambda').value || 0.5),
        score_limit: Number($('scoreLimit').value || 250),
        n_neighbors: Number($('nNeighbors').value || 5),
        auto_suggest: $('autoSuggest').checked
      };
    }

    function render() {
      if (!state) return;
      const key = state.key_status.openai_configured;
      $('keyStatus').textContent = key ? `OpenAI key set ${state.key_status.openai_hint}` : 'OpenAI key missing';
      $('keyStatus').className = `chip ${key ? 'good' : 'warn'}`;
      $('datasetStatus').textContent = `${state.candidate_count} candidates`;
      $('datasetStatus').className = `chip ${state.candidate_count ? 'good' : 'warn'}`;
      $('observationStatus').textContent = `${state.observations.length} observations`;
      $('observationStatus').className = `chip ${state.observations.length ? 'good' : 'warn'}`;
      $('modelStatus').textContent = state.last_model_status || '';

      renderConfig();
      renderCandidateSelect();
      renderPlot();
      renderSuggestions();
      renderObservations();
      renderMessages();
    }

    function renderConfig() {
      const config = state.config;
      $('objectiveName').value = config.objective_name;
      $('objectiveDirection').value = config.objective_direction;
      $('batchSize').value = config.batch_size;
      $('ucbLambda').value = config.ucb_lambda;
      $('scoreLimit').value = config.score_limit;
      $('nNeighbors').value = config.n_neighbors;
      $('autoSuggest').checked = Boolean(config.auto_suggest);

      const current = $('acquisition').value || config.acquisition;
      $('acquisition').innerHTML = state.acquisition_functions
        .map((name) => `<option value="${name}">${name.replaceAll('_', ' ')}</option>`)
        .join('');
      $('acquisition').value = current;
    }

    function renderCandidateSelect() {
      const suggestions = state.suggestions || [];
      const observed = new Set(state.observations.map((obs) => obs.candidate_id).filter(Boolean));
      const candidates = (state.candidates || []).filter((cand) => !observed.has(cand.id)).slice(0, 500);
      const options = ['<option value="">Manual procedure</option>'];
      suggestions.forEach((sug) => {
        options.push(`<option value="${escapeHtml(sug.candidate_id)}">Suggested: ${escapeHtml(sug.procedure.slice(0, 90))}</option>`);
      });
      candidates.forEach((cand) => {
        if (!suggestions.some((sug) => sug.candidate_id === cand.id)) {
          options.push(`<option value="${escapeHtml(cand.id)}">${escapeHtml(cand.procedure.slice(0, 90))}</option>`);
        }
      });
      $('candidateSelect').innerHTML = options.join('');
    }

    function renderPlot() {
      const host = $('plot');
      const trace = state.best_trace || [];
      if (!trace.length) {
        host.innerHTML = '<div class="empty" style="margin: 18px;">No observations yet</div>';
        return;
      }
      const obs = state.observations;
      const width = Math.max(560, host.clientWidth || 760);
      const height = 280;
      const pad = { left: 52, right: 22, top: 22, bottom: 42 };
      const values = obs.flatMap((item) => {
        const unc = Number(item.uncertainty || 0);
        return [item.value - unc, item.value + unc, item.value];
      }).concat(trace.map((item) => item.best));
      let minY = Math.min(...values);
      let maxY = Math.max(...values);
      if (minY === maxY) { minY -= 1; maxY += 1; }
      const x = (i) => pad.left + ((i - 1) / Math.max(1, trace.length - 1)) * (width - pad.left - pad.right);
      const y = (value) => pad.top + (1 - ((value - minY) / (maxY - minY))) * (height - pad.top - pad.bottom);
      const bestPath = trace.map((item, idx) => `${idx ? 'L' : 'M'} ${x(item.index).toFixed(1)} ${y(item.best).toFixed(1)}`).join(' ');
      const points = obs.map((item, idx) => {
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
      host.innerHTML = `<svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img">
        ${ticks}
        <line x1="${pad.left}" x2="${width - pad.right}" y1="${height - pad.bottom}" y2="${height - pad.bottom}" stroke="#cfd6e2" />
        <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" stroke="#cfd6e2" />
        <path d="${bestPath}" fill="none" stroke="#0f766e" stroke-width="3" />
        ${points}
        <text x="${width / 2 - 28}" y="${height - 10}">experiment</text>
      </svg>`;
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
          openrouter_api_key: $('openrouterKey').value
        })
      });
      $('openaiKey').value = '';
      $('openrouterKey').value = '';
    });

    $('importDataset').addEventListener('click', async () => {
      const file = $('datasetFile').files[0];
      if (!file) return renderError('Choose a CSV or Excel file.');
      await request(`/api/import-dataset?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        body: await file.arrayBuffer()
      });
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
    });

    $('suggest').addEventListener('click', updateSuggestions);
    $('suggestTop').addEventListener('click', updateSuggestions);
    $('resetRun').addEventListener('click', () => request('/api/reset', { method: 'POST' }));

    refresh();
  </script>
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
