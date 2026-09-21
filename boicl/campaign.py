"""Transactional MoC campaign service shared by library, CLI and local browser."""
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import math
import re
import subprocess
import threading
import uuid

from .campaign_config import resolve_config, ENGINE_LABELS
from .moc_import import (
    load_moc_package,
    digest,
    validate_candidates,
    _validate_seeds,
    _validate_archive,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


def fingerprint(value):
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def toolkit_revision():
    """Read code provenance without reading environment, remotes or credentials."""
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        return {"commit": commit, "working_tree_modified": bool(changed.strip())}
    except (OSError, subprocess.SubprocessError):
        return {
            "commit": None,
            "working_tree_modified": None,
            "note": "Git revision unavailable in installed distribution",
        }


def _reject_secrets(value):
    sensitive = {
        "apikey",
        "authorization",
        "authorizationheader",
        "password",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "secretkey",
        "privatekey",
        "credentials",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub("[^a-z0-9]", "", str(key).lower())
            if normalized in sensitive or normalized.endswith("apikey"):
                raise ValueError(
                    f"Campaign bundles cannot contain credential field {key!r}"
                )
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)
    elif isinstance(value, str):
        if (
            "sk-" in value
            and re.search(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}", value)
        ) or ("-----BEGIN " + "PRIVATE KEY-----") in value:
            raise ValueError(
                "Campaign text contains a credential-shaped secret; remove it before saving or exporting"
            )


def atomic_json(path, value):
    atomic_bytes(
        path, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    )


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uid() + ".tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


class CampaignService:
    def __init__(self, root, runner=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.campaigns = {}
        self.jobs = {}
        self.runner = runner
        for path in self.root.glob("campaign-*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self._validate_bundle(data)
            self.campaigns[data["campaign_id"]] = data

    def _save(self, data, reason="Campaign saved"):
        if not isinstance(data.get("campaign_id"), str) or not re.fullmatch(
            "[a-f0-9]{32}", data["campaign_id"]
        ):
            raise ValueError("Invalid campaign ID")
        checkpoint = self._write_checkpoint(data, reason=reason)
        try:
            atomic_json(self.root / f"campaign-{data['campaign_id']}.json", data)
        except Exception:
            self._remove_checkpoint_files(
                data["campaign_id"], checkpoint["checkpoint_id"]
            )
            raise
        self.campaigns[data["campaign_id"]] = deepcopy(data)

    def _checkpoint_directory(self, cid):
        if not isinstance(cid, str) or not re.fullmatch("[a-f0-9]{32}", cid):
            raise ValueError("Invalid campaign ID")
        return self.root / "checkpoints" / cid

    def _remove_checkpoint_files(self, cid, checkpoint_id):
        directory = self._checkpoint_directory(cid)
        for suffix in (".json", ".json.gz"):
            (directory / (checkpoint_id + suffix)).unlink(missing_ok=True)

    def _write_checkpoint(self, data, name=None, reason="Manual checkpoint"):
        """Persist an immutable full snapshot and a small independently readable index."""
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ValueError("Checkpoint name must be nonempty text or None")
        _reject_secrets(data)
        _reject_secrets(name)
        payload = json.dumps(
            data, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        checkpoint = dict(
            checkpoint_version=1,
            checkpoint_id=uid(),
            campaign_id=data["campaign_id"],
            created_at=now(),
            name=name.strip() if name is not None else None,
            reason=reason,
            history_revision=data["history_revision"],
            config_name=data["config"]["name"],
            engine=ENGINE_LABELS[data["config"]["engine"]],
            pending_count=sum(s["status"] == "pending" for s in data["suggestions"]),
            observation_count=len(self.active(data)),
            new_measurements=self.completed(data),
            suggestion_count=len(data["suggestions"]),
            snapshot_sha256=hashlib.sha256(payload).hexdigest(),
        )
        directory = self._checkpoint_directory(data["campaign_id"])
        stem = directory / checkpoint["checkpoint_id"]
        try:
            atomic_bytes(
                stem.with_suffix(".json.gz"),
                gzip.compress(payload, compresslevel=3, mtime=0),
            )
            atomic_json(stem.with_suffix(".json"), checkpoint)
        except Exception:
            self._remove_checkpoint_files(
                data["campaign_id"], checkpoint["checkpoint_id"]
            )
            raise
        return checkpoint

    def list_checkpoints(self, cid):
        """List saved points without decompressing candidate pools or model traces."""
        with self.lock:
            if cid not in self.campaigns:
                raise ValueError("Unknown campaign")
            directory = self._checkpoint_directory(cid)
            result = []
            for path in directory.glob("*.json"):
                row = json.loads(path.read_text(encoding="utf-8"))
                if (
                    not isinstance(row, dict)
                    or row.get("checkpoint_version") != 1
                    or row.get("campaign_id") != cid
                    or row.get("checkpoint_id") != path.stem
                    or not re.fullmatch("[a-f0-9]{32}", path.stem)
                    or not path.with_suffix(".json.gz").is_file()
                ):
                    raise ValueError("Invalid checkpoint metadata")
                _reject_secrets(row)
                result.append(row)
            return sorted(result, key=lambda row: row["created_at"], reverse=True)

    def save_checkpoint(self, cid, name=None):
        with self.lock:
            return self._write_checkpoint(self.get(cid), name=name)

    def restore_checkpoint(self, cid, checkpoint_id, name=None):
        """Fork a saved state into a new campaign; never replace its live source."""
        if not isinstance(checkpoint_id, str) or not re.fullmatch(
            "[a-f0-9]{32}", checkpoint_id
        ):
            raise ValueError("Invalid checkpoint ID")
        with self.lock:
            metadata = next(
                (
                    r
                    for r in self.list_checkpoints(cid)
                    if r["checkpoint_id"] == checkpoint_id
                ),
                None,
            )
            if metadata is None:
                raise ValueError("Unknown checkpoint")
            path = self._checkpoint_directory(cid) / (checkpoint_id + ".json.gz")
            try:
                payload = gzip.decompress(path.read_bytes())
                if hashlib.sha256(payload).hexdigest() != metadata["snapshot_sha256"]:
                    raise ValueError("Checkpoint snapshot checksum mismatch")
                data = json.loads(payload)
            except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid checkpoint snapshot") from exc
            self._validate_bundle(data)
            if data["campaign_id"] != cid:
                raise ValueError("Checkpoint does not belong to this campaign")
            old_hash = fingerprint(data["config"])
            if name is not None:
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Campaign name must be nonempty text")
                data["config"]["name"] = name.strip()
                # A display rename does not invalidate an otherwise current proposal.
                new_hash = fingerprint(data["config"])
                for row in data["suggestions"]:
                    if row["status"] == "suggested" and row["config_hash"] == old_hash:
                        row["config_hash"] = new_hash
            data["campaign_id"] = uid()
            data["restored_from_checkpoint"] = dict(
                campaign_id=cid,
                checkpoint_id=checkpoint_id,
                checkpoint_created_at=metadata["created_at"],
                restored_at=now(),
            )
            self._validate_bundle(data)
            self._save(data, reason="Checkpoint resumed as independent campaign")
            return data["campaign_id"]

    def get(self, cid):
        with self.lock:
            if cid not in self.campaigns:
                raise ValueError("Unknown campaign")
            return deepcopy(self.campaigns[cid])

    def view_snapshot(self, cid, all_candidates=False):
        """Compact defensive display projection; full export/get remain unchanged.

        Comparisons need measured/pending recipe text, not a copy of every pool
        row or thousands of acquisition scores. This projection is not a portable
        bundle and must never be used to fit or persist a campaign.
        """
        with self.lock:
            if cid not in self.campaigns:
                raise ValueError("Unknown campaign")
            data = self.campaigns[cid]
            result = deepcopy(
                {
                    k: v
                    for k, v in data.items()
                    if k not in {"candidates", "suggestions"}
                }
            )
            result["suggestions"] = [
                deepcopy({k: v for k, v in row.items() if k != "engine_result"})
                for row in data["suggestions"]
            ]
            referenced = {r["candidate_id"] for r in data["observations"]}
            referenced.update(
                r["candidate_id"] for r in data["suggestions"] if r.get("candidate_id")
            )
            result["candidates"] = deepcopy(
                [
                    r
                    for r in data["candidates"]
                    if all_candidates or r["candidate_id"] in referenced
                ]
            )
            result["view_only"] = True
            return result

    @staticmethod
    def active(data):
        return [
            r
            for r in data["observations"]
            if r.get("training_included", True) is True
            and r.get("record_status") == "measured"
        ]

    @classmethod
    def eligible(cls, data):
        blocked = {r["candidate_id"] for r in cls.active(data)} | set(
            data.get("excluded_ids", [])
        )
        blocked.update(
            r["candidate_id"] for r in data["suggestions"] if r["status"] == "pending"
        )
        if data["config"]["repeat_policy"] == "exclude_all_historical":
            blocked.update(r["candidate_id"] for r in data["archive"])
        return [r for r in data["candidates"] if r["candidate_id"] not in blocked]

    @classmethod
    def completed(cls, data):
        return len(
            {
                r["physical_measurement_id"]
                for r in cls.active(data)
                if not r.get("is_seed")
            }
        )

    def create(
        self, preset="moc_llm", package=None, overrides=None, synthetic_demo=False
    ):
        config = resolve_config(preset, overrides)
        if config["data_schema"] == "generic" and package is None:
            raise ValueError("Generic campaigns require explicit mapped input records")
        package = deepcopy(package or load_moc_package())
        if config["initialization"] == "eight_observations":
            for row in package["archive"]:
                promoted = deepcopy(row)
                promoted.update(training_included=True, record_status="measured")
                package["observations"].append(promoted)
        data = dict(
            bundle_version=1,
            campaign_id=uid(),
            created_at=now(),
            config=config,
            history_revision=0,
            suggestions=[],
            excluded_ids=[],
            started=False,
            synthetic_demo=bool(synthetic_demo),
            rng_state={"seed": config["seed"], "suggestion_sequence": 0},
            events=[],
            **package,
        )
        data["initialization_fingerprint"] = fingerprint(data["observations"])
        data["initial_observations"] = deepcopy(data["observations"])
        data["toolkit_code_revision"] = toolkit_revision()
        self._validate_bundle(data)
        with self.lock:
            self._save(data, reason="Campaign created")
            self.campaigns[data["campaign_id"]] = data
        return data["campaign_id"]

    def create_generic(
        self,
        records,
        feature_spec,
        *,
        objective="value",
        direction="maximize",
        bounds=None,
        units="",
        preset="generic_gp",
        observations=None,
        name=None,
        overrides=None,
        procedure_column="procedure",
        id_column="candidate_id",
        sigma_column=None,
        synthetic_demo=False,
    ):
        from .generic_import import load_generic_package, validate_bounds

        if not preset.startswith("generic_"):
            raise ValueError("Use a generic engine preset for generic inputs")
        bounds = validate_bounds(bounds)
        package = load_generic_package(
            records,
            feature_spec,
            objective=objective,
            bounds=bounds,
            observations=observations,
            procedure_column=procedure_column,
            id_column=id_column,
            sigma_column=sigma_column,
        )
        changes = deepcopy(overrides or {})
        changes.update(
            objective=objective, direction=direction, bounds=bounds, units=units
        )
        if name is not None:
            changes["name"] = name
        changes.setdefault("structured_gp", {})["feature_spec"] = package["provenance"][
            "feature_spec"
        ]
        if preset == "generic_gp" and not feature_spec:
            raise ValueError("Structured GP needs explicitly mapped feature columns")
        # Padding is a fraction of the user-specified range, never a chemistry unit.
        if bounds is not None:
            changes["structured_gp"].setdefault(
                "logit_delta_pp", 0.0065 * (bounds[1] - bounds[0])
            )
        return self.create(preset, package, changes, synthetic_demo)

    def create_control(self, cid, reset=False):
        with self.lock:
            parent = self.get(cid)
            if not reset:
                existing = [
                    row
                    for row in self.campaigns.values()
                    if row["config"].get("comparison_parent_id") == cid
                    and row["config"].get("selection_policy") == "random_control"
                ]
                if existing:
                    return existing[-1]["campaign_id"]
            config = deepcopy(parent["config"])
            config.update(
                selection_policy="random_control",
                comparison_parent_id=cid,
                name=parent["config"]["name"] + " — independent random control",
            )
            package = {
                k: deepcopy(parent[k])
                for k in ["candidates", "archive", "pool_fingerprint", "provenance"]
            }
            package["observations"] = deepcopy(parent["initial_observations"])
            # Eight-source initialization is already complete, so avoid promotion twice.
            if config["initialization"] == "eight_observations":
                package["observations"] = [
                    r
                    for r in package["observations"]
                    if r.get("run_id") in {"M7", "M12", "M13"}
                ]
            return self.create(
                config["preset"], package, config, parent["synthetic_demo"]
            )

    @staticmethod
    def objective_field(data):
        return (
            "value" if data["config"].get("data_schema") == "generic" else "moc_wt_pct"
        )

    def create_pair(self, package=None, overrides=None, synthetic_demo=False):
        package = package or load_moc_package()
        gp = self.create("moc_gp", package, overrides, synthetic_demo)
        llm = self.create("moc_llm", package, overrides, synthetic_demo)
        manifest = self.comparison(gp, llm)
        atomic_json(self.root / f"matched-{gp}-{llm}.json", manifest)
        return {"gp": gp, "llm": llm, "manifest": manifest}

    def comparison(self, gp, llm):
        a, b = self.get(gp), self.get(llm)
        keys = [
            "objective",
            "bounds",
            "direction",
            "repeat_policy",
            "new_measurement_budget",
            "seed",
        ]
        if (
            a["pool_fingerprint"] != b["pool_fingerprint"]
            or a["initialization_fingerprint"] != b["initialization_fingerprint"]
            or any(a["config"][k] != b["config"][k] for k in keys)
        ):
            raise ValueError("Campaigns do not have matched initial conditions")
        return dict(
            schema="matched-campaign-v1",
            campaigns=[gp, llm],
            pool_fingerprint=a["pool_fingerprint"],
            initialization_fingerprint=a["initialization_fingerprint"],
            shared={k: a["config"][k] for k in keys},
            shared_later_outcomes=False,
            method_difference="GP maximin below 10 distinct measured designs, then bounded transformed EI; LLM empirical EI from two designs",
            m12_resolution=a["provenance"].get("m12_resolution"),
            synthetic_demo=a["synthetic_demo"] or b["synthetic_demo"],
        )

    def summary(self, cid):
        with self.lock:
            if cid not in self.campaigns:
                raise ValueError("Unknown campaign")
            data = self.campaigns[cid]
            active = self.active(data)
            suggestions = [
                deepcopy({k: v for k, v in row.items() if k != "engine_result"})
                for row in data["suggestions"]
            ]
            lookup = {r["candidate_id"]: r for r in data["candidates"]}
            for row in suggestions:
                row["candidate"] = deepcopy(lookup.get(row.get("candidate_id")))
            return dict(
                campaign_id=cid,
                config=deepcopy(data["config"]),
                engine_label=ENGINE_LABELS[data["config"]["engine"]],
                counts=dict(
                    candidates=len(data["candidates"]),
                    measured=len(active),
                    unique_measured=len({r["candidate_id"] for r in active}),
                    pending=sum(r["status"] == "pending" for r in suggestions),
                    available=len(self.eligible(data)),
                    new_measurements=self.completed(data),
                ),
                best=(max if data["config"]["direction"] == "maximize" else min)(
                    (r[self.objective_field(data)] for r in active), default=None
                ),
                observations=deepcopy(data["observations"]),
                archive=deepcopy(data["archive"]),
                suggestions=suggestions,
                provenance=deepcopy(data["provenance"]),
                started=data["started"],
                synthetic_demo=data["synthetic_demo"],
                history_revision=data["history_revision"],
                progress=deepcopy(
                    {
                        k: v
                        for k, v in self.jobs.get(cid, {}).items()
                        if k not in {"cancel", "thread"}
                    }
                ),
                events=deepcopy(data["events"][-20:]),
            )

    def list(self):
        with self.lock:
            return [
                {
                    "campaign_id": d["campaign_id"],
                    "name": d["config"]["name"],
                    "engine": ENGINE_LABELS[d["config"]["engine"]],
                    "data_schema": d["config"].get("data_schema", "moc"),
                    "selection_policy": d["config"].get("selection_policy", "engine"),
                    "comparison_parent_id": d["config"].get("comparison_parent_id"),
                    "created_at": d["created_at"],
                    "candidate_count": len(d["candidates"]),
                    "observation_count": len(self.active(d)),
                    "synthetic_demo": d["synthetic_demo"],
                }
                for d in self.campaigns.values()
            ]

    def update_config(self, cid, changes, apply=True):
        with self.lock:
            data = self.get(cid)
            merged = deepcopy(data["config"])
            for key, value in changes.items():
                if isinstance(merged.get(key), dict) and isinstance(value, dict):
                    merged[key].update(deepcopy(value))
                else:
                    merged[key] = deepcopy(value)
            config = resolve_config(data["config"]["preset"], merged)
            if config == data["config"]:
                return (
                    self.summary(cid)
                    if apply
                    else {"before": data["config"], "after": config}
                )
            if config["initialization"] != data["config"]["initialization"]:
                raise ValueError(
                    "Changing initialization requires a new campaign; history is preserved"
                )
            if (
                config.get("data_schema") == "generic"
                and config["objective"] != data["config"]["objective"]
            ):
                raise ValueError(
                    "Changing the objective mapping requires a new explicitly mapped campaign"
                )
            budget = config["new_measurement_budget"]
            if (
                budget is not None
                and self.completed(data)
                + sum(r["status"] == "pending" for r in data["suggestions"])
                > budget
            ):
                raise ValueError(
                    "Budget cannot be lower than completed and reserved measurements"
                )
            if not apply:
                return {"before": data["config"], "after": config}
            data["config"] = config
            self._validate_bundle(data)
            self._invalidate(
                data, "Settings changed; previous unreserved suggestions superseded"
            )
            self._save(data, reason="Settings changed")
        return self.summary(cid)

    def _invalidate(self, data, reason):
        data["history_revision"] += 1
        for suggestion in data["suggestions"]:
            if suggestion["status"] == "suggested":
                suggestion["status"] = "superseded"
        job = self.jobs.get(data["campaign_id"])
        if job and job.get("status") == "running":
            job["cancel"].set()
        data["events"].append({"at": now(), "message": reason})

    def start_suggestion(self, cid, background=True):
        with self.lock:
            data = self.get(cid)
            from .moc_demo import demo_runner

            if self.runner is demo_runner and not data["synthetic_demo"]:
                raise ValueError(
                    "The synthetic demo runner requires a campaign explicitly marked synthetic"
                )
            job = self.jobs.get(cid)
            if job and job["status"] == "running":
                return {"job_id": job["job_id"], "coalesced": True}
            pending = sum(r["status"] == "pending" for r in data["suggestions"])
            budget = data["config"]["new_measurement_budget"]
            if budget is not None and self.completed(data) + pending >= budget:
                raise ValueError(
                    "New-measurement budget is filled by completed and reserved measurements"
                )
            data["started"] = True
            sequence = data["rng_state"]["suggestion_sequence"]
            data["rng_state"]["suggestion_sequence"] += 1
            snapshot = deepcopy(data)
            snapshot["step_seed"] = data["config"]["seed"] + sequence
            job = {
                "job_id": uid(),
                "status": "running",
                "detail": "Validating inputs",
                "cancel": threading.Event(),
                "started_at": now(),
            }
            self._save(data, reason="Suggestion started")
            self.jobs[cid] = job
        if background:
            thread = threading.Thread(
                target=self._compute, args=(cid, snapshot, job), daemon=True
            )
            job["thread"] = thread
            thread.start()
        else:
            self._compute(cid, snapshot, job)
        return {"job_id": job["job_id"], "status": job["status"]}

    def _compute(self, cid, snapshot, job):
        def progress(*args, **kwargs):
            job["detail"] = " ".join(str(a) for a in args) if args else str(kwargs)

        try:
            eligible = self.eligible(snapshot)
            if not eligible:
                result = {
                    "status": "exhausted",
                    "candidate_id": None,
                    "selection_reason": "No eligible designs remain",
                }
            elif snapshot["config"].get("selection_policy") == "random_control":
                result = run_engine(
                    snapshot, eligible, self.root, job["cancel"], progress
                )
            elif self.runner:
                result = self.runner(snapshot, eligible, job["cancel"], progress)
            elif snapshot["synthetic_demo"]:
                from .moc_demo import demo_runner

                result = demo_runner(snapshot, eligible, job["cancel"], progress)
            else:
                result = run_engine(
                    snapshot, eligible, self.root, job["cancel"], progress
                )
            with self.lock:
                data = self.get(cid)
                stale = data["history_revision"] != snapshot[
                    "history_revision"
                ] or fingerprint(data["config"]) != fingerprint(snapshot["config"])
                selected = result.get("candidate_id") or result.get(
                    "selected_candidate_id"
                )
                if selected and selected not in {
                    r["candidate_id"] for r in data["candidates"]
                }:
                    result = {
                        **result,
                        "status": "failed",
                        "reason": "Engine returned a candidate outside the canonical pool",
                    }
                    selected = None
                status = (
                    "cancelled"
                    if job["cancel"].is_set()
                    else "superseded"
                    if stale
                    else "suggested"
                    if selected
                    else result.get("status", "failed")
                )
                if (
                    selected
                    and status == "suggested"
                    and selected not in {r["candidate_id"] for r in self.eligible(data)}
                ):
                    status = "superseded"
                record = dict(
                    suggestion_id=uid(),
                    job_id=job["job_id"],
                    candidate_id=selected,
                    status=status,
                    created_at=now(),
                    history_revision=snapshot["history_revision"],
                    config_hash=fingerprint(snapshot["config"]),
                    pool_fingerprint=snapshot["pool_fingerprint"],
                    seed=snapshot["step_seed"],
                    planned_quality_repeat=selected
                    in {r["candidate_id"] for r in data["archive"]},
                    selection_reason=result.get("selection_reason")
                    or result.get("reason")
                    or result.get("stage")
                    or "Highest empirical acquisition; stable retrieval-order ties",
                    prediction=result.get("prediction"),
                    score=result.get("score"),
                    acquisition_units=result.get("acquisition_units"),
                    inverse_target=result.get("target"),
                    sampler_diagnostics={
                        k: v
                        for k, v in result.get("diagnostics", {}).items()
                        if k
                        in {
                            "acceptance_rate",
                            "burn_acceptance_rate",
                            "retained_acceptance_rate",
                            "effective_sample_size",
                            "warnings",
                            "prediction_components",
                            "metadata_fallback_flags",
                            "outcome_transform",
                        }
                    },
                    engine_result=result,
                )
                if selected and result.get("predictions"):
                    pred = result["predictions"].get(selected, {})
                    record["prediction"] = pred
                    record["score"] = pred.get("score", pred.get("acquisition"))
                    acquisition = data["config"]["llm"]["acquisition"]
                    record["acquisition_units"] = {
                        "expected_improvement": data["config"]["units"]
                        or "objective units",
                        "probability_of_improvement": "probability",
                        "upper_confidence_bound": data["config"]["units"]
                        or "objective units",
                    }[acquisition]
                    if (
                        data["config"].get("data_schema") == "moc"
                        and acquisition == "expected_improvement"
                    ):
                        record["acquisition_units"] = "percentage points"
                    record["acquisition"] = acquisition
                data["suggestions"].append(record)
                self._save(data, reason=f"Suggestion {status}")
                job.update(
                    status=status, detail=record["selection_reason"], completed_at=now()
                )
        except InterruptedError:
            job.update(
                status="cancelled",
                detail="Cancelled; accepted measurements and reservations preserved",
            )
        except Exception as exc:
            job.update(status="failed", detail=str(exc))
            with self.lock:
                data = self.get(cid)
                data["events"].append(
                    {"at": now(), "message": f"Suggestion failed: {exc}"}
                )
                self._save(data, reason="Suggestion failed")

    def cancel_job(self, cid):
        with self.lock:
            job = self.jobs.get(cid)
            if job and job["status"] == "running" and not job["cancel"].is_set():
                data = self.get(cid)
                data["events"].append(
                    {"at": now(), "message": "Suggestion cancellation requested"}
                )
                self._save(data, reason="Suggestion cancellation requested")
                job["cancel"].set()
        return self.summary(cid)

    @staticmethod
    def _suggestion(data, suggestion_id):
        row = next(
            (r for r in data["suggestions"] if r["suggestion_id"] == suggestion_id),
            None,
        )
        if row is None:
            raise ValueError("Unknown suggestion")
        return row

    def reserve(self, cid, suggestion_id):
        with self.lock:
            data = self.get(cid)
            row = self._suggestion(data, suggestion_id)
            if row["status"] == "pending":
                return self.summary(cid)
            if (
                row["status"] != "suggested"
                or row["history_revision"] != data["history_revision"]
                or row["config_hash"] != fingerprint(data["config"])
            ):
                raise ValueError(
                    "This suggestion is stale or unavailable; request a new suggestion"
                )
            if row["candidate_id"] not in {
                r["candidate_id"] for r in self.eligible(data)
            }:
                raise ValueError("Candidate is already measured, reserved or excluded")
            budget = data["config"]["new_measurement_budget"]
            pending = sum(s["status"] == "pending" for s in data["suggestions"])
            if budget is not None and self.completed(data) + pending >= budget:
                raise ValueError("Measurement budget reached")
            row.update(status="pending", reserved_at=now())
            self._save(data, reason="Experiment reserved")
        return self.summary(cid)

    def cancel_reservation(self, cid, suggestion_id):
        with self.lock:
            data = self.get(cid)
            row = self._suggestion(data, suggestion_id)
            if row["status"] != "pending":
                raise ValueError("Only pending reservations can be released")
            row.update(status="cancelled", cancelled_at=now())
            self._save(data, reason="Reservation released")
        return self.summary(cid)

    def measure(self, cid, suggestion_id, values, request_id=None, refresh=True):
        with self.lock:
            data = self.get(cid)
            row = self._suggestion(data, suggestion_id)
            measurement = self._measurement(values, data["config"])
            prior = next(
                (
                    r
                    for r in data["observations"]
                    if request_id and r.get("request_id") == request_id
                ),
                None,
            )
            if prior is not None:
                if row.get("observation_id") != prior["observation_id"] or any(
                    prior.get(k) != v for k, v in measurement.items()
                ):
                    raise ValueError(
                        "Measurement request ID was already used for a different submission"
                    )
                return self.summary(cid)
            if row["status"] != "pending":
                raise ValueError(
                    "Reserve the proposed experiment before recording its measurement"
                )
            measurement.update(
                observation_id=uid(),
                physical_measurement_id=uid(),
                candidate_id=row["candidate_id"],
                training_included=True,
                record_status="measured",
                is_seed=False,
                refinement_version="operator-v1",
                measured_at=values.get("measured_at") or now(),
                recorded_at=now(),
                request_id=request_id,
                objective_authority="synthetic demonstration"
                if data["synthetic_demo"]
                else "operator measurement",
                synthetic=data["synthetic_demo"],
                planned_quality_repeat=row["planned_quality_repeat"],
            )
            data["observations"].append(measurement)
            row["status"] = "measured"
            row["observation_id"] = measurement["observation_id"]
            self._invalidate(data, "Confirmed measurement saved; model must refit")
            self._save(data, reason="Measurement recorded")
            auto = refresh and data["started"] and data["config"]["auto_suggest"]
            budget = data["config"]["new_measurement_budget"]
            auto = auto and (
                budget is None
                or self.completed(data)
                + sum(r["status"] == "pending" for r in data["suggestions"])
                < budget
            )
        if auto:
            self.start_suggestion(cid)
        return self.summary(cid)

    @staticmethod
    def _measurement(values, config=None):
        if config and config.get("data_schema") == "generic":
            from .generic_import import generic_measurement

            return generic_measurement(
                values, bounds=config["bounds"], objective=config["objective"]
            )
        if config and config.get("selection_policy") == "random_control":
            from .generic_import import generic_measurement

            adapted = {**values, "value": values.get("moc_wt_pct", values.get("value"))}
            if values.get("moc_wt_pct_sigma") is not None:
                adapted["objective_sigma"] = values["moc_wt_pct_sigma"]
            if values.get("closure_gap_wt_pct") is not None:
                adapted["closure_gap"] = values["closure_gap_wt_pct"]
            normalized = generic_measurement(adapted, bounds=[0, 100])
            return {
                "moc_wt_pct": normalized["value"],
                "moc_wt_pct_sigma": normalized.get("objective_sigma"),
                "gof": normalized.get("gof"),
                "closure_gap_wt_pct": normalized.get("closure_gap"),
                "closure_gap_origin": normalized.get(
                    "closure_gap_origin",
                    "not reported; random control does not fit a model",
                ),
                "source_note": normalized["source_note"],
                "phase_accounting_complete": False,
                "phase_accounting_residual_wt_pct": None,
            }
        result = {}
        for key in ["moc_wt_pct", "moc_wt_pct_sigma", "gof", "closure_gap_wt_pct"]:
            value = values.get(key)
            if value is None or value == "":
                raise ValueError(f"{key} is required; unknown is not zero")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if key != "closure_gap_wt_pct" and value < 0:
                raise ValueError(f"{key} must be nonnegative")
            result[key] = value
        if result["moc_wt_pct"] > 100:
            raise ValueError("MoC wt% must be in 0–100")
        if not values.get("closure_gap_origin"):
            raise ValueError("Record the gap source or explicit override reason")
        result["closure_gap_origin"] = str(values["closure_gap_origin"])
        result["source_note"] = str(values.get("source_note", ""))
        for key in [
            "mo_wt_pct",
            "mo2c_wt_pct",
            "moo2_wt_pct",
            "mo_wt_pct_sigma",
            "mo2c_wt_pct_sigma",
            "moo2_wt_pct_sigma",
        ]:
            val = values.get(key)
            if val not in (None, ""):
                val = float(val)
                if (
                    not math.isfinite(val)
                    or val < 0
                    or ("sigma" not in key and val > 100)
                ):
                    raise ValueError(f"Invalid {key}")
                result[key] = val
        phases = [
            result.get(k)
            for k in ["moc_wt_pct", "mo_wt_pct", "mo2c_wt_pct", "moo2_wt_pct"]
        ]
        complete = values.get("phase_accounting_complete", False)
        if not isinstance(complete, bool):
            raise ValueError("Phase-accounting completeness must be boolean")
        if complete and any(v is None for v in phases):
            raise ValueError(
                "Complete phase accounting requires all four phase fractions"
            )
        result["phase_accounting_complete"] = complete
        result["phase_accounting_residual_wt_pct"] = (
            100 - sum(phases) if all(v is not None for v in phases) else None
        )
        return result

    def refine(self, cid, observation_id, values, reason):
        if not reason:
            raise ValueError("A refinement revision requires a reason")
        with self.lock:
            data = self.get(cid)
            old = next(
                (
                    r
                    for r in data["observations"]
                    if r["observation_id"] == observation_id
                ),
                None,
            )
            if old is None:
                raise ValueError("Unknown observation")
            if old["record_status"] != "measured":
                raise ValueError("Only the active refinement can be revised")
            updated = {
                **old,
                **self._measurement(values, data["config"]),
                "observation_id": uid(),
                "supersedes": observation_id,
                "refinement_version": uid(),
                "revision_reason": reason,
                "recorded_at": now(),
            }
            old.update(training_included=False, record_status="superseded_refinement")
            data["observations"].append(updated)
            self._invalidate(data, "Refinement revision saved; predictions invalidated")
            self._save(data, reason="Refinement revised")
        return self.summary(cid)

    @staticmethod
    def _validate_bundle(data):
        if not isinstance(data, dict) or data.get("bundle_version") != 1:
            raise ValueError("Unsupported campaign bundle version")
        _reject_secrets(data)
        known = {
            "bundle_version",
            "campaign_id",
            "created_at",
            "config",
            "history_revision",
            "suggestions",
            "excluded_ids",
            "started",
            "synthetic_demo",
            "rng_state",
            "events",
            "candidates",
            "observations",
            "archive",
            "pool_fingerprint",
            "provenance",
            "initialization_fingerprint",
            "initial_observations",
            "toolkit_code_revision",
            "imported_from_campaign_id",
            "restored_from_checkpoint",
        }
        if set(data) - known:
            raise ValueError(
                f"Unknown campaign bundle fields: {sorted(set(data)-known)}"
            )
        required = known - {
            "initial_observations",
            "toolkit_code_revision",
            "imported_from_campaign_id",
            "restored_from_checkpoint",
        }
        if required - set(data):
            raise ValueError(
                f"Missing campaign bundle fields: {sorted(required-set(data))}"
            )
        if not isinstance(data["campaign_id"], str) or not re.fullmatch(
            "[a-f0-9]{32}", data["campaign_id"]
        ):
            raise ValueError(
                "Invalid campaign ID; expected a local 32-character hexadecimal ID"
            )
        restored = data.get("restored_from_checkpoint")
        if restored is not None and (
            not isinstance(restored, dict)
            or set(restored)
            != {"campaign_id", "checkpoint_id", "checkpoint_created_at", "restored_at"}
            or any(
                not isinstance(restored.get(key), str)
                or not re.fullmatch("[a-f0-9]{32}", restored[key])
                for key in ("campaign_id", "checkpoint_id")
            )
            or any(
                not isinstance(restored.get(key), str)
                for key in ("checkpoint_created_at", "restored_at")
            )
        ):
            raise ValueError("Invalid checkpoint restore provenance")
        try:
            fingerprint(data)
        except (TypeError, ValueError) as exc:
            raise ValueError("Campaign bundle must contain finite JSON data") from exc
        if not isinstance(data["config"], dict) or "preset" not in data["config"]:
            raise ValueError("Campaign configuration is missing")
        data["config"] = resolve_config(data["config"]["preset"], data["config"])
        generic = data["config"].get("data_schema") == "generic"
        objective = "value" if generic else "moc_wt_pct"
        for field in (
            "candidates",
            "observations",
            "archive",
            "suggestions",
            "excluded_ids",
            "events",
        ):
            if not isinstance(data[field], list):
                raise ValueError(f"Campaign {field} must be an array")
        for field in ("started", "synthetic_demo"):
            if not isinstance(data[field], bool):
                raise ValueError(f"Campaign {field} must be boolean")
        revision = data["history_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("Invalid history revision")
        state = data["rng_state"]
        if (
            not isinstance(state, dict)
            or set(state) != {"seed", "suggestion_sequence"}
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in state.values()
            )
        ):
            raise ValueError("Invalid campaign random-state schedule")
        if generic:
            from .generic_import import (
                validate_generic_candidates,
                pool_fingerprint,
                resolve_features,
            )

            specification = data["config"]["structured_gp"]["feature_spec"]
            resolved = resolve_features(
                specification, data["candidates"], objective=data["config"]["objective"]
            )
            if resolved != specification:
                raise ValueError(
                    "Generic feature mappings must contain resolved full-space bounds/categories"
                )
            validate_generic_candidates(data["candidates"], specification)
            if data["config"]["engine"] == "gpr_features" and not specification:
                raise ValueError("Structured GP requires mapped features")
        else:
            validate_candidates(data["candidates"])
        ids = {r["candidate_id"] for r in data["candidates"]}
        if len(set(data["excluded_ids"])) != len(data["excluded_ids"]) or any(
            cid not in ids for cid in data["excluded_ids"]
        ):
            raise ValueError("Invalid excluded candidate IDs")
        observation_ids = set()
        active_physical = set()
        for row in data["observations"] + data["archive"]:
            if not isinstance(row, dict) or row.get("candidate_id") not in ids:
                raise ValueError("Unknown observation candidate")
            value = row.get(objective)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Missing or invalid measured objective")
            bounds = data["config"]["bounds"]
            if not math.isfinite(value) or (
                bounds is not None and not bounds[0] <= value <= bounds[1]
            ):
                raise ValueError("Invalid measured objective")
            if (
                not isinstance(row.get("observation_id"), str)
                or not row["observation_id"]
            ):
                raise ValueError("Missing observation ID")
            if (
                not isinstance(row.get("physical_measurement_id"), str)
                or not row["physical_measurement_id"]
            ):
                raise ValueError("Missing physical measurement ID")
            if not isinstance(row.get("training_included"), bool) or not isinstance(
                row.get("is_seed"), bool
            ):
                raise ValueError("Measurement inclusion and seed flags must be boolean")
            if row.get("record_status") not in {
                "measured",
                "superseded_refinement",
                "historically_measured_excluded",
            }:
                raise ValueError("Invalid observation record status")
            for field in (
                ("objective_sigma", "gof", "closure_gap")
                if generic
                else ("moc_wt_pct_sigma", "gof", "closure_gap_wt_pct")
            ):
                value = row.get(field)
                if value is not None:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or (
                            field not in {"closure_gap_wt_pct", "closure_gap"}
                            and value < 0
                        )
                    ):
                        raise ValueError(f"Invalid measurement metadata: {field}")
            if not row.get("is_seed") and row.get("record_status") == "measured":
                CampaignService._measurement(row, data["config"])
        for row in data["observations"]:
            oid = row["observation_id"]
            if oid in observation_ids:
                raise ValueError("Duplicate observation ID")
            observation_ids.add(oid)
            if row["training_included"] and row["record_status"] == "measured":
                physical = row["physical_measurement_id"]
                if physical in active_physical:
                    raise ValueError(
                        "Multiple active refinements of one physical measurement"
                    )
                active_physical.add(physical)
        for row in data["observations"]:
            if row.get("supersedes") and row["supersedes"] not in observation_ids:
                raise ValueError("Refinement references an unknown prior record")
        suggestion_ids = set()
        pending = set()
        for row in data["suggestions"]:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("suggestion_id"), str)
                or row["suggestion_id"] in suggestion_ids
            ):
                raise ValueError("Invalid or duplicate suggestion ID")
            suggestion_ids.add(row["suggestion_id"])
            if row.get("candidate_id") is not None and row["candidate_id"] not in ids:
                raise ValueError("Unknown suggested candidate")
            if row.get("status") not in {
                "suggested",
                "pending",
                "measured",
                "superseded",
                "cancelled",
                "failed",
                "exhausted",
                "error",
            }:
                raise ValueError("Invalid suggestion status")
            if (
                row["status"] in {"suggested", "pending", "measured"}
                and row.get("candidate_id") is None
            ):
                raise ValueError("Suggestion requires a candidate ID")
            if not isinstance(row.get("engine_result"), dict):
                raise ValueError("Missing recorded engine result")
            if row.get("pool_fingerprint") != data["pool_fingerprint"]:
                raise ValueError("Suggestion pool fingerprint mismatch")
            if row["status"] == "pending":
                if row["candidate_id"] in pending:
                    raise ValueError("Duplicate pending reservation")
                pending.add(row["candidate_id"])
            if (
                row["status"] == "measured"
                and row.get("observation_id") not in observation_ids
            ):
                raise ValueError("Measured suggestion lacks an observation")
        active_ids = {row["candidate_id"] for row in CampaignService.active(data)}
        if pending & (active_ids | set(data["excluded_ids"])):
            raise ValueError("A reserved candidate is already measured or excluded")
        budget = data["config"]["new_measurement_budget"]
        if (
            budget is not None
            and CampaignService.completed(data) + len(pending) > budget
        ):
            raise ValueError(
                "Completed and reserved measurements exceed the campaign budget"
            )
        expected = (
            pool_fingerprint(data["candidates"])
            if generic
            else digest(
                json.dumps(
                    [
                        (r["candidate_id"], r["procedure_sha256"])
                        for r in data["candidates"]
                    ],
                    separators=(",", ":"),
                )
            )
        )
        if data["pool_fingerprint"] != expected:
            raise ValueError("Campaign pool fingerprint mismatch")
        if "initial_observations" not in data:
            initial = deepcopy(
                [
                    r
                    for r in data["observations"]
                    if r.get("is_seed") and not r.get("supersedes")
                ]
            )
            for row in initial:
                row.update(training_included=True, record_status="measured")
            if fingerprint(initial) != data["initialization_fingerprint"]:
                raise ValueError("Cannot recover immutable initialization snapshot")
            data["initial_observations"] = initial
        if (
            not isinstance(data["initial_observations"], list)
            or fingerprint(data["initial_observations"])
            != data["initialization_fingerprint"]
        ):
            raise ValueError("Initialization fingerprint mismatch")
        if generic:
            if data["provenance"].get("schema") != "generic-campaign-v1":
                raise ValueError("Generic input provenance is required")
            if data["archive"]:
                raise ValueError(
                    "Generic historical archives require an explicit supported adapter"
                )
            for row in data["initial_observations"]:
                if row.get("candidate_id") not in ids or not row.get("is_seed"):
                    raise ValueError("Invalid generic initialization record")
                CampaignService._measurement(row, data["config"])
        else:
            source_seeds = {
                r.get("run_id"): r.get("moc_wt_pct")
                for r in data["initial_observations"]
                if r.get("run_id") in {"M7", "M12", "M13"}
            }
            if source_seeds != {"M7": 72.1, "M12": 83.8, "M13": 23.4}:
                raise ValueError(
                    "Initial snapshot must preserve the confirmed M12 correction"
                )
            expected_count = (
                8 if data["config"]["initialization"] == "eight_observations" else 3
            )
            if len(data["initial_observations"]) != expected_count:
                raise ValueError("Initialization count does not match selected preset")
            _validate_seeds(
                deepcopy(
                    [
                        r
                        for r in data["initial_observations"]
                        if r.get("run_id") in {"M7", "M12", "M13"}
                    ]
                ),
                data["candidates"],
            )
            _validate_archive(data["archive"], data["candidates"])
            if (
                not isinstance(data["provenance"], dict)
                or data["provenance"].get("m12_resolution", {}).get("authoritative")
                != 83.8
            ):
                raise ValueError("Confirmed M12 provenance is required")
        data.setdefault(
            "toolkit_code_revision",
            {
                "commit": None,
                "working_tree_modified": None,
                "note": "Imported legacy bundle did not record code revision",
            },
        )

    def export(self, cid):
        with self.lock:
            result = self.get(cid)
            _reject_secrets(result)
            return result

    def import_bundle(self, data):
        data = deepcopy(data)
        self._validate_bundle(data)
        with self.lock:
            cid = data["campaign_id"]
            if cid in self.campaigns and fingerprint(data) != fingerprint(
                self.campaigns[cid]
            ):
                data["imported_from_campaign_id"] = cid
                data["campaign_id"] = uid()
            self._save(data, reason="Campaign imported")
            self.campaigns[data["campaign_id"]] = data
        return data["campaign_id"]

    def replay(self, cid, suggestion_id):
        data = self.get(cid)
        step = next(
            r for r in data["suggestions"] if r["suggestion_id"] == suggestion_id
        )
        result = step["engine_result"]
        if result.get("predictions"):
            from .llm_engine import replay_step

            return replay_step(result)
        return {
            "candidate_id": step["candidate_id"],
            "score": step["score"],
            "replayed_from_recorded_result": True,
            "engine_result": deepcopy(result),
        }


def run_engine(snapshot, eligible, root, cancel, progress):
    """Dispatch without ever silently switching statistical engines."""
    config = snapshot["config"]
    active = CampaignService.active(snapshot)
    if config.get("selection_policy") == "random_control":
        import numpy as np

        if cancel.is_set():
            raise InterruptedError("Random control selection cancelled")
        selected = eligible[
            int(np.random.default_rng(snapshot["step_seed"]).integers(len(eligible)))
        ]
        return {
            "status": "recommended",
            "candidate_id": selected["candidate_id"],
            "stage": "random_control",
            "selection_reason": "Independent uniform random control; initialization is shared and later outcomes remain separate",
            "prediction": None,
            "score": None,
            "acquisition_units": None,
            "random_seed": snapshot["step_seed"],
        }
    if snapshot.get("synthetic_demo") and config["engine"] != "gpr_features":
        from .moc_demo import demo_runner

        return demo_runner(snapshot, eligible, cancel, progress)
    if config["engine"] == "gpr_features":
        from .structured_gp import StructuredGP

        settings = {
            **config["structured_gp"],
            "seed": snapshot["step_seed"],
            "direction": config["direction"],
            "objective_field": CampaignService.objective_field(snapshot),
            "objective_bounds": config["bounds"],
            "units": config["units"],
        }
        gp = StructuredGP(settings)
        progress("Fitting synthesis-parameter GP / sampling hyperparameters")
        gp.fit(snapshot["candidates"], active, cancel=cancel, progress=progress)
        return gp.recommend(eligible, cancel=cancel, progress=progress)
    return run_text_engine(snapshot, eligible, root, cancel, progress)


def embedding_spec(config):
    """Resolve an explicit model and exact input template, never relabel vectors."""
    from .embedding_cache import EmbeddingSpec

    models = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
    }
    engine = config["engine"]
    if engine not in {"llm", "gpr_embeddings"}:
        raise ValueError("This engine does not use an embedding cache")
    model = (
        config["llm"]["embedding_model"]
        if engine == "llm"
        else config["embedding_gp"]["embedding_model"]
    )
    if model not in models:
        raise ValueError(
            f"Unsupported embedding model {model!r}; vector dimensions must be explicit"
        )
    if engine == "llm":
        return EmbeddingSpec(
            model,
            models[model],
            "experimental procedure: {procedure}",
            "crystal-prefixed-v1",
        )
    return EmbeddingSpec(model, models[model])


def embedding_cache_directory(root, config):
    return (
        Path(root)
        / "embeddings"
        / config["engine"]
        / embedding_spec(config).fingerprint
    )


def run_text_engine(snapshot, eligible, root, cancel, progress):
    # Imported only for text engines. Structured GP creates no provider client.
    from .embedding_cache import create_embedding_cache
    from .request_policy import RequestPolicy, ReliableClient
    from openai import OpenAI

    config = snapshot["config"]
    request_log = []
    policy = RequestPolicy(config["api"], cancel, request_log)
    client = None

    def live_client():
        nonlocal client
        if client is None:
            client = ReliableClient(OpenAI(max_retries=0), policy)
        return client

    spec = embedding_spec(config)
    cache = create_embedding_cache(embedding_cache_directory(root, config), spec)

    def embedder(texts):
        return live_client().embeddings.create(model=spec.model, input=texts)

    progress("Validating embedding cache; generating actual misses only")
    cache.prepare(
        snapshot["candidates"], embedder, cancelled=cancel.is_set, progress=progress
    )
    vectors = cache.matrix(snapshot["candidates"])
    active = CampaignService.active(snapshot)
    if config["engine"] == "gpr_embeddings":
        from .embedding_gp import EmbeddingGPEngine

        progress("Fitting fixed-corpus Isomap and embedding GP")
        gp = EmbeddingGPEngine(
            snapshot["candidates"],
            vectors,
            {**config["embedding_gp"], "maximize": config["direction"] == "maximize"},
            projection_cache=Path(root) / f"projection-{spec.fingerprint}.npz",
        )
        gp.fit(active)
        result = gp.suggest(eligible)
    else:
        from .llm_engine import LLMEngine

        lookup = {r["candidate_id"]: r for r in snapshot["candidates"]}
        observations = []
        for row in active:
            phases = {
                k: v
                for k, v in row.items()
                if (k.endswith("_wt_pct") or k.endswith("_sigma")) and k != "moc_wt_pct"
            }
            observations.append(
                {
                    **row,
                    "value": row[CampaignService.objective_field(snapshot)],
                    "procedure": lookup[row["candidate_id"]]["procedure"],
                    "phase_context": json.dumps(phases, ensure_ascii=False),
                }
            )
        engine = LLMEngine(
            {
                **config["llm"],
                "seed": snapshot["step_seed"],
                "objective_name": config["objective"],
                "objective_units": config["units"],
                "prompt_style": "generic"
                if config.get("data_schema") == "generic"
                else "moc",
            },
            client=live_client(),
        )
        excluded = {r["candidate_id"] for r in snapshot["candidates"]} - {
            r["candidate_id"] for r in eligible
        }
        result = engine.suggest(
            snapshot["candidates"],
            observations,
            excluded_ids=excluded,
            candidate_vectors=vectors,
            query_embedder=lambda procedure: cache.query(
                procedure, embedder, cancelled=cancel.is_set
            ),
            cancel=cancel.is_set,
        )
    result["provider_attempts"] = request_log
    return result
