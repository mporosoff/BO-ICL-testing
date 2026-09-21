"""Read-only main-toolkit projection over the shared campaign service.

CampaignService is the only owner of observations, settings and reservations.
This adapter never loads those records into the mutable legacy LocalBOState.
"""
from copy import deepcopy
import csv
from io import BytesIO, StringIO
import json
from urllib.parse import parse_qs


def catalog(service):
    return [
        dict(
            id="shared:" + row["campaign_id"],
            name=row["name"],
            candidate_count=row["candidate_count"],
            observation_count=row["observation_count"],
            benchmark_count=0,
            dataset_filename="shared campaign",
        )
        for row in service.list()
    ]


def legacy_config(config):
    from .local_app import DEFAULT_CONFIG

    result = deepcopy(DEFAULT_CONFIG)
    llm = config["llm"]
    gp = config["embedding_gp"]
    api = config["api"]
    bounds = config["bounds"] or [None, None]
    result.update(
        workflow_mode="live",
        optimizer=config["engine"],
        objective_name=config["objective"],
        objective_direction=config["direction"],
        objective_lower_bound=bounds[0],
        objective_upper_bound=bounds[1],
        acquisition=llm["acquisition"],
        embedding_model=llm["embedding_model"]
        if config["engine"] == "llm"
        else gp["embedding_model"],
        prediction_model=llm["forward_model"],
        inverse_model=llm["inverse_model"],
        prediction_system_message=llm.get("forward_system_message") or "",
        inverse_system_message=llm.get("inverse_system_message") or "",
        llm_samples=llm["n_samples"],
        llm_uncertainty_calibration=llm["uncertainty_scalar"],
        llm_prediction_temperature=llm["forward_temperature"],
        llm_inverse_temperature=llm["inverse_temperature"],
        selector_k=llm["selector_k"] or 5,
        inverse_filter=llm["shortlist_size"],
        inverse_random_candidates=llm["random_addons"],
        inverse_target_multiplier=llm["inverse_multiplier"],
        inverse_target_jitter=llm["inverse_jitter"],
        inverse_target_reference_scale=llm["reference_scale"],
        inverse_target_floor_value=llm.get("target_floor"),
        inverse_target_value=llm.get("manual_inverse_target"),
        inverse_design_count=llm.get("inverse_proposal_count", 1),
        batch_size=config["batch_size"],
        iterations_per_trial=config["new_measurement_budget"],
        ucb_lambda=llm["ucb_lambda"],
        n_neighbors=gp["neighbors"],
        n_components=gp["dimensions"],
        api_pause_seconds=api["request_spacing_s"],
        api_retry_attempts=api["maximum_attempts"],
        api_rate_limit_cooldown_seconds=api["base_cooldown_s"],
        auto_suggest=config["auto_suggest"],
        benchmark_seed=config["seed"],
    )
    return result


def config_changes(payload, config):
    """Map visible main controls to the one validated campaign schema."""
    c = dict(payload)
    engine = c.get("optimizer", config["engine"])
    if engine == "gpr":
        engine = "gpr_embeddings"
    llm = {}
    mapping = {
        "prediction_model": "forward_model",
        "inverse_model": "inverse_model",
        "llm_samples": "n_samples",
        "llm_uncertainty_calibration": "uncertainty_scalar",
        "llm_prediction_temperature": "forward_temperature",
        "llm_inverse_temperature": "inverse_temperature",
        "inverse_filter": "shortlist_size",
        "inverse_random_candidates": "random_addons",
        "inverse_target_multiplier": "inverse_multiplier",
        "inverse_target_jitter": "inverse_jitter",
        "inverse_target_reference_scale": "reference_scale",
        "ucb_lambda": "ucb_lambda",
        "acquisition": "acquisition",
    }
    for old, new in mapping.items():
        if old in c:
            llm[new] = c[old]
    if engine != "llm":
        llm.pop("acquisition", None)
    for old, new in [
        ("prediction_system_message", "forward_system_message"),
        ("inverse_system_message", "inverse_system_message"),
    ]:
        if old in c:
            llm[new] = (
                None if c[old] == "" and config["llm"].get(new) is None else c[old]
            )
    if "inverse_target_floor_value" in c:
        llm["target_floor"] = (
            None
            if c["inverse_target_floor_value"] in (None, "")
            else float(c["inverse_target_floor_value"])
        )
    if "inverse_target_value" in c:
        llm["manual_inverse_target"] = (
            None
            if c["inverse_target_value"] in (None, "")
            else float(c["inverse_target_value"])
        )
    if "inverse_design_count" in c:
        llm["inverse_proposal_count"] = c["inverse_design_count"]
    if "selector_mode" in c:
        llm["selector_mode"] = c["selector_mode"]
        llm["selector_k"] = c.get("selector_k", 5)
    changes = {"engine": engine, "llm": llm}
    if "auto_suggest" in c:
        changes["auto_suggest"] = c["auto_suggest"]
    if "iterations_per_trial" in c:
        changes["new_measurement_budget"] = c["iterations_per_trial"]
    if "name" in c:
        changes["name"] = c["name"]
    if "benchmark_seed" in c:
        changes["seed"] = c["benchmark_seed"]
    api = {
        new: c[old]
        for old, new in [
            ("api_pause_seconds", "request_spacing_s"),
            ("api_retry_attempts", "maximum_attempts"),
            ("api_rate_limit_cooldown_seconds", "base_cooldown_s"),
        ]
        if old in c
    }
    if api:
        changes["api"] = api
    if engine == "llm" and "embedding_model" in c:
        llm["embedding_model"] = c["embedding_model"]
    if engine == "gpr_embeddings":
        changes["embedding_gp"] = {
            new: c[old]
            for old, new in [
                ("embedding_model", "embedding_model"),
                ("n_neighbors", "neighbors"),
                ("n_components", "dimensions"),
            ]
            if old in c
        }
    if "structured_gp" in c:
        changes["structured_gp"] = c["structured_gp"]
    if config.get("data_schema") == "generic":
        if "objective_direction" in c:
            changes["direction"] = c["objective_direction"]
        if "objective_lower_bound" in c or "objective_upper_bound" in c:
            bounds = config["bounds"] or [None, None]
            changes["bounds"] = [
                None
                if c.get(key, bounds[index]) in (None, "")
                else float(c.get(key, bounds[index]))
                for index, key in enumerate(
                    ("objective_lower_bound", "objective_upper_bound")
                )
            ]
        if changes.get("bounds") == [None, None]:
            changes["bounds"] = None
    return changes


def project(service, cid, state):
    from .campaign_plot import plot_payload
    from .campaign_controls import (
        comparison_campaigns,
        random_control_campaign,
        random_control_state,
    )

    with service.lock:
        data = service.view_snapshot(cid, all_candidates=True)
        summary = service.summary(cid)
    config = data["config"]
    payload = state.to_json()
    lookup = {r["candidate_id"]: r for r in data["candidates"]}

    def candidate(row, index=0):
        return dict(
            id=row["candidate_id"],
            candidate_id=row["candidate_id"],
            row=index + 1,
            procedure=row.get("procedure", ""),
            objectives={},
            uncertainties={},
        )

    observations = []
    measured_suggestions = {
        row.get("observation_id"): row
        for row in data["suggestions"]
        if row.get("observation_id")
    }
    for row in data["observations"]:
        if row.get("record_status") != "measured":
            continue
        record = {
            **row,
            "id": row["observation_id"],
            "value": row.get("value", row.get("moc_wt_pct")),
            "uncertainty": row.get("objective_sigma", row.get("moc_wt_pct_sigma")),
            "procedure": lookup[row["candidate_id"]].get("procedure", ""),
            "time": row.get("measured_at")
            or row.get("source_date")
            or "source order; date unknown",
        }
        selected = measured_suggestions.get(row["observation_id"])
        if selected:
            prediction = selected.get("prediction") or {}
            record["prediction"] = {
                **prediction,
                "std": prediction.get("std", prediction.get("sd")),
                "source": config["engine"],
                "acquisition_function": selected.get("acquisition")
                or selected.get("engine_result", {}).get("acquisition")
                or selected.get("selection_reason", ""),
            }
        observations.append(record)
    suggestions = []
    for row in summary["suggestions"]:
        if row["status"] not in {"suggested", "pending"}:
            continue
        pred = row.get("prediction") or {}
        suggestions.append(
            {
                **row,
                "procedure": lookup[row["candidate_id"]].get("procedure", ""),
                "mean": pred.get("mean"),
                "std": pred.get("std", pred.get("sd")),
                "acquisition": row.get("score"),
                "source": config["engine"],
                "prediction_model": config["llm"]["forward_model"]
                if config["engine"] == "llm"
                else "",
                "acquisition_function": row.get("acquisition")
                or row.get("engine_result", {}).get("acquisition")
                or row.get("selection_reason", ""),
            }
        )
    compatible = comparison_campaigns(service, cid)
    random_campaign = random_control_campaign(service, cid)
    progress = deepcopy(summary["progress"])
    progress["status"] = {
        "suggested": "complete",
        "failed": "error",
        "exhausted": "complete",
    }.get(progress.get("status"), progress.get("status", "idle"))
    progress.update(
        label=summary["engine_label"],
        percent=0 if progress["status"] == "running" else 100,
    )
    payload.update(
        config=legacy_config(config),
        shared_campaign=summary,
        shared_config=config,
        campaign=dict(id="shared:" + cid, name=config["name"], saved=True),
        campaigns=state.list_campaigns() + catalog(service),
        dataset=dict(
            id=data["pool_fingerprint"],
            filename=config["name"],
            imported_at=data["created_at"],
        ),
        candidate_count=len(data["candidates"]),
        label_count=0,
        candidates=[candidate(r, i) for i, r in enumerate(data["candidates"][:500])],
        available_candidates=[
            candidate(r, i) for i, r in enumerate(service.eligible(data)[:500])
        ],
        available_count=summary["counts"]["available"],
        objective_names=[config["objective"]],
        observations=observations,
        suggestions=suggestions,
        inverse_designs=[],
        shared_inverse_proposals=summary.get("inverse_proposals", []),
        progress=progress,
        live_benchmark_run=None,
        last_error=progress.get("detail", "") if progress["status"] == "error" else "",
        last_model_status=summary["engine_label"]
        + (" · SYNTHETIC DEMO" if data["synthetic_demo"] else ""),
        events=[
            dict(time=e.get("at", ""), message=e.get("message", ""))
            for e in reversed(data["events"][-5:])
        ],
        acquisition_functions=[
            "expected_improvement",
            "probability_of_improvement",
            "upper_confidence_bound",
        ],
        embedding_cache=dict(
            total_count=0,
            cached_count=0,
            ready=config["engine"] == "gpr_features",
            model=config["llm"]["embedding_model"],
        ),
    )
    payload.update(
        plot_payload(data, comparisons=compatible, random_campaign=random_campaign)
    )
    payload["live_random_walk"] = random_control_state(service, cid)
    payload["shared_history"] = {
        key: deepcopy(data[key])
        for key in ("observations", "events", "archive", "provenance")
        if key in data
    }
    payload["shared_control_id"] = (
        random_campaign["campaign_id"] if random_campaign else None
    )
    from .llm_engine import managed_system_message

    # These are editing placeholders only. Full effective requests (including
    # custom text and selected examples) come from the read-only preview route.
    payload["shared_prompt_templates"] = {
        role: managed_system_message(
            role,
            prompt_style=config["data_schema"],
            objective_name=config["objective"],
            objective_units=config["units"],
        )
        for role in ("forward", "inverse")
    }
    if config["selection_policy"] == "random_control":
        payload["last_model_status"] = "Independent random control"
        payload["shared_campaign"]["engine_label"] = "Independent random control"
    if config["engine"] != "gpr_features" and not data["synthetic_demo"]:
        from .moc_http import cache_for

        try:
            cache = cache_for(service, cid)
            coverage = cache.coverage(data["candidates"])
            payload["embedding_cache"] = dict(
                total_count=coverage["requested"],
                cached_count=coverage["hit_count"],
                ready=not coverage["missing_ids"],
                model=cache.spec.model,
            )
        except ValueError as error:
            payload["embedding_cache"].update(ready=False, error=str(error))
    return payload


def get(handler, parsed):
    from .moc_http import service

    svc = service(handler)
    query = parse_qs(parsed.query)
    cid = query.get("campaign", [""])[0]
    action = parsed.path.rsplit("/", 1)[-1]
    if action == "state":
        payload = project(svc, cid, handler.state) if cid else handler.state.to_json()
        if not cid:
            payload["campaigns"] += catalog(svc)
        handler._send_json(payload)
        return
    data = svc.get(cid)
    if action == "history":
        bundle = svc.export(cid)
        handler._send_json(
            {
                key: bundle[key]
                for key in (
                    "campaign_id",
                    "config",
                    "observations",
                    "archive",
                    "suggestions",
                    "provenance",
                    "events",
                    "inverse_proposals",
                )
                if key in bundle
            }
        )
        return
    if action == "candidate-search":
        text = query.get("q", [""])[0].casefold()
        rows = [
            dict(
                id=r["candidate_id"],
                candidate_id=r["candidate_id"],
                procedure=r.get("procedure", ""),
                row=i + 1,
            )
            for i, r in enumerate(svc.eligible(data))
            if text in r.get("procedure", "").casefold()
            or text in r["candidate_id"].casefold()
        ]
        handler._send_json(
            dict(
                candidates=rows[:80],
                matched_count=len(rows),
                available_count=len(svc.eligible(data)),
            )
        )
        return
    if action == "export":
        kind = query.get("kind", ["observations"])[0]
        rows = data["candidates"] if kind == "procedures" else svc.active(data)
        output = StringIO()
        keys = list(
            dict.fromkeys(
                k
                for row in rows
                for k, v in row.items()
                if not isinstance(v, (dict, list))
            )
        )
        writer = csv.DictWriter(output, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handler._send_text(output.getvalue(), "text/csv")
        return
    raise ValueError("Unknown toolkit endpoint")


def post(handler, parsed):
    from .moc_http import service, prepare_cache

    svc = service(handler)
    p = handler._read_json()
    if not isinstance(p, dict):
        raise ValueError("Expected JSON object")
    cid = p.get("campaign")
    action = p.get("action")
    if parsed.path.endswith("create-generic"):
        records = p["records"]
        procedure_column = p.get("procedure_column")
        if procedure_column and procedure_column in {
            p["objective"],
            p.get("sigma_column"),
            "gof",
            "gap",
            "closure_gap",
            "closure_gap_wt_pct",
            "esd",
            "objective_sigma",
        }:
            raise ValueError(
                "Measured objectives and quality columns cannot be procedure text"
            )
        fields = [item["column"] for item in p["feature_spec"]]
        records = [
            {
                **row,
                "procedure": str(row.get(procedure_column, ""))
                if procedure_column
                else "; ".join(f"{key}: {row.get(key)}" for key in fields),
            }
            for row in records
        ]
        cid = svc.create_generic(
            records,
            p["feature_spec"],
            objective=p["objective"],
            direction=p["direction"],
            bounds=p["bounds"],
            name=p.get("name"),
            units=p.get("units", ""),
            sigma_column=p.get("sigma_column"),
            synthetic_demo=bool(getattr(handler, "moc_demo", False)),
        )
        handler._send_json(dict(campaign_id=cid))
        return
    data = svc.get(cid)
    if action == "config":
        svc.update_config(cid, config_changes(p["values"], data["config"]))
    elif action == "suggest":
        svc.start_suggestion(cid)
    elif action == "inverse-proposal":
        svc.start_inverse_proposals(cid)
    elif action == "cancel":
        svc.cancel_job(cid)
    elif action == "reserve":
        svc.reserve(cid, p["suggestion_id"])
    elif action == "release":
        svc.cancel_reservation(cid, p["suggestion_id"])
    elif action == "measure":
        svc.measure(cid, p["suggestion_id"], p["values"], p.get("request_id"))
    elif action == "refine":
        svc.refine(cid, p["observation_id"], p["values"], p["reason"])
    elif action == "save":
        if p.get("save_as"):
            bundle = svc.export(cid)
            bundle["config"]["name"] = p.get("name") or data["config"]["name"] + " copy"
            cid = svc.import_bundle(bundle)
        elif p.get("name"):
            svc.update_config(cid, {"name": p["name"]})
    elif action == "cache-prepare":
        prepare_cache(svc, cid)
    elif action == "random-start":
        from .campaign_controls import start_random_control

        start_random_control(svc, cid, p)
    elif action == "random-measure":
        from .campaign_controls import record_random_control

        record_random_control(svc, cid, p)
    elif action == "random-cancel":
        from .campaign_controls import clear_random_control

        clear_random_control(svc, cid)
    elif action == "reset-prompts":
        svc.update_config(
            cid,
            {"llm": {"forward_system_message": None, "inverse_system_message": None}},
        )
    else:
        raise ValueError("Unsupported action for a shared campaign")
    handler._send_json(project(svc, cid, handler.state))


def inspect_upload(filename, raw):
    """Read tabular values for explicit feature mapping; no model calls."""
    import pandas as pd

    frame = (
        pd.read_excel(BytesIO(raw))
        if filename.lower().endswith((".xlsx", ".xls"))
        else pd.read_csv(BytesIO(raw))
    )
    frame = frame.astype(object).where(pd.notna(frame), None)
    records = frame.to_dict(orient="records")
    columns = []
    for name in frame.columns:
        vals = [row[name] for row in records if row[name] is not None]
        numeric = bool(vals) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals
        )
        columns.append(
            dict(
                name=str(name),
                numeric=numeric,
                bounds=[min(vals), max(vals)] if numeric else None,
                values=list(dict.fromkeys(vals))[:100] if not numeric else None,
            )
        )
    return dict(columns=columns, records=records)
