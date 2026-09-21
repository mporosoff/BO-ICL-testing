"""Shared live random-control actions for the existing toolkit controls.

Controls are independent campaigns with the same immutable initialization. All
selection, reservations, measurements and persistence use CampaignService; no
candidate labels or model predictions are converted to measured outcomes.
"""
from copy import deepcopy

from .campaign_plot import best_trace, comparison_compatibility, measured_points


def _control_id(service, parent_cid):
    choices = []
    for index, item in enumerate(service.list()):
        if item.get("selection_policy") != "random_control":
            continue
        if item.get("comparison_parent_id") != parent_cid:
            continue
        choices.append((item.get("created_at", ""), index, item["campaign_id"]))
    return max(choices)[2] if choices else None


def random_control_campaign(service, parent_cid):
    """Read the current independent random arm without creating state."""
    control = _control_id(service, parent_cid)
    return service.view_snapshot(control) if control else None


def comparison_campaigns(service, parent_cid):
    parent = service.view_snapshot(parent_cid)
    return [
        candidate
        for row in service.list()
        if row["campaign_id"] != parent_cid
        and row.get("selection_policy", "engine") == "engine"
        for candidate in [service.view_snapshot(row["campaign_id"])]
        if candidate["config"].get("selection_policy", "engine") == "engine"
        and comparison_compatibility(parent, candidate)["compatible"]
    ]


def random_control_state(service, parent_cid):
    data = random_control_campaign(service, parent_cid)
    if data is None:
        return {}
    points = measured_points(data)
    new_points = [point for point in points if not point["is_seed"]]
    pending = next(
        (row for row in reversed(data["suggestions"]) if row["status"] == "pending"),
        None,
    )
    current = None
    if pending:
        candidate = next(
            row
            for row in data["candidates"]
            if row["candidate_id"] == pending["candidate_id"]
        )
        current = {
            "id": candidate["candidate_id"],
            "candidate_id": candidate["candidate_id"],
            "suggestion_id": pending["suggestion_id"],
            "procedure": candidate["procedure"],
        }
    budget = data["config"].get("new_measurement_budget")
    status = (
        "waiting_for_result"
        if current
        else "complete"
        if budget is not None and len(new_points) >= budget
        else "idle"
    )
    return {
        "campaign_id": data["campaign_id"],
        "comparison_parent_id": parent_cid,
        "status": status,
        "target_count": budget,
        "seed": data["config"]["seed"],
        "observations": new_points,
        "initialization_observations": [point for point in points if point["is_seed"]],
        "current_candidate": current,
        "training_shared": False,
        "trace": best_trace(points, data["config"].get("direction", "maximize")),
    }


def _ensure_pending(service, control_cid):
    data = service.get(control_cid)
    if any(row["status"] == "pending" for row in data["suggestions"]):
        return
    budget = data["config"].get("new_measurement_budget")
    if budget is not None and service.completed(data) >= budget:
        return
    if not service.eligible(data):
        return
    proposal = next(
        (
            row
            for row in reversed(data["suggestions"])
            if row["status"] == "suggested"
            and row["history_revision"] == data["history_revision"]
        ),
        None,
    )
    if proposal is None:
        service.start_suggestion(control_cid, background=False)
        data = service.get(control_cid)
        proposal = next(
            (
                row
                for row in reversed(data["suggestions"])
                if row["status"] == "suggested"
                and row["history_revision"] == data["history_revision"]
            ),
            None,
        )
    if proposal is None:
        progress = service.summary(control_cid).get("progress", {})
        raise ValueError(
            "Random-control selection did not complete: "
            + str(progress.get("detail", "no eligible proposal"))
        )
    service.reserve(control_cid, proposal["suggestion_id"])


def start_random_control(service, parent_cid, payload=None):
    payload = dict(payload or {})
    parent = service.get(parent_cid)
    requested = payload.get("target_count")
    if requested not in (None, ""):
        if isinstance(requested, bool):
            raise ValueError(
                "Random-control target must be a nonnegative count of new measurements"
            )
        number = float(requested)
        if not number.is_integer() or number < 0:
            raise ValueError(
                "Random-control target must be a nonnegative count of new measurements"
            )
        requested = int(number)
        parent_budget = parent["config"].get("new_measurement_budget")
        if parent_budget is not None and requested > parent_budget:
            raise ValueError(
                "Random-control target exceeds the parent campaign's measurement budget"
            )
    control_cid = service.create_control(
        parent_cid, reset=bool(payload.get("reset", False))
    )
    changes = {"auto_suggest": False}
    if requested not in (None, ""):
        changes["new_measurement_budget"] = requested
    if payload.get("seed") not in (None, ""):
        number = float(payload["seed"])
        if not number.is_integer() or number < 0:
            raise ValueError("Random-control seed must be a nonnegative integer")
        changes["seed"] = int(number)
    current_config = service.get(control_cid)["config"]
    changes = {
        key: value for key, value in changes.items() if current_config.get(key) != value
    }
    if changes:
        service.update_config(control_cid, changes)
    _ensure_pending(service, control_cid)
    return random_control_state(service, parent_cid)


def record_random_control(service, parent_cid, payload):
    control_cid = _control_id(service, parent_cid)
    if control_cid is None:
        raise ValueError(
            "Start the independent random control before recording its measurement"
        )
    data = service.get(control_cid)
    suggestion_id = payload.get("suggestion_id")
    if not suggestion_id:
        raise ValueError(
            "Include the displayed random suggestion_id to reject delayed duplicate submissions"
        )
    suggestion = next(
        (row for row in data["suggestions"] if row["suggestion_id"] == suggestion_id),
        None,
    )
    if suggestion is None:
        raise ValueError(
            "The random-control suggestion belongs to a different control campaign"
        )
    if (
        payload.get("candidate_id")
        and payload["candidate_id"] != suggestion["candidate_id"]
    ):
        raise ValueError("Random-control result candidate differs from its reservation")
    objective = data["config"].get("objective", "moc_wt_pct")
    values = deepcopy(payload.get("values", {}))
    if not isinstance(values, dict):
        raise ValueError("Measured values must be an object")
    if "value" in payload:
        values[objective] = payload["value"]
        values["value"] = payload["value"]
    if payload.get("uncertainty") not in (None, ""):
        values[objective + "_sigma"] = payload["uncertainty"]
        values["objective_sigma"] = payload["uncertainty"]
        values["uncertainty"] = payload["uncertainty"]
    for key in (
        "gof",
        "closure_gap",
        "closure_gap_wt_pct",
        "closure_gap_origin",
        "measured_at",
        "source_note",
    ):
        if key in payload:
            values[key] = payload[key]
    values.setdefault(
        "source_note", "Independent random-control measurement entered by operator"
    )
    service.measure(
        control_cid,
        suggestion_id,
        values,
        request_id=payload.get("request_id") or "random-control:" + suggestion_id,
        refresh=False,
    )
    _ensure_pending(service, control_cid)
    return random_control_state(service, parent_cid)


def clear_random_control(service, parent_cid):
    """Release pending work and create a fresh arm while preserving old history."""
    control_cid = _control_id(service, parent_cid)
    if control_cid:
        service.cancel_job(control_cid)
        for proposal in service.get(control_cid)["suggestions"]:
            if proposal["status"] == "pending":
                service.cancel_reservation(control_cid, proposal["suggestion_id"])
    new_control = service.create_control(parent_cid, reset=True)
    service.update_config(new_control, {"auto_suggest": False})
    return random_control_state(service, parent_cid)


cancel_random_control = clear_random_control
