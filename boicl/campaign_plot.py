"""Read-only campaign projection into the existing toolkit plot schema.

The browser's existing renderPlot draws these records. Only confirmed active
measurements affect best traces. Candidate tables (which may contain hidden
oracle columns) never supply plot statistics, measurements or random baselines.
"""
from copy import deepcopy
import math
from .measurement_quality import comparison_definition, quality_metadata


COLORS = ("#7c3aed", "#dc2626", "#0891b2", "#db2777", "#65a30d")


def _number(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _outcome(row, config):
    name = config.get("objective", "moc_wt_pct")
    return _number(row.get(name, row.get("value", row.get("moc_wt_pct"))))


def _physical_id(row, index):
    return str(
        row.get("physical_measurement_id")
        or row.get("observation_id")
        or f"ledger-{index}"
    )


def _prediction(raw):
    """Whitelisted display metadata, retaining uncertainty's scientific type."""
    raw = raw or {}
    mean = _number(raw.get("mean"))
    if mean is None:
        return {}
    standard_deviation = _number(raw.get("std", raw.get("sd")))
    if standard_deviation is not None and standard_deviation < 0:
        raise ValueError("Prediction standard deviation must be nonnegative")
    result = {
        "mean": mean,
        "std": standard_deviation,
        "uncertainty_type": raw.get("uncertainty_type", "model predictive uncertainty"),
        "measured": False,
    }
    for key in (
        "lower95",
        "upper95",
        "accepted_samples",
        "requested_samples",
        "sample_count",
        "source",
        "optimizer",
        "embedding_model",
        "prediction_model",
        "acquisition_function",
    ):
        if key in raw:
            result[key] = deepcopy(raw[key])
    return result


def measured_points(campaign):
    """One display position per active physical measurement, including seeds.

    Refinement replacements retain the original physical measurement position.
    Individual physical repeats retain separate positions and uncertainties.
    Initialization positions are supplied order, not optimization iterations.
    """
    config = campaign["config"]
    ledger = campaign.get("observations", [])
    order, ids_to_physical, active = {}, {}, {}
    for index, row in enumerate(ledger):
        physical = _physical_id(row, index)
        order.setdefault(physical, index)
        if row.get("observation_id"):
            ids_to_physical[row["observation_id"]] = physical
        if row.get("record_status", "measured") != "measured" or not row.get(
            "training_included", True
        ):
            continue
        value = _outcome(row, config)
        if value is None:
            continue
        if physical in active:
            raise ValueError(
                "Multiple active refinements for the same physical measurement cannot form separate plot points"
            )
        active[physical] = row
    prediction_by_physical = {}
    for suggestion in campaign.get("suggestions", []):
        physical = ids_to_physical.get(suggestion.get("observation_id"))
        if physical and suggestion.get("status") == "measured":
            prediction_by_physical[physical] = _prediction(suggestion.get("prediction"))
    # Read only original synthesis text from candidates; never objective labels.
    procedures = {
        str(row["candidate_id"]): str(row.get("procedure", ""))
        for row in campaign.get("candidates", [])
    }
    ordered = sorted(
        active.items(),
        key=lambda pair: (not bool(pair[1].get("is_seed")), order[pair[0]]),
    )
    points, initialized, completed = [], 0, 0
    for physical, row in ordered:
        seed = bool(row.get("is_seed", False))
        if seed:
            initialized += 1
        else:
            completed += 1
        sigma = _number(
            row.get(
                config.get("objective", "moc_wt_pct") + "_sigma",
                row.get(
                    "objective_sigma",
                    row.get("uncertainty", row.get("moc_wt_pct_sigma")),
                ),
            )
        )
        if sigma is not None and sigma < 0:
            raise ValueError("Measured uncertainty must be nonnegative")
        candidate_id = str(row["candidate_id"])
        points.append(
            {
                "index": len(points) + 1,
                "axis_label": f"i{initialized}" if seed else str(completed),
                "initialization_index": initialized if seed else None,
                "optimization_step": 0 if seed else completed,
                "candidate_id": candidate_id,
                "observation_id": row.get("observation_id"),
                "physical_measurement_id": physical,
                "procedure": procedures.get(
                    candidate_id, str(row.get("procedure", ""))
                ),
                "value": _outcome(row, config),
                "uncertainty": sigma,
                "measurement_uncertainty": sigma,
                "replicate_count": 1,
                "best_basis": "confirmed physical measurement",
                "is_seed": seed,
                "initialization": seed,
                "measured": True,
                "refinement_version": row.get("refinement_version"),
                "measurement_quality": quality_metadata(row),
                "measured_at": row.get("measured_at"),
                "recorded_at": row.get("recorded_at"),
                "chronology_basis": "initialization cohort"
                if seed
                else "physical measurement ledger order",
                "prediction": prediction_by_physical.get(physical, {}),
            }
        )
    return points


def best_trace(points, direction="maximize"):
    if direction not in ("maximize", "minimize"):
        raise ValueError("Objective direction must be maximize or minimize")
    seeds = [point for point in points if point.get("is_seed")]
    select = min if direction == "minimize" else max
    incumbent = None
    trace = []

    def append(point, initialization=False):
        trace.append(
            {
                "index": point["index"],
                "axis_label": point.get("axis_label", str(point["index"])),
                "initialization_index": point.get("initialization_index"),
                "optimization_step": point.get("optimization_step"),
                "value": point["value"],
                "best": incumbent["value"],
                "best_uncertainty": incumbent.get("uncertainty"),
                "best_replicate_count": 1,
                "best_candidate_id": incumbent["candidate_id"],
                "best_basis": "confirmed physical measurement",
                "baseline": initialization,
                "initialization_count": len(seeds) if initialization else 0,
            }
        )

    for point in points:
        incumbent = (
            point
            if incumbent is None
            else select((incumbent, point), key=lambda row: row["value"])
        )
        append(point, bool(point.get("is_seed")))
    return trace


def pending_predictions(campaign, completed_count, initialization_count=0):
    result = []
    for suggestion in campaign.get("suggestions", []):
        if suggestion.get("status") not in {"suggested", "pending"}:
            continue
        prediction = _prediction(suggestion.get("prediction"))
        if not prediction:
            continue
        spread = prediction["std"]
        step = completed_count + len(result) + 1
        result.append(
            {
                "index": initialization_count + step,
                "axis_label": str(step),
                "initialization_index": None,
                "optimization_step": step,
                "initialization": False,
                **prediction,
                "lower": prediction.get("lower95", prediction["mean"] - (spread or 0)),
                "upper": prediction.get("upper95", prediction["mean"] + (spread or 0)),
                "candidate_id": suggestion.get("candidate_id"),
                "suggestion_id": suggestion.get("suggestion_id"),
                "status": suggestion["status"],
                "count": 1,
                "is_pending_prediction": True,
            }
        )
    return result


def comparison_compatibility(reference, other):
    mismatches = []
    try:
        if comparison_definition(reference) != comparison_definition(other):
            mismatches.append("measurement_definition")
    except ValueError:
        mismatches.append("measurement_definition")
    for key in ("pool_fingerprint", "initialization_fingerprint"):
        if not reference.get(key) or reference.get(key) != other.get(key):
            mismatches.append(key)
    keys = ["objective", "units", "bounds", "direction", "repeat_policy"]
    if other["config"].get("selection_policy") != "random_control":
        keys.extend(("new_measurement_budget", "seed"))
    for key in keys:
        if reference["config"].get(key) != other["config"].get(key):
            mismatches.append(key)
    if reference.get("campaign_id") == other.get("campaign_id"):
        mismatches.append("same_campaign")
    return {
        "compatible": not mismatches,
        "mismatches": mismatches,
        "shared_later_outcomes": False,
        "measurement_budgets": [
            reference["config"].get("new_measurement_budget"),
            other["config"].get("new_measurement_budget"),
        ],
        "random_seeds": [reference["config"].get("seed"), other["config"].get("seed")],
    }


def comparison_run(reference, other, color=COLORS[0]):
    compatibility = comparison_compatibility(reference, other)
    if not compatibility["compatible"]:
        raise ValueError(
            "Campaign comparison requires matched inputs: "
            + ", ".join(compatibility["mismatches"])
        )
    points = measured_points(other)
    trace = best_trace(points, other["config"].get("direction", "maximize"))
    completed = sum(not point["is_seed"] for point in points)
    initialization_count = len(points) - completed
    summary = [
        {
            "index": row["index"],
            "axis_label": row["axis_label"],
            "initialization_index": row["initialization_index"],
            "optimization_step": row["optimization_step"],
            "mean": row["best"],
            "lower": row["best"],
            "upper": row["best"],
            "std": None,
            "count": 1,
            "measured": True,
            "baseline": row["baseline"],
            "spread_type": "single independent campaign; no replicate spread",
        }
        for row in trace
    ]
    return {
        "id": "campaign-comparison:" + other["campaign_id"],
        "campaign_id": other["campaign_id"],
        "name": other["config"].get("name", other["campaign_id"]),
        "color": color,
        "status": "live independent campaign",
        "partial": True,
        "kind": "independent_campaign_comparison",
        "initialization_count": initialization_count,
        "summary": summary,
        "prediction_summary": pending_predictions(
            other, completed, initialization_count
        ),
        "replicate_observations": [points],
        "config": {
            "optimizer": other["config"].get("engine"),
            "benchmark_replicates": 1,
            "acquisition": "campaign selection policy",
        },
        "compatibility": compatibility,
    }


def plot_payload(campaign, comparisons=(), random_campaign=None):
    """Return existing renderPlot keys; no new plotting implementation required."""
    points = measured_points(campaign)
    completed = sum(not point["is_seed"] for point in points)
    initialization_count = len(points) - completed
    config = campaign["config"]
    runs, diagnostics = [], []
    for other in comparisons:
        compatibility = comparison_compatibility(campaign, other)
        if compatibility["compatible"]:
            runs.append(
                comparison_run(campaign, other, COLORS[len(runs) % len(COLORS)])
            )
        else:
            diagnostics.append(
                {"campaign_id": other.get("campaign_id"), **compatibility}
            )
    pending = pending_predictions(campaign, completed, initialization_count)
    if pending:
        runs.append(
            {
                "id": "pending-predictions:" + campaign["campaign_id"],
                "name": "Awaiting measurement",
                "color": "#2563eb",
                "status": "prediction only",
                "partial": True,
                "kind": "pending_predictions",
                "initialization_count": initialization_count,
                "summary": [],
                "prediction_summary": pending,
                "replicate_observations": [],
                "config": {"benchmark_replicates": 1},
            }
        )
    random_points, random_trace, random_state = [], [], {}
    if random_campaign is not None:
        compatibility = comparison_compatibility(campaign, random_campaign)
        if not compatibility["compatible"]:
            raise ValueError(
                "Random control must use the same initialization and candidate policy"
            )
        random_points = measured_points(random_campaign)
        random_trace = best_trace(random_points, config.get("direction", "maximize"))
        random_state = {
            "campaign_id": random_campaign["campaign_id"],
            "observations": random_points,
            "initialization_count": sum(point["is_seed"] for point in random_points),
            "status": "independent measured random control",
            "training_shared": False,
        }
    bounds = config.get("bounds") or [None, None]
    axis_points = points + random_points
    for run in runs:
        axis_points.extend(run.get("summary", []))
        axis_points.extend(run.get("prediction_summary", []))
    last_position = max((point["index"] for point in axis_points), default=0)
    return {
        "initialization_count": initialization_count,
        "live_observation_points": points,
        "best_trace": best_trace(points, config.get("direction", "maximize")),
        "benchmark_runs": runs,
        "live_random_walk": random_state,
        "live_random_walk_trace": random_trace,
        "random_walk_trace": [],
        "dataset_stats": [],
        "plot_objective_bounds": {"lower": bounds[0], "upper": bounds[1]},
        "plot_x_axis": {
            "min": 0.5,
            "label": "Initialization and new completed measurements",
            "initialization_count": initialization_count,
            "initialization_end": initialization_count + 0.5
            if initialization_count
            else None,
            "labels": [
                {
                    "index": index,
                    "label": f"i{index}"
                    if index <= initialization_count
                    else str(index - initialization_count),
                    "initialization": index <= initialization_count,
                }
                for index in range(1, last_position + 1)
            ],
        },
        "plot_counts": {
            "initialization": initialization_count,
            "new_completed": completed,
            "pending_predictions": len(pending),
        },
        "comparison_diagnostics": diagnostics,
        "random_baseline_note": "Live random control requires its own measured results; no hidden candidate labels were used",
        "chronology_note": "Seed order is not experiment chronology; subsequent positions follow physical measurement ledger order",
    }
