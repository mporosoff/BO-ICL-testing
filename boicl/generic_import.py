"""Explicit, outcome-free generic candidate mapping and fixed-space features."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
from .measurement_quality import quality_metadata, QUALITY_FIELDS as DEFINITION_FIELDS


SCHEMA = "generic-campaign-v1"
QUALITY_FIELDS = {
    "objective_sigma",
    "sigma",
    "esd",
    "gof",
    "GOF",
    "gap",
    "closure_gap",
    "closure_gap_origin",
    "moc_wt_pct_sigma",
    "closure_gap_wt_pct",
    "phase_accounting_residual_wt_pct",
}
QUALITY_FIELDS.update(DEFINITION_FIELDS | {"measurement_quality"})


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def validate_bounds(bounds):
    if bounds is None:
        return None
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("Objective bounds must be two finite numbers or None")
    lo, hi = (number(value, "objective bound") for value in bounds)
    if lo >= hi:
        raise ValueError("Objective lower bound must be smaller than upper bound")
    return [lo, hi]


def resolve_features(spec, records, *, objective="value"):
    if not isinstance(spec, list):
        raise ValueError("Feature mapping must be an ordered list")
    result, seen = [], set()
    for item in spec:
        if not isinstance(item, dict) or set(item) - {
            "column",
            "transform",
            "bounds",
            "values",
            "units",
            "encoding",
        }:
            raise ValueError(
                "Feature entries contain column, transform, bounds or category values, and optional units"
            )
        column = item.get("column")
        if not isinstance(column, str) or not column or column in seen:
            raise ValueError("Feature columns must be present and unique")
        if column == objective or column == "value" or column in QUALITY_FIELDS:
            raise ValueError(
                "Objective and measurement-quality columns cannot be synthesis features"
            )
        seen.add(column)
        transform = item.get("transform", "linear")
        if transform not in {"linear", "log", "log2", "log10", "categorical"}:
            raise ValueError(f"Unsupported feature transform: {transform}")
        normalized = {
            "column": column,
            "transform": transform,
            "units": str(item.get("units", "")),
        }
        values = [row.get(column) for row in records]
        if transform == "categorical":
            if any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Categorical features require nonempty string values")
            categories = item.get("values", list(dict.fromkeys(values)))
            if (
                not isinstance(categories, list)
                or not categories
                or len(categories) != len(set(categories))
                or any(not isinstance(v, str) or not v for v in categories)
            ):
                raise ValueError("Categories must be a nonempty unique list of strings")
            if any(v not in categories for v in values):
                raise ValueError(f"Unknown category in {column}")
            normalized["values"] = list(categories)
            normalized["encoding"] = "one_hot"  # no arbitrary ordered distances
        else:
            values = [number(v, column) for v in values]
            bounds = item.get("bounds")
            if bounds is None:
                if not values:
                    raise ValueError(
                        "Full-space bounds require candidates or explicit values"
                    )
                bounds = [min(values), max(values)]
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(
                    "Feature bounds must contain lower and upper full-space values"
                )
            lo, hi = (number(v, column + " bound") for v in bounds)
            if lo > hi or any(v < lo or v > hi for v in values):
                raise ValueError(f"Feature {column} is outside its full-space bounds")
            if transform != "linear" and (lo <= 0 or any(v <= 0 for v in values)):
                raise ValueError(
                    f"Logarithmic feature {column} requires positive values"
                )
            normalized["bounds"] = [lo, hi]
        result.append(normalized)
    return result


class FeatureTransform:
    def __init__(self, spec):
        self.spec = deepcopy(spec)
        self.order = []
        for item in spec:
            self.order.extend(
                [f"{item['column']}={c}" for c in item["values"]]
                if item["transform"] == "categorical"
                else [item["column"]]
            )
        self.dimension = len(self.order)
        if self.dimension == 0:
            raise ValueError("Structured GP requires at least one mapped feature")

    def transform(self, rows):
        output = []
        functions = {
            "linear": lambda x: x,
            "log": np.log,
            "log2": np.log2,
            "log10": np.log10,
        }
        for row in rows:
            current = []
            for item in self.spec:
                column, kind = item["column"], item["transform"]
                if kind == "categorical":
                    value = row.get(column)
                    if value not in item["values"]:
                        raise ValueError(f"Unknown category for {column}")
                    current.extend(
                        [0.0]
                        if len(item["values"]) == 1
                        else [float(value == category) for category in item["values"]]
                    )
                else:
                    value = number(row.get(column), column)
                    lo, hi = item["bounds"]
                    if value < lo or value > hi or (kind != "linear" and value <= 0):
                        raise ValueError(f"Feature {column} outside configured bounds")
                    fn = functions[kind]
                    current.append(
                        0.0
                        if lo == hi
                        else float((fn(value) - fn(lo)) / (fn(hi) - fn(lo)))
                    )
            output.append(current)
        return np.asarray(output, dtype=float).reshape(-1, self.dimension)


def generic_measurement(values, *, bounds=None, objective="value"):
    value = number(values.get("value", values.get(objective)), objective)
    bounds = validate_bounds(bounds)
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        raise ValueError("Measured objective is outside configured bounds")
    result = {
        "value": value,
        "source_note": str(values.get("source_note", "")),
        "measurement_quality": quality_metadata(values),
    }
    for field in ("objective_sigma", "gof", "closure_gap"):
        if values.get(field) not in (None, ""):
            val = number(values[field], field)
            if field != "closure_gap" and val < 0:
                raise ValueError(f"{field} cannot be negative")
            result[field] = val
    if values.get("gap") not in (None, ""):
        gap = number(values["gap"], "gap")
        if "closure_gap" in result and result["closure_gap"] != gap:
            raise ValueError("gap and closure_gap report conflicting quality values")
        result["closure_gap"] = gap
        result["closure_gap_origin"] = "reported gap field; completeness unspecified"
    if values.get("closure_gap_origin") is not None:
        result["closure_gap_origin"] = str(values["closure_gap_origin"])
    return result


def pool_fingerprint(candidates):
    # Both recipe features and exact procedure text contribute to state identity.
    return _hash(candidates)


def validate_generic_candidates(candidates, feature_spec):
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A generic campaign requires a nonempty candidate list")
    ids, designs = set(), set()
    feature_columns = [item["column"] for item in feature_spec]
    for row in candidates:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("candidate_id"), str)
            or not row["candidate_id"]
            or row["candidate_id"] in ids
        ):
            raise ValueError("Generic candidate IDs must be nonempty and unique")
        if not isinstance(row.get("procedure"), str) or not row["procedure"].strip():
            raise ValueError(
                "Candidates require nonempty procedure text or mapped features"
            )
        if hashlib.sha256(row["procedure"].encode()).hexdigest() != row.get(
            "procedure_sha256"
        ):
            raise ValueError("Generic procedure checksum mismatch")
        if set(row) != {
            "candidate_id",
            "procedure",
            "procedure_sha256",
            *feature_columns,
        }:
            raise ValueError(
                "Generic candidates may contain only identity, procedure and mapped features; labels belong in observations"
            )
        identity = _hash(
            {
                "features": {key: row[key] for key in feature_columns},
                "procedure": row["procedure"],
            }
        )
        if identity in designs:
            raise ValueError(
                "Duplicate generic design; record repetitions as separate observations"
            )
        ids.add(row["candidate_id"])
        designs.add(identity)
    if feature_spec:
        FeatureTransform(feature_spec).transform(candidates)
    return candidates


def load_generic_package(
    records,
    feature_spec,
    *,
    objective="value",
    bounds=None,
    observations=None,
    procedure_column="procedure",
    id_column="candidate_id",
    sigma_column=None,
):
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(r, dict) for r in records)
    ):
        raise ValueError("Generic records must be a nonempty list of objects")
    if (
        procedure_column == objective
        or (sigma_column is not None and procedure_column == sigma_column)
        or procedure_column in QUALITY_FIELDS
    ):
        raise ValueError(
            "Measured objective and quality columns cannot be procedure inputs"
        )
    resolved = resolve_features(feature_spec, records, objective=objective)
    if sigma_column is not None and sigma_column in {
        item["column"] for item in resolved
    }:
        raise ValueError(
            "Reported uncertainty is measurement quality, not a synthesis feature"
        )
    bounds = validate_bounds(bounds)
    candidates, measured = [], []
    for index, row in enumerate(records):
        feature_values = {
            item["column"]: str(row[item["column"]])
            if item["transform"] == "categorical"
            else number(row[item["column"]], item["column"])
            for item in resolved
        }
        procedure = row.get(procedure_column)
        if procedure in (None, ""):
            if not feature_values:
                raise ValueError(
                    "Supply procedure text or explicitly mapped feature columns"
                )
            procedure = "; ".join(
                f"{key}: {value}" for key, value in feature_values.items()
            )
        if not isinstance(procedure, str):
            raise ValueError("Procedure text must be a string")
        cid = (
            row.get(id_column)
            or "design-"
            + _hash({"features": feature_values, "procedure": procedure})[:20]
        )
        candidate = dict(
            feature_values,
            candidate_id=str(cid),
            procedure=procedure,
            procedure_sha256=hashlib.sha256(procedure.encode()).hexdigest(),
        )
        candidates.append(candidate)
        # Separate observation mappings take precedence; no hidden oracle is read.
        if observations is None and row.get(objective) not in (None, ""):
            values = {
                "value": row[objective],
                "source_note": row.get("source_note", ""),
            }
            for field in (
                "objective_sigma",
                "gof",
                "gap",
                "closure_gap",
                "closure_gap_origin",
                "measurement_quality",
                *sorted(DEFINITION_FIELDS),
            ):
                if row.get(field) is not None:
                    values[field] = row[field]
            if sigma_column is not None:
                values["objective_sigma"] = row.get(sigma_column)
            measured.append(
                dict(
                    generic_measurement(values, bounds=bounds, objective=objective),
                    candidate_id=str(cid),
                    observation_id=f"initial-{index}",
                )
            )
    if observations is not None:
        if not isinstance(observations, list):
            raise ValueError("Observations must be a separate list")
        for index, row in enumerate(observations):
            values = dict(row)
            if sigma_column is not None and sigma_column in row:
                values["objective_sigma"] = row[sigma_column]
            measured.append(
                dict(
                    generic_measurement(values, bounds=bounds, objective=objective),
                    candidate_id=str(row.get("candidate_id", row.get(id_column, ""))),
                    observation_id=str(row.get("observation_id", f"initial-{index}")),
                    physical_measurement_id=str(
                        row.get(
                            "physical_measurement_id",
                            row.get("observation_id", f"initial-{index}"),
                        )
                    ),
                )
            )
    validate_generic_candidates(candidates, resolved)
    ids = {c["candidate_id"] for c in candidates}
    for row in measured:
        if row["candidate_id"] not in ids:
            raise ValueError("Observation ID does not map to the candidate pool")
        row.update(
            physical_measurement_id=row.get(
                "physical_measurement_id", row["observation_id"]
            ),
            training_included=True,
            record_status="measured",
            is_seed=True,
            refinement_version="import-v1",
            measured_at=None,
            chronology="import order; experimental dates unspecified",
            objective_authority="explicit user-mapped measured initialization",
        )
    return dict(
        candidates=candidates,
        observations=measured,
        archive=[],
        pool_fingerprint=pool_fingerprint(candidates),
        provenance={
            "schema": SCHEMA,
            "feature_spec": resolved,
            "objective_column": objective,
            "quality_aliases": (
                {"gap": "closure_gap"}
                if any(
                    row.get("gap") not in (None, "")
                    and (
                        observations is not None or row.get(objective) not in (None, "")
                    )
                    for row in (observations if observations is not None else records)
                )
                else {}
            ),
            "initialization": "user-supplied measured initialization; not model-selected steps",
            "feature_scaling": "fixed full candidate space, independent of observed objectives",
        },
    )
