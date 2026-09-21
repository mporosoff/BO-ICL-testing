"""Versioned reported measurement definitions; never infer historical quantification."""
from copy import deepcopy


METHODS = {
    "gsas_ii_mass_fraction",
    "xrd_area_fraction",
    "other",
    "historical_unspecified",
}
QUALITY_FIELDS = {
    "quantification_method",
    "normalization",
    "source_file",
    "source_identifier",
    "refinement_id",
    "uncertainty_method",
    "definition_note",
}


def quality_metadata(values=None):
    values = values or {}
    if "measurement_quality" in values:
        raw = values["measurement_quality"]
        if not isinstance(raw, dict):
            raise ValueError("Measurement quality must be a versioned object")
    else:
        raw = {key: values[key] for key in QUALITY_FIELDS if key in values}
    if set(raw) - (QUALITY_FIELDS | {"schema_version"}):
        raise ValueError("Unknown measurement-quality provenance field")
    if raw.get("schema_version", 1) != 1 or isinstance(raw.get("schema_version"), bool):
        raise ValueError("Unsupported measurement-quality schema")
    result = dict(schema_version=1, quantification_method="historical_unspecified")
    result.update({key: None for key in QUALITY_FIELDS - {"quantification_method"}})
    for key, value in raw.items():
        if key == "schema_version":
            continue
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Measurement quality {key} must be text or null")
        result[key] = value.strip() or None if isinstance(value, str) else value
    if result["quantification_method"] not in METHODS:
        raise ValueError(
            "Choose an explicit quantification method or historical_unspecified"
        )
    if result["quantification_method"] == "other" and not result["definition_note"]:
        raise ValueError("Other quantification methods require a definition note")
    return result


def definition_signature(row):
    quality = quality_metadata(row)
    if quality["quantification_method"] == "historical_unspecified":
        return None
    return (
        quality["quantification_method"],
        quality["normalization"],
        quality["definition_note"]
        if quality["quantification_method"] == "other"
        else None,
    )


def default_definition():
    return dict(
        schema_version=1,
        quantification_method="historical_unspecified",
        normalization=None,
        definition_note=None,
        historical_policy="unresolved",
        historical_observation_ids=[],
        decision_reason=None,
    )


def resolve_definition(value):
    if not isinstance(value, dict) or set(value) - set(default_definition()):
        raise ValueError("Invalid campaign measurement definition")
    result = {**default_definition(), **deepcopy(value)}
    quality = quality_metadata(
        {
            key: result[key]
            for key in ("quantification_method", "normalization", "definition_note")
        }
    )
    result.update(
        {
            key: quality[key]
            for key in ("quantification_method", "normalization", "definition_note")
        }
    )
    if result["schema_version"] != 1 or isinstance(result["schema_version"], bool):
        raise ValueError("Unsupported measurement-definition schema")
    if result["historical_policy"] not in {
        "unresolved",
        "exclude",
        "retain_with_justification",
    }:
        raise ValueError("Unknown historical measurement decision")
    ids = result["historical_observation_ids"]
    if (
        not isinstance(ids, list)
        or any(not isinstance(v, str) for v in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(
            "Historical measurement decision requires unique observation IDs"
        )
    if result["decision_reason"] is not None and not isinstance(
        result["decision_reason"], str
    ):
        raise ValueError("Scientific decision reason must be text")
    if (
        result["historical_policy"] != "unresolved"
        and not (result["decision_reason"] or "").strip()
    ):
        raise ValueError(
            "Document the scientific reason for retaining or excluding historical measurements"
        )
    return result


def active_records(data):
    return [
        r
        for r in data["observations"]
        if r.get("training_included", True) and r.get("record_status") == "measured"
    ]


def validate_training_definitions(data):
    records = active_records(data)
    signatures = {definition_signature(row) for row in records}
    explicit = signatures - {None}
    definition = resolve_definition(
        data["config"].get("measurement_definition", default_definition())
    )
    configured = definition_signature(definition)
    if len(explicit) > 1 or (configured is not None and explicit - {configured}):
        raise ValueError(
            "Incompatible measurement definitions cannot be combined for training; revise the definition and explicitly exclude incompatible records"
        )
    unknown = [r for r in records if definition_signature(r) is None]
    if (explicit or configured is not None) and unknown:
        allowed = set(definition["historical_observation_ids"])
        if definition["historical_policy"] != "retain_with_justification" or any(
            r["observation_id"] not in allowed for r in unknown
        ):
            raise ValueError(
                "Historical quantification is unspecified; record a scientific decision to exclude it or retain it with justification before mixing with an explicit method"
            )
    return definition


def quality_status(data):
    rows = active_records(data)
    unknown = [r["observation_id"] for r in rows if definition_signature(r) is None]
    definition = data["config"].get("measurement_definition", default_definition())
    return dict(
        schema_version=1,
        unknown_active_observation_ids=unknown,
        excluded_count=sum(
            r.get("record_status") == "measured"
            and not r.get("training_included", True)
            for r in data["observations"]
        ),
        scientific_validation_required=bool(unknown),
        historical_policy=definition["historical_policy"],
        decision_reason=definition["decision_reason"],
        note="Historical quantification remains unspecified; a numerical value does not establish mass-fraction or area-fraction equivalence."
        if unknown
        else "Reported definitions are explicit; independent scientific validation remains the operator's responsibility.",
    )


def comparison_definition(data):
    definition = validate_training_definitions(data)
    declared = definition_signature(definition)
    # A declared cohort remains the same while independent arms collect results
    # at different times, including before their first compatible measurement.
    # Legacy undeclared cohorts still need their reported explicit basis checked.
    effective = declared
    if effective is None:
        effective = next(
            (
                signature
                for row in active_records(data)
                if (signature := definition_signature(row)) is not None
            ),
            None,
        )
    return dict(
        declared=declared,
        effective=effective,
        historical_policy=definition["historical_policy"],
    )
