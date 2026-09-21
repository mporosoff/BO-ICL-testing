"""Versioned, validated configuration shared by MoC library, browser and CLI."""
from copy import deepcopy
import math
import re
from .measurement_quality import default_definition, resolve_definition

SCHEMA_VERSION = 1
ENGINE_LABELS = {
    "gpr_features": "GP: synthesis parameters",
    "llm": "BO-ICL: LLM",
    "gpr_embeddings": "GP: text embeddings",
}
PRESETS = {
    "moc_five_gp": (
        "MoC five-point comparison — six-variable GP",
        "gpr_features",
        "confirmed_three",
    ),
    "moc_five_llm": (
        "MoC five-point comparison — BO-ICL LLM",
        "llm",
        "confirmed_three",
    ),
    "moc_gp": (
        "MoC 810c3f7 — structured GP continuation",
        "gpr_features",
        "confirmed_three",
    ),
    "moc_llm": ("MoC 810c3f7 — matched LLM replicate", "llm", "confirmed_three"),
    "moc_embedding_gp": (
        "MoC 810c3f7 — embedding GP baseline",
        "gpr_embeddings",
        "confirmed_three",
    ),
    "moc_eight": (
        "MoC T1 — eight-observation LLM continuation",
        "llm",
        "eight_observations",
    ),
    "generic_gp": ("Structured-input GP", "gpr_features", "user_mapped"),
    "generic_llm": ("BO-ICL: LLM", "llm", "user_mapped"),
    "generic_embedding_gp": ("Text-embedding GP", "gpr_embeddings", "user_mapped"),
}

PRESET_VERSIONS = {preset: "1.0.0" for preset in PRESETS}
STUDY_SETTINGS = {
    "moc_five_gp": {
        "new_measurement_budget": 5,
        "auto_suggest": False,
        "structured_gp": {"ei_after_unique_measured_designs": 3},
    },
    "moc_five_llm": {"new_measurement_budget": 5, "auto_suggest": False},
}

# Reusable numerical/model defaults contain no MoC objective, bounds or seed data.
ENGINE_DEFAULTS = {
    "llm": dict(
        forward_model="gpt-4o",
        inverse_model="gpt-4o",
        forward_temperature=0.7,
        inverse_temperature=0.7,
        forward_max_tokens=256,
        inverse_max_tokens=576,
        n_samples=5,
        min_samples=2,
        inverse_n=1,
        inverse_proposal_count=1,
        manual_inverse_target=None,
        uncertainty_scalar=1.0,
        selector_k=5,
        selector_mode="nearest",
        shortlist_size=16,
        fetch_k=100,
        mmr_lambda=0.5,
        embedding_model="text-embedding-3-large",
        selector_embedding_model="text-embedding-3-large",
        seed=616,
        acquisition="expected_improvement",
        xi=0.0,
        maximize=True,
        objective_bounds=None,
        inverse_multiplier=1.2,
        inverse_jitter=0.05,
        reference_scale=1.0,
        target_floor=None,
        ucb_lambda=0.5,
        random_addons=0,
        target_ceiling=None,
        forward_system_message=None,
        inverse_system_message=None,
        include_phase_context=False,
    ),
    "structured_gp": dict(
        burn_in=1000,
        retained_draws=4000,
        predict_thin=20,
        proposal_step=0.3,
        seed=616,
        gof_power=1.0,
        sigma_floor_pp=0.5,
        logit_delta_pp=0.65,
        ei_after_unique_measured_designs=2,
        ei_xi_standardized_logit=0.01,
        missing_metadata_policy="error",
        metadata_fallbacks={},
        chunk_size=512,
        quadrature_nodes=96,
        feature_spec=[],
        noise_policy="reported_or_fixed",
        default_observation_sigma=1.0,
    ),
    "embedding_gp": dict(
        embedding_model="text-embedding-ada-002",
        dimensions=32,
        neighbors=5,
        seed=616,
    ),
}


def resolve_config(preset="moc_llm", overrides=None):
    if preset not in PRESETS:
        raise ValueError("Unknown campaign preset")
    name, engine, initialization = PRESETS[preset]
    generic = preset.startswith("generic_")
    result = dict(
        schema_version=SCHEMA_VERSION,
        profile_version="moc-corrected-v1",
        preset=preset,
        preset_version=PRESET_VERSIONS[preset],
        preset_provenance=dict(
            preset_id=preset,
            version=PRESET_VERSIONS[preset],
            display_name=name,
            origin="builtin",
        ),
        name=name,
        engine=engine,
        initialization=initialization,
        workflow_mode="live",
        objective="moc_wt_pct",
        units="wt%",
        bounds=[0.0, 100.0],
        direction="maximize",
        seed=616,
        data_schema="generic" if generic else "moc",
        selection_policy="engine",
        comparison_parent_id=None,
        batch_size=1,
        new_measurement_budget=None,
        measurements_per_candidate=1,
        independent_campaign_replicates=1,
        repeat_policy="source_reset_quality_repeats",
        auto_suggest=True,
        measurement_definition=default_definition(),
        api=dict(maximum_attempts=8, request_spacing_s=0.5, base_cooldown_s=10.0),
        llm=deepcopy(ENGINE_DEFAULTS["llm"]),
        structured_gp=deepcopy(ENGINE_DEFAULTS["structured_gp"]),
        embedding_gp=deepcopy(ENGINE_DEFAULTS["embedding_gp"]),
    )
    if not generic:
        result["llm"].update(
            include_phase_context=True,
            reference_scale=100.0,
            objective_bounds=[0.0, 100.0],
        )
        result["structured_gp"].update(
            feature_spec=None,
            noise_policy="moc_quality",
            ei_after_unique_measured_designs=10,
        )
    if generic:
        result.update(
            profile_version="generic-shared-v1",
            objective="value",
            units="",
            bounds=None,
            repeat_policy="exclude_all_historical",
        )
        result["structured_gp"].update(
            feature_spec=[],
            noise_policy="reported_or_fixed",
            ei_after_unique_measured_designs=2,
        )
        result["llm"].update(include_phase_context=False, reference_scale=1.0)
    for key, value in STUDY_SETTINGS.get(preset, {}).items():
        if isinstance(value, dict):
            result[key].update(deepcopy(value))
        else:
            result[key] = deepcopy(value)
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Configuration overrides must be an object")
    if overrides and "schema_version" in overrides and "preset" in overrides:
        # Older complete saved configurations have no identifiable factory version.
        # Preserve their effective settings without claiming the current preset.
        if "preset_version" not in overrides:
            result["preset_version"] = "legacy-unversioned"
            result["preset_provenance"].update(
                version="legacy-unversioned", origin="legacy_saved_configuration"
            )
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for key, value in overrides.items():
            if isinstance(result.get(key), dict):
                if not isinstance(value, dict):
                    raise ValueError(f"{key} settings must be an object")
                extra = set(value) - set(result[key])
                if extra:
                    raise ValueError(
                        f"Unknown {key} configuration fields: {sorted(extra)}"
                    )
                result[key].update(deepcopy(value))
            else:
                result[key] = deepcopy(value)
    if (
        isinstance(result["schema_version"], bool)
        or result["schema_version"] != SCHEMA_VERSION
        or result["engine"] not in ENGINE_LABELS
    ):
        raise ValueError(
            "Unsupported schema or engine; no automatic engine substitution"
        )
    if result["preset"] != preset or result["profile_version"] != (
        "generic-shared-v1" if generic else "moc-corrected-v1"
    ):
        raise ValueError(
            "Preset and profile version must match the resolved configuration"
        )
    version = result["preset_version"]
    provenance = result["preset_provenance"]
    if (
        not isinstance(version, str)
        or not (
            re.fullmatch(r"\d+\.\d+\.\d+", version) or version == "legacy-unversioned"
        )
        or provenance["preset_id"] != preset
        or provenance["version"] != version
        or provenance["origin"] not in {"builtin", "legacy_saved_configuration"}
        or not isinstance(provenance["display_name"], str)
        or not provenance["display_name"].strip()
    ):
        raise ValueError("Invalid saved preset provenance or version")
    if result["initialization"] not in (
        {"user_mapped"} if generic else {"confirmed_three", "eight_observations"}
    ):
        raise ValueError("Unknown initialization snapshot")
    if result["data_schema"] != ("generic" if generic else "moc"):
        raise ValueError("Preset and data schema disagree")
    if result["selection_policy"] not in {"engine", "random_control"}:
        raise ValueError("Unknown selection policy")
    if result["comparison_parent_id"] is not None and not isinstance(
        result["comparison_parent_id"], str
    ):
        raise ValueError("Comparison parent must be a campaign ID or None")
    if not isinstance(result["name"], str) or not result["name"].strip():
        raise ValueError("Campaign name must be nonempty text")
    if result["workflow_mode"] != "live":
        raise ValueError(
            "Sparse MoC data is a live campaign. Offline optimization requires an explicitly supplied oracle or simulator."
        )
    if isinstance(result["batch_size"], bool) or result["batch_size"] != 1:
        raise ValueError(
            "MoC presets support one sequential recommendation, not joint batch BO"
        )
    if not generic and (
        result["bounds"] != [0, 100]
        or result["direction"] != "maximize"
        or result["objective"] != "moc_wt_pct"
        or result["units"] != "wt%"
    ):
        raise ValueError("The MoC schema uses cubic MoC wt%, maximize, bounds 0–100")
    if generic:
        from .generic_import import validate_bounds

        result["bounds"] = validate_bounds(result["bounds"])
        if (
            result["direction"] not in {"maximize", "minimize"}
            or not isinstance(result["objective"], str)
            or not result["objective"]
        ):
            raise ValueError(
                "Generic objective requires a name and maximize/minimize direction"
            )
        if not isinstance(result["units"], str):
            raise ValueError("Objective units must be text")
        result["llm"].update(
            maximize=result["direction"] == "maximize",
            objective_bounds=result["bounds"],
        )
        if result["bounds"] is not None:
            result["llm"]["reference_scale"] = result["bounds"][1] - result["bounds"][0]
    if result["repeat_policy"] not in {
        "source_reset_quality_repeats",
        "exclude_all_historical",
    }:
        raise ValueError("Unknown repeat policy")

    def integer(value, name, minimum=1):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or int(value) != value
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")

    def number(value, name, minimum=0):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
        ):
            raise ValueError(f"{name} must be finite and >= {minimum}")

    budget = result["new_measurement_budget"]
    if budget is not None:
        integer(budget, "Measurement budget", 0)
    integer(result["seed"], "seed", 0)
    integer(
        result["independent_campaign_replicates"], "independent_campaign_replicates"
    )
    if result["measurements_per_candidate"] != 1 or isinstance(
        result["measurements_per_candidate"], bool
    ):
        raise ValueError(
            "MoC sequential campaigns record one measurement per reservation"
        )
    if not isinstance(result["auto_suggest"], bool):
        raise ValueError("auto_suggest must be boolean")
    for group, fields in [
        (
            "llm",
            [
                "uncertainty_scalar",
                "inverse_jitter",
                "xi",
                "ucb_lambda",
                "inverse_multiplier",
            ],
        ),
        ("api", ["request_spacing_s", "base_cooldown_s"]),
    ]:
        for key in fields:
            number(result[group][key], f"{group}.{key}")
    integer(
        result["api"]["maximum_attempts"], "Maximum attempts (including first request)"
    )
    llm = result["llm"]
    result["measurement_definition"] = resolve_definition(
        result["measurement_definition"]
    )
    integer(llm["inverse_proposal_count"], "llm.inverse_proposal_count")
    if llm["inverse_proposal_count"] > 20:
        raise ValueError("Standalone inverse proposal count must be at most 20")
    for field in (
        "n_samples",
        "min_samples",
        "inverse_n",
        "forward_max_tokens",
        "inverse_max_tokens",
        "shortlist_size",
        "fetch_k",
    ):
        integer(llm[field], f"llm.{field}")
    integer(llm["seed"], "llm.seed", 0)
    integer(llm["random_addons"], "llm.random_addons", 0)
    if llm["random_addons"] != 0:
        raise ValueError("The corrected finite-pool retrieval uses no random additions")
    for field in ("forward_temperature", "inverse_temperature"):
        number(llm[field], f"llm.{field}")
        if llm[field] > 2:
            raise ValueError(f"llm.{field} must be <= 2")
    number(llm["mmr_lambda"], "llm.mmr_lambda")
    if llm["mmr_lambda"] > 1:
        raise ValueError("MMR lambda must be within [0,1]")
    number(llm["reference_scale"], "llm.reference_scale", 1e-12)
    for field in (
        "forward_model",
        "inverse_model",
        "embedding_model",
        "selector_embedding_model",
    ):
        if not isinstance(llm[field], str) or not llm[field].strip():
            raise ValueError(f"llm.{field} must be nonempty text")
    for field in ("forward_system_message", "inverse_system_message"):
        if llm[field] is not None and not isinstance(llm[field], str):
            raise ValueError(f"llm.{field} must be text or None")
    if not isinstance(llm["include_phase_context"], bool):
        raise ValueError("include_phase_context must be boolean")
    if not generic and (
        llm["maximize"] is not True or llm["objective_bounds"] != [0, 100]
    ):
        raise ValueError("LLM objective direction and bounds must match the campaign")
    if llm["acquisition"] not in {
        "expected_improvement",
        "probability_of_improvement",
        "upper_confidence_bound",
    }:
        raise ValueError("Unknown LLM acquisition")
    for field in ("target_floor", "target_ceiling", "manual_inverse_target"):
        if llm[field] is not None:
            value = llm[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"llm.{field} must be finite")
            if (
                result["bounds"] is not None
                and not result["bounds"][0] <= value <= result["bounds"][1]
            ):
                raise ValueError(f"llm.{field} must be within physical bounds")
    if (
        llm["target_floor"] is not None
        and llm["target_ceiling"] is not None
        and llm["target_floor"] > llm["target_ceiling"]
    ):
        raise ValueError("Target floor cannot exceed ceiling")
    manual = llm["manual_inverse_target"]
    if manual is not None and (
        (llm["target_floor"] is not None and manual < llm["target_floor"])
        or (llm["target_ceiling"] is not None and manual > llm["target_ceiling"])
    ):
        raise ValueError(
            "Manual inverse target must be within configured target bounds"
        )
    if not 2 <= llm["min_samples"] <= llm["n_samples"] or llm["inverse_n"] != 1:
        raise ValueError(
            "Require at least two accepted predictions and one inverse completion"
        )
    if llm["selector_mode"] not in {"nearest", "all"}:
        raise ValueError("Choose nearest positive count or all observed examples")
    if llm["selector_mode"] == "all":
        llm["selector_k"] = None
    else:
        integer(llm["selector_k"], "llm.selector_k")
    gp = result["structured_gp"]
    if gp["noise_policy"] not in (
        {"reported_or_fixed"} if generic else {"moc_quality"}
    ):
        raise ValueError(
            "Observation noise policy must match the explicit dataset adapter"
        )
    number(gp["default_observation_sigma"], "structured_gp.default_observation_sigma")
    if generic:
        if not isinstance(gp["feature_spec"], list):
            raise ValueError(
                "Generic structured features require an ordered mapping list"
            )
    elif gp["feature_spec"] is not None:
        raise ValueError("MoC structured features are fixed by the versioned preset")
    for field in (
        "retained_draws",
        "predict_thin",
        "ei_after_unique_measured_designs",
        "chunk_size",
    ):
        integer(gp[field], f"structured_gp.{field}")
    integer(gp["burn_in"], "structured_gp.burn_in", 0)
    integer(gp["seed"], "structured_gp.seed", 0)
    integer(gp["quadrature_nodes"], "structured_gp.quadrature_nodes", 16)
    for field in ("proposal_step", "sigma_floor_pp", "logit_delta_pp"):
        number(gp[field], f"structured_gp.{field}", 1e-12)
    for field in ("gof_power", "ei_xi_standardized_logit"):
        number(gp[field], f"structured_gp.{field}")
    if gp["missing_metadata_policy"] not in {"error", "fallback"}:
        raise ValueError("Unknown missing GP metadata policy")
    if not isinstance(gp["metadata_fallbacks"], dict) or set(
        gp["metadata_fallbacks"]
    ) - {"moc_wt_pct_sigma", "gof", "closure_gap_wt_pct"}:
        raise ValueError("Only explicit sigma, GOF and gap fallbacks are supported")
    for field, value in gp["metadata_fallbacks"].items():
        number(value, f"structured_gp.metadata_fallbacks.{field}")
    emb = result["embedding_gp"]
    for field in ("dimensions", "neighbors"):
        integer(emb[field], f"embedding_gp.{field}")
    integer(emb["seed"], "embedding_gp.seed", 0)
    if (
        not isinstance(emb["embedding_model"], str)
        or not emb["embedding_model"].strip()
    ):
        raise ValueError("Embedding model must be nonempty text")
    return result


def preset_catalog():
    """Read-only built-in choices; saved campaigns retain their own version/settings."""
    return [
        {
            "preset": preset,
            "name": name,
            "version": PRESET_VERSIONS[preset],
            "engine": engine,
            "data_schema": "generic" if preset.startswith("generic_") else "moc",
            "description": "Three confirmed seeds; five new physical syntheses per arm"
            if preset.startswith("moc_five_")
            else "Reusable mapped-dataset engine"
            if preset.startswith("generic_")
            else "Source-compatible MoC continuation",
        }
        for preset, (name, engine, _) in PRESETS.items()
    ]


def preview_preset(preset, overrides=None):
    """Resolve a complete fresh configuration without a campaign or model request."""
    config = resolve_config(preset, overrides)
    return {
        "preset": preset,
        "name": config["preset_provenance"]["display_name"],
        "version": config["preset_version"],
        "config": config,
        "provenance": deepcopy(config["preset_provenance"]),
    }


def migrate_legacy_settings(config):
    """Preserve historical meaning without silently turning legacy GPR into features."""
    saved = deepcopy(config)
    return {
        "legacy_settings": saved,
        "engine": {"gpr": "gpr_embeddings", "llm": "llm"}.get(
            saved.get("optimizer"), saved.get("optimizer")
        ),
        "maximum_attempts": saved.get("api_retry_attempts", 8),
        "iteration_limit": None
        if saved.get("iterations_per_trial") == 0
        else saved.get("iterations_per_trial"),
        "iteration_limit_meaning": "legacy observation-count semantics; not reinterpreted as new measurements",
    }
