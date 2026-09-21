"""Corrected crystal LLM method; all provider access is injected and replayable.

No API access happens at construction. The campaign service owns durable cache,
rate-limit/retry policy, reservation and persistence; this engine owns scientific
selection and ID-keyed request/results. It never retries or substitutes engines.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .aqfxns import expected_improvement, probability_of_improvement
from .llm_model import make_dd, scale_distribution


METHOD_VERSION = "moc-corrected-v1"
PROMPT_PATH = Path(__file__).with_name("prompts")
NUMERIC_ONLY = re.compile(r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$")


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def resolve_inverse_target(
    best,
    *,
    maximize=True,
    multiplier=1.2,
    jitter=0.05,
    bounds=None,
    reference_scale=None,
    manual=None,
    floor=None,
    ceiling=None,
    rng=None,
):
    """Resolve aspirations in raw units with an auditable, sign-safe policy."""
    best = float(best)
    if (
        not np.isfinite(best)
        or not np.isfinite(multiplier)
        or not np.isfinite(jitter)
        or jitter < 0
    ):
        raise ValueError("Incumbent, multiplier and nonnegative jitter must be finite")
    lower, upper = (None, None) if bounds is None else bounds
    for value in (lower, upper, floor, ceiling):
        if value is not None and not np.isfinite(value):
            raise ValueError("Target bounds must be finite or None")
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("Physical lower bound exceeds upper bound")
    if lower is not None and best < lower or upper is not None and best > upper:
        raise ValueError("Incumbent lies outside physical bounds")
    effective_lower = (
        max(v for v in (lower, floor) if v is not None)
        if any(v is not None for v in (lower, floor))
        else None
    )
    effective_upper = (
        min(v for v in (upper, ceiling) if v is not None)
        if any(v is not None for v in (upper, ceiling))
        else None
    )
    if (
        effective_lower is not None
        and effective_upper is not None
        and effective_lower > effective_upper
    ):
        raise ValueError("Target floor/ceiling conflicts with physical bounds")
    direction = 1 if maximize else -1
    record = dict(
        policy="improving-direction-v1",
        best=best,
        direction=direction,
        bounds=None if bounds is None else list(bounds),
        floor=floor,
        ceiling=ceiling,
    )
    if manual is not None:
        raw = float(manual)
        if (
            not np.isfinite(raw)
            or effective_lower is not None
            and raw < effective_lower
            or effective_upper is not None
            and raw > effective_upper
        ):
            raise ValueError("Manual inverse target lies outside valid bounds")
        record.update(
            mode="manual",
            multiplier_draw=None,
            improvement_fraction=None,
            reference_scale=None,
        )
    else:
        scale = abs(best)
        if scale == 0:
            if (
                reference_scale is None
                or not np.isfinite(reference_scale)
                or reference_scale <= 0
            ):
                raise ValueError(
                    "Zero incumbent requires a manual target or a positive reference scale"
                )
            scale = float(reference_scale)
        draw = float((rng or np.random.default_rng()).normal(multiplier, jitter))
        fraction = max(0.0, draw - 1.0)
        raw = best + direction * fraction * scale
        record.update(
            mode="automatic",
            multiplier_draw=draw,
            improvement_fraction=fraction,
            nonnegative_improvement_applied=draw < 1,
            reference_scale=scale,
        )
    resolved = raw
    if effective_lower is not None:
        resolved = max(resolved, effective_lower)
    if effective_upper is not None:
        resolved = min(resolved, effective_upper)
    record.update(
        raw_target=raw,
        resolved_target=resolved,
        bounds_applied=resolved != raw,
        saturated=(upper is not None and maximize and best >= upper)
        or (lower is not None and not maximize and best <= lower),
    )
    return record


def crystal_embedding_input(procedure):
    """Do not normalize the original Unicode or whitespace of the body."""
    return "experimental procedure: " + procedure


def normalized_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Embeddings must be a finite two-dimensional matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot use zero vectors for cosine retrieval")
    return values / norms


def retrieve_candidates(
    candidate_ids,
    vectors,
    query_vector,
    *,
    fetch_k=100,
    shortlist_size=16,
    mmr_lambda=0.5,
):
    """Full eligible matrix → nearest 100 cosine → MMR 16, stable ID ties."""
    if not 0 <= mmr_lambda <= 1 or fetch_k < 1 or shortlist_size < 1:
        raise ValueError("Invalid retrieval counts or MMR lambda")
    matrix = normalized_vectors(vectors)
    query = normalized_vectors([query_vector])[0]
    if len(matrix) != len(candidate_ids) or matrix.shape[1] != len(query):
        raise ValueError("Candidate IDs, query and embedding dimensions do not align")
    similarity = matrix @ query
    nearest = sorted(
        range(len(candidate_ids)),
        key=lambda i: (-float(similarity[i]), candidate_ids[i]),
    )[:fetch_k]
    selected = []
    while nearest and len(selected) < shortlist_size:
        scores = {
            i: float(similarity[i])
            if not selected
            else float(
                mmr_lambda * similarity[i]
                - (1 - mmr_lambda) * np.max(matrix[selected] @ matrix[i])
            )
            for i in nearest
        }
        chosen = min(nearest, key=lambda i: (-scores[i], candidate_ids[i]))
        selected.append(chosen)
        nearest.remove(chosen)
    return [
        dict(
            candidate_id=candidate_ids[i],
            pool_index=i,
            rank=rank,
            cosine_similarity=float(similarity[i]),
        )
        for rank, i in enumerate(selected)
    ]


def _procedure(record):
    for key in ("procedure", "procedure_text", "original_procedure_text"):
        if key in record:
            return record[key]
    raise ValueError("Candidate or observation lacks procedure text")


def _value(record):
    return float(record["value"] if "value" in record else record["moc_wt_pct"])


def _context(record):
    if record.get("phase_context") is not None:
        return str(record["phase_context"])
    keys = (
        "moc_wt_pct_sigma",
        "mo_wt_pct",
        "mo_wt_pct_sigma",
        "mo2c_wt_pct",
        "mo2c_wt_pct_sigma",
        "moo2_wt_pct",
        "moo2_wt_pct_sigma",
        "gof",
        "closure_gap_wt_pct",
        "closure_gap_origin",
    )
    known = {key: record[key] for key in keys if record.get(key) is not None}
    return json.dumps(known, ensure_ascii=False) if known else "not reported"


def managed_system_message(
    kind, *, prompt_style="moc", objective_name="objective", objective_units=""
):
    if kind not in {"forward", "inverse"}:
        raise ValueError("Unknown prompt kind")
    if prompt_style == "moc":
        return (PROMPT_PATH / f"moc_{kind}_v1.txt").read_text(encoding="utf-8").strip()
    if prompt_style != "generic":
        raise ValueError("Unknown prompt style")
    label = objective_name + (f" ({objective_units})" if objective_units else "")
    if kind == "forward":
        return f"Use the observed examples to predict the measured objective {label} for the candidate experimental procedure. Return exactly one finite numeric value in the original objective units, without explanation or extra text. Use only the provided observations; unmeasured candidates have no outcome labels."
    return f"Use the observed examples to propose one experimental procedure targeting the requested {label}. Follow the synthesis format and controllable variables in the examples. Return only one procedure as a retrieval query for the finite candidate pool; do not claim that its outcome has been measured."


def render_messages(
    kind,
    observations,
    query,
    system_message=None,
    include_phase_context=True,
    *,
    prompt_style="moc",
    objective_name="objective",
    objective_units="",
):
    """Return exactly one system and one user message; never render candidate labels."""
    if kind not in {"forward", "inverse"}:
        raise ValueError("Unknown prompt kind")
    system = (
        managed_system_message(
            kind,
            prompt_style=prompt_style,
            objective_name=objective_name,
            objective_units=objective_units,
        )
        if system_message is None
        else system_message
    )
    label = (
        "cubic MoC weight fraction (%)"
        if prompt_style == "moc"
        else objective_name + (f" ({objective_units})" if objective_units else "")
    )
    blocks = []
    for row in observations:
        context = (
            f"Additional observed phase context: {_context(row)}\n"
            if include_phase_context
            else ""
        )
        if kind == "forward":
            blocks.append(
                f"Observed example {row['observation_id']}\nSynthesis procedure: {_procedure(row)}\n"
                f"Measured {label}: {_value(row):g}\n{context}"
            )
        else:
            blocks.append(
                f"Observed {label}: {_value(row):g}\n{context}"
                f"Associated synthesis procedure: {_procedure(row)}\n"
            )
    if kind == "forward":
        blocks.append(f"Candidate synthesis procedure: {query}\nPredict {label}:")
    else:
        blocks.append(
            f"Target {label}: {float(query):g}\nPropose a synthesis procedure to use as a retrieval query:"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(blocks)},
    ]


def build_chat_request(
    config, role, observations=(), query=None, *, count=None, messages=None
):
    """Build the exact provider payload without constructing or calling a client."""
    if role not in {"forward", "inverse"}:
        raise ValueError("Unknown prompt kind")
    n = config["n_samples"] if role == "forward" else 1
    if count is not None:
        if (
            role != "inverse"
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 20
        ):
            raise ValueError(
                "Standalone inverse proposal count must be an integer from 1 to 20"
            )
        n = count
    if messages is None:
        messages = render_messages(
            role,
            observations,
            query,
            config.get(f"{role}_system_message"),
            config.get("include_phase_context", True),
            **{
                key: config.get(key, default)
                for key, default in (
                    ("prompt_style", "moc"),
                    ("objective_name", "objective"),
                    ("objective_units", ""),
                )
            },
        )
    return dict(
        model=config[f"{role}_model"],
        messages=deepcopy(messages),
        temperature=config[f"{role}_temperature"],
        max_tokens=config[f"{role}_max_tokens"],
        n=n,
    )


def prompt_provenance(config):
    result = {}
    for role in ("forward", "inverse"):
        message = build_chat_request(
            config, role, query=0 if role == "inverse" else ""
        )["messages"][0]["content"]
        result[role] = dict(
            origin="managed"
            if config.get(f"{role}_system_message") is None
            else "custom",
            version=METHOD_VERSION,
            sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        )
    return result


def _active_observations(observations):
    rows = [
        dict(row)
        for row in observations
        if row.get("training_included", True)
        and row.get("record_status", "measured") == "measured"
    ]
    ids = [str(row["candidate_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "Resolve refinements/replicates to one active demonstration per design"
        )
    return rows


def select_examples(config, observations, query_vector=None, observation_vectors=None):
    """Procedure-only selection shared by execution and previews, with no I/O."""
    if config.get("selector_mode") == "all" or config["selector_k"] is None:
        return list(observations)
    if query_vector is None or observation_vectors is None:
        raise ValueError(
            "Nearest-example selection requires cached procedure embeddings; preview never generates embeddings"
        )
    matrix = normalized_vectors(observation_vectors)
    vector = normalized_vectors([query_vector])[0]
    if len(matrix) != len(observations) or matrix.shape[1] != len(vector):
        raise ValueError(
            "Observation IDs, query and selector embedding dimensions do not align"
        )
    rankings = sorted(
        range(len(observations)),
        key=lambda i: (-float(matrix[i] @ vector), observations[i]["observation_id"]),
    )
    return [observations[i] for i in rankings[: config["selector_k"]]]


def _vector_lookup(rows, vectors, key):
    if vectors is None:
        return {}
    if isinstance(vectors, dict):
        return {str(k): normalized_vectors([v])[0] for k, v in vectors.items()}
    matrix = normalized_vectors(vectors)
    if len(matrix) != len(rows):
        raise ValueError("Cached embedding matrix does not align with record IDs")
    return {str(row[key]): matrix[i] for i, row in enumerate(rows)}


def _cached_selector_vectors(
    config,
    candidates,
    observations,
    candidate_vectors=None,
    observation_vectors=None,
    selector_candidate_vectors=None,
):
    """Resolve only supplied vectors; absent cache entries remain absent."""
    supplied = (
        candidate_vectors
        if config["selector_embedding_model"] == config["embedding_model"]
        else selector_candidate_vectors
    )
    by_candidate = _vector_lookup(candidates, supplied, "candidate_id")
    by_observation = _vector_lookup(observations, observation_vectors, "observation_id")
    rows = [
        by_observation.get(
            str(row["observation_id"]), by_candidate.get(str(row["candidate_id"]))
        )
        for row in observations
    ]
    matrix = None if any(row is None for row in rows) else np.asarray(rows)
    return by_candidate, matrix


def _target(config, observations, rng, target=None):
    if not observations:
        raise ValueError(
            "Inverse prompting requires at least one active measured observation"
        )
    best_row = (max if config["maximize"] else min)(observations, key=_value)
    aspiration = resolve_inverse_target(
        _value(best_row),
        maximize=config["maximize"],
        multiplier=config["inverse_multiplier"],
        jitter=config["inverse_jitter"],
        bounds=config["objective_bounds"],
        reference_scale=config["reference_scale"],
        manual=config.get("manual_inverse_target") if target is None else target,
        floor=config.get("target_floor"),
        ceiling=config.get("target_ceiling"),
        rng=rng,
    )
    return best_row, aspiration


def preview_request(
    config,
    candidates=(),
    observations=(),
    *,
    role="forward",
    candidate_id=None,
    target=None,
    candidate_vectors=None,
    observation_vectors=None,
    selector_candidate_vectors=None,
    recorded_result=None,
    rng_state=None,
    count=None,
):
    """Preview a real payload or explain unresolved selection; never make provider calls.

    A recorded result is authoritative even when current settings differ. Current
    automatic targets use a cloned generator, leaving the campaign RNG untouched.
    Partial vector mappings are allowed when they include the requested candidate
    and all active observed procedures in the selector model's representation.
    """
    if role not in {"forward", "inverse"}:
        raise ValueError("Unknown prompt kind")
    result = dict(
        role=role,
        candidate_id=candidate_id,
        source="recorded" if recorded_result is not None else "current",
        status="unresolved",
        request=None,
        example_ids=None,
    )
    if recorded_result is not None:
        key = (
            candidate_id
            if candidate_id is not None
            else recorded_result.get("selected_candidate_id")
        )
        logs = [
            row
            for row in recorded_result.get("request_log", [])
            if row.get("role") == role
            and (role == "inverse" or row.get("candidate_id") == key)
        ]
        if not logs:
            result["reason"] = "No recorded request for the selected role and candidate"
            return result
        log = logs[-1]
        request = deepcopy(log["request"])
        result.update(
            status="exact",
            candidate_id=log.get("candidate_id"),
            request=request,
            request_sha256=fingerprint(request),
            target=deepcopy(recorded_result.get("target")),
            example_ids=deepcopy(
                recorded_result.get("inverse_example_ids", [])
                if role == "inverse"
                else recorded_result.get("predictions", {})
                .get(key, {})
                .get("example_ids", [])
            ),
            prompt_provenance=deepcopy(recorded_result.get("prompt_provenance", {})),
            recorded_request_status=log.get("status"),
            example_selection=dict(resolved=True, source="recorded"),
        )
        return result
    engine = LLMEngine(config, rng_state=rng_state)
    c = engine.config
    result["prompt_provenance"] = prompt_provenance(c)
    # Static request fields remain useful before a candidate or target can resolve.
    template = build_chat_request(
        c,
        role,
        query=0 if role == "inverse" else "[candidate not selected]",
        count=count,
    )
    result.update(
        request_parameters={
            key: value for key, value in template.items() if key != "messages"
        },
        system_message=template["messages"][0]["content"],
        query_message=None,
        example_selection=dict(
            resolved=False, mode=c["selector_mode"], count=c["selector_k"]
        ),
    )
    observations = _active_observations(observations)
    candidates = list(candidates)
    # Show role, system, suffix and effective settings even when cache is missing.
    try:
        best_row, aspiration = _target(c, observations, engine.rng, target)
        result["target"] = aspiration
        if role == "inverse":
            query = aspiration["resolved_target"]
        else:
            matches = [
                row
                for row in candidates
                if str(row["candidate_id"]) == str(candidate_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Select one candidate to preview its forward request; the next BO shortlist depends on the inverse response and retrieval"
                )
            query = _procedure(matches[0])
        template = build_chat_request(c, role, query=query, count=count)
        result.update(
            request_parameters={
                key: value for key, value in template.items() if key != "messages"
            },
            system_message=template["messages"][0]["content"],
            query_message=template["messages"][1]["content"],
            example_selection=dict(
                resolved=False, mode=c["selector_mode"], count=c["selector_k"]
            ),
        )
        by_candidate, matrix = (
            ({}, None)
            if c["selector_mode"] == "all" or c["selector_k"] is None
            else _cached_selector_vectors(
                c,
                candidates,
                observations,
                candidate_vectors,
                observation_vectors,
                selector_candidate_vectors,
            )
        )
        vector = (
            (None if matrix is None else matrix[observations.index(best_row)])
            if role == "inverse"
            else by_candidate.get(str(candidate_id))
        )
        selected = select_examples(c, observations, vector, matrix)
        request = build_chat_request(c, role, selected, query, count=count)
        result.update(
            status="exact",
            request=request,
            request_sha256=fingerprint(request),
            example_ids=[row["observation_id"] for row in selected],
            example_selection=dict(
                resolved=True,
                mode=c["selector_mode"],
                source="supplied_cache"
                if c["selector_mode"] != "all"
                else "all_observations",
            ),
        )
    except ValueError as error:
        result["reason"] = str(error)
    return result


def score_responses(candidate_id, raw_responses, best, config, rank=0):
    """Pure replay function. Invalid completions retain reasons and ownership."""
    values, rejected = [], []
    bounds = config.get("objective_bounds")
    for index, raw in enumerate(raw_responses):
        reason = None
        if not isinstance(raw, str) or not NUMERIC_ONLY.fullmatch(raw):
            reason = "not a numeric-only response"
        else:
            value = float(raw)
            if not np.isfinite(value):
                reason = "nonfinite response"
            elif bounds is not None and not bounds[0] <= value <= bounds[1]:
                reason = "response outside physical bounds"
            else:
                values.append(value)
        if reason:
            rejected.append(dict(index=index, response=raw, reason=reason))
    record = dict(
        candidate_id=candidate_id,
        rank=rank,
        raw_responses=list(raw_responses),
        accepted_values=values,
        rejected_responses=rejected,
        requested_samples=config["n_samples"],
        accepted_samples=len(values),
        partial_samples=len(values) < config["n_samples"],
        acquisition=None,
        mean=None,
        std=None,
        status="insufficient_samples",
    )
    if len(values) < config["min_samples"]:
        return record
    dist = make_dd(values, np.full(len(values), 1 / len(values)))
    dist = scale_distribution(dist, config["uncertainty_scalar"], bounds)
    acquisition = config.get("acquisition", "expected_improvement")
    maximize = config.get("maximize", True)
    if acquisition == "expected_improvement":
        score = expected_improvement(dist, best, config.get("xi", 0), maximize)
    elif acquisition == "probability_of_improvement":
        score = probability_of_improvement(dist, best, config.get("xi", 0), maximize)
    elif acquisition == "upper_confidence_bound":
        score = (1 if maximize else -1) * dist.mean() + config.get(
            "ucb_lambda", 0.5
        ) * dist.std()
    else:
        raise ValueError(f"Unsupported empirical acquisition: {acquisition}")
    record.update(
        status="scored",
        acquisition=float(score),
        mean=float(dist.mean()),
        std=float(dist.std()),
        distribution=dict(
            family="empirical",
            values=dist.values.tolist(),
            probabilities=dist.probs.tolist(),
        ),
        support_bounding_policy="clip_to_physical_bounds"
        if bounds is not None and config["uncertainty_scalar"] > 1
        else "none",
        agreement_label="agreement among accepted model responses"
        if dist.std() == 0
        else None,
    )
    return record


class LLMEngine:
    def __init__(self, config=None, client=None, embedding_client=None, rng_state=None):
        from .campaign_config import resolve_config

        supplied = asdict(config) if is_dataclass(config) else (config or {})
        llm = deepcopy(supplied.get("llm", supplied))
        runtime = {
            key: llm.pop(key, default)
            for key, default in (
                (
                    "prompt_style",
                    "generic" if supplied.get("data_schema") == "generic" else "moc",
                ),
                ("objective_name", supplied.get("objective", "objective")),
                ("objective_units", supplied.get("units", "")),
            )
        }
        overrides = {"llm": llm}
        preset = "generic_llm" if runtime["prompt_style"] == "generic" else "moc_llm"
        if preset == "generic_llm":
            overrides.update(
                objective=runtime["objective_name"],
                units=runtime["objective_units"],
                bounds=llm.get("objective_bounds"),
                direction="maximize" if llm.get("maximize", True) else "minimize",
            )
        self.config = {**resolve_config(preset, overrides)["llm"], **runtime}
        self.client = client
        self.embedding_client = embedding_client
        self.rng = np.random.default_rng(self.config["seed"])
        if rng_state is not None:
            self.rng.bit_generator.state = deepcopy(rng_state)
        self.request_log = []
        self._validate()

    def _validate(self):
        c = self.config
        if (
            not 2 <= c["min_samples"] <= c["n_samples"]
            or int(c["n_samples"]) != c["n_samples"]
        ):
            raise ValueError(
                "Require at least two accepted samples per ranked candidate"
            )
        if c.get("inverse_n", 1) != 1:
            raise ValueError("BO requires exactly one inverse completion")
        if not np.isfinite(c["uncertainty_scalar"]) or c["uncertainty_scalar"] < 0:
            raise ValueError("Uncertainty scalar must be finite and nonnegative")
        if c["selector_k"] is not None and c["selector_k"] < 1:
            raise ValueError(
                "Use selector_mode='all' for all examples; zero is not zero-shot BO"
            )
        for role in ("forward", "inverse"):
            if c[f"{role}_model"].startswith(("gpt-5", "o1", "o3", "o4")):
                raise ValueError(
                    "Configured model requires a different sampling adapter; model substitution is disabled"
                )

    def _chat(self, role, messages, candidate_id=None, count=None):
        if self.client is None:
            raise ValueError(
                "A provider client must be explicitly supplied for a live suggestion"
            )
        request = build_chat_request(self.config, role, messages=messages, count=count)
        record = dict(
            role=role,
            candidate_id=candidate_id,
            request=deepcopy(request),
            request_sha256=fingerprint(request),
            status="requested",
        )
        self.request_log.append(record)
        try:
            response = self.client.chat.completions.create(**request)
            data = response if isinstance(response, dict) else response.model_dump()
            raw = [
                choice.get("message", {}).get("content", "")
                for choice in data.get("choices", [])
            ]
            record.update(
                status="completed",
                raw_responses=raw,
                returned_model=data.get("model"),
                response_id=data.get("id"),
                usage=data.get("usage"),
            )
            return raw
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise

    def _embed(self, procedures, model):
        provider = self.embedding_client or self.client
        if provider is None:
            raise ValueError(
                "Validated vectors or an explicitly supplied embedding provider are required"
            )
        inputs = [crystal_embedding_input(p) for p in procedures]
        record = dict(
            role="embedding",
            model=model,
            input_texts=inputs,
            input_sha256=[
                hashlib.sha256(text.encode("utf-8")).hexdigest() for text in inputs
            ],
            status="requested",
        )
        self.request_log.append(record)
        response = provider.embeddings.create(model=model, input=inputs)
        data = response if isinstance(response, dict) else response.model_dump()
        if data.get("model") and data["model"] != model:
            raise ValueError("Embedding provider returned a different model")
        rows = data["data"]
        indexed = {row["index"]: row["embedding"] for row in rows}
        if len(indexed) != len(procedures) or len(indexed) != len(rows):
            raise ValueError("Embedding response indices are missing or duplicated")
        matrix = normalized_vectors([indexed[i] for i in range(len(procedures))])
        record.update(
            status="completed",
            returned_model=data.get("model"),
            dimensions=int(matrix.shape[1]),
            usage=data.get("usage"),
        )
        return matrix

    def propose_inverse(
        self,
        candidates,
        observations,
        *,
        count=1,
        target=None,
        candidate_vectors=None,
        observation_vectors=None,
        cancel=None,
    ):
        """Generate standalone procedure queries; never score or reserve candidates."""
        self.request_log = []
        c = self.config
        result = dict(
            method_version=METHOD_VERSION,
            status="failed",
            procedures=[],
            requested_count=count,
            returned_count=0,
            request_log=self.request_log,
            config=deepcopy(c),
            prompt_provenance=prompt_provenance(c),
            rng_state_before=deepcopy(self.rng.bit_generator.state),
        )
        try:
            if cancel and cancel():
                raise InterruptedError("Inverse proposal cancelled")
            # Validate count before drawing a target or making any provider call.
            build_chat_request(c, "inverse", query=0, count=count)
            observations = _active_observations(observations)
            best_row, aspiration = _target(c, observations, self.rng, target)
            result["target"] = aspiration
            if c["selector_mode"] == "all" or c["selector_k"] is None:
                selected = observations
            else:
                _, matrix = _cached_selector_vectors(
                    c,
                    list(candidates),
                    observations,
                    candidate_vectors,
                    observation_vectors,
                )
                if matrix is None:
                    matrix = self._embed(
                        [_procedure(row) for row in observations],
                        c["selector_embedding_model"],
                    )
                selected = select_examples(
                    c, observations, matrix[observations.index(best_row)], matrix
                )
            result["inverse_example_ids"] = [row["observation_id"] for row in selected]
            request = build_chat_request(
                c, "inverse", selected, aspiration["resolved_target"], count=count
            )
            if cancel and cancel():
                raise InterruptedError("Inverse proposal cancelled")
            raw = self._chat("inverse", request["messages"], count=count)
            result["raw_responses"] = raw
            result["returned_count"] = len(raw)
            result["procedures"] = [
                value for value in raw if isinstance(value, str) and value.strip()
            ]
            if len(raw) != count or len(result["procedures"]) != count:
                raise ValueError(
                    "Inverse request did not return the requested number of nonempty procedures"
                )
            if cancel and cancel():
                raise InterruptedError("Inverse proposal cancelled")
            result["status"] = "proposed"
        except InterruptedError as error:
            result.update(status="cancelled", reason=str(error))
        except Exception as error:
            result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        result["rng_state_after"] = deepcopy(self.rng.bit_generator.state)
        return result

    def suggest(
        self,
        candidates,
        observations,
        excluded_ids=(),
        target=None,
        candidate_vectors=None,
        query_embedder=None,
        cancel=None,
    ):
        self.request_log = []
        c = self.config
        candidates = [dict(row) for row in candidates]
        ids = [str(row["candidate_id"]) for row in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate IDs must be unique")
        observations = _active_observations(observations)
        observed_ids = [row["candidate_id"] for row in observations]
        if len(set(observed_ids)) != len(observed_ids):
            raise ValueError(
                "Resolve refinements/replicates to one active demonstration per design"
            )
        blocked = set(excluded_ids) | set(observed_ids)
        eligible_indices = [i for i, key in enumerate(ids) if key not in blocked]
        result = dict(
            method_version=METHOD_VERSION,
            status="failed",
            selected_candidate_id=None,
            predictions={},
            request_log=self.request_log,
            eligible_count=len(eligible_indices),
            config=deepcopy(c),
            rng_state_before=deepcopy(self.rng.bit_generator.state),
        )
        result["prompt_provenance"] = prompt_provenance(c)
        if not eligible_indices:
            result.update(status="exhausted", reason="No eligible candidates")
            return result
        if len(set(observed_ids)) < 2:
            chosen = sorted(ids[i] for i in eligible_indices)[
                int(self.rng.integers(len(eligible_indices)))
            ]
            result.update(
                status="initial_design",
                selected_candidate_id=chosen,
                reason="Fewer than two distinct active observed designs; seeded random initial design",
                rng_state_after=deepcopy(self.rng.bit_generator.state),
            )
            return result

        def check_cancel():
            if cancel and cancel():
                raise InterruptedError("Suggestion cancelled")

        try:
            check_cancel()
            best_row, aspiration = _target(c, observations, self.rng, target)
            best = _value(best_row)
            result["target"] = aspiration
            matrix = (
                self._embed(
                    [_procedure(row) for row in candidates], c["embedding_model"]
                )
                if candidate_vectors is None
                else normalized_vectors(candidate_vectors)
            )
            if len(matrix) != len(candidates):
                raise ValueError("Candidate matrix does not align with candidate IDs")
            positions = {key: i for i, key in enumerate(ids)}
            if c["selector_embedding_model"] == c["embedding_model"] and all(
                key in positions for key in observed_ids
            ):
                observed_vectors = np.array(
                    [matrix[positions[key]] for key in observed_ids]
                )
            else:
                observed_vectors = self._embed(
                    [_procedure(row) for row in observations],
                    c["selector_embedding_model"],
                )

            def examples(vector):
                return select_examples(c, observations, vector, observed_vectors)

            best_index = observations.index(best_row)
            inv_examples = examples(observed_vectors[best_index])
            inv_messages = build_chat_request(
                c, "inverse", inv_examples, aspiration["resolved_target"]
            )["messages"]
            inverse = self._chat("inverse", inv_messages)
            if (
                len(inverse) != 1
                or not isinstance(inverse[0], str)
                or not inverse[0].strip()
            ):
                raise ValueError(
                    "Inverse request did not return exactly one nonempty procedure"
                )
            result["inverse_procedure"] = inverse[0]
            result["inverse_example_ids"] = [
                row["observation_id"] for row in inv_examples
            ]
            check_cancel()
            query_vector = (
                query_embedder(inverse[0])
                if query_embedder
                else self._embed(inverse, c["embedding_model"])[0]
            )
            eligible_ids = [ids[i] for i in eligible_indices]
            retrieval = retrieve_candidates(
                eligible_ids,
                matrix[eligible_indices],
                query_vector,
                fetch_k=c["fetch_k"],
                shortlist_size=c["shortlist_size"],
                mmr_lambda=c["mmr_lambda"],
            )
            if c.get("random_addons", 0):
                remaining = [
                    key
                    for key in eligible_ids
                    if key not in {row["candidate_id"] for row in retrieval}
                ]
                for key in self.rng.choice(
                    remaining, min(c["random_addons"], len(remaining)), replace=False
                ):
                    retrieval.append(
                        dict(
                            candidate_id=str(key),
                            rank=len(retrieval),
                            source="random_addon",
                        )
                    )
            result["retrieval"] = dict(
                scope="full_eligible_pool",
                metric="float32-l2-normalized-inner-product",
                input_version="crystal-prefixed-v1",
                prefilter_count=min(c["fetch_k"], len(eligible_ids)),
                records=retrieval,
            )
            for retrieval_row in retrieval:
                check_cancel()
                key = retrieval_row["candidate_id"]
                candidate = candidates[positions[key]]
                vector = (
                    matrix[positions[key]]
                    if c["selector_embedding_model"] == c["embedding_model"]
                    else self._embed(
                        [_procedure(candidate)], c["selector_embedding_model"]
                    )[0]
                )
                selected_examples = examples(vector)
                messages = build_chat_request(
                    c, "forward", selected_examples, _procedure(candidate)
                )["messages"]
                try:
                    raw = self._chat("forward", messages, key)
                    record = score_responses(key, raw, best, c, retrieval_row["rank"])
                except InterruptedError:
                    raise
                except Exception as error:
                    record = dict(
                        candidate_id=key,
                        rank=retrieval_row["rank"],
                        status="request_failed",
                        acquisition=None,
                        error=f"{type(error).__name__}: {error}",
                    )
                record["example_ids"] = [
                    row["observation_id"] for row in selected_examples
                ]
                record["prediction_fingerprint"] = fingerprint(
                    dict(
                        candidate_id=key,
                        procedure=_procedure(candidate),
                        observations=selected_examples,
                        messages=messages,
                        config=c,
                        method=METHOD_VERSION,
                    )
                )
                result["predictions"][key] = record
            scored = sorted(
                (
                    row
                    for row in result["predictions"].values()
                    if row["status"] == "scored"
                ),
                key=lambda row: (-row["acquisition"], row["rank"], row["candidate_id"]),
            )
            if scored:
                result.update(
                    status="suggested", selected_candidate_id=scored[0]["candidate_id"]
                )
            else:
                result[
                    "reason"
                ] = "No candidate has enough valid samples; no fallback was used"
        except InterruptedError as error:
            result.update(status="cancelled", reason=str(error))
        except Exception as error:
            result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        result["rng_state_after"] = deepcopy(self.rng.bit_generator.state)
        return result


def replay_step(result):
    """Recompute a recorded step offline; no clients, embeddings or random draws."""
    replay = deepcopy(result)
    if "target" not in result:
        replay["replay_verified"] = True
        return replay
    best = result["target"]["best"]
    for key, original in result["predictions"].items():
        if "raw_responses" not in original:
            continue
        scored = score_responses(
            key, original["raw_responses"], best, result["config"], original["rank"]
        )
        replay["predictions"][key].update(scored)
    ranked = sorted(
        (row for row in replay["predictions"].values() if row["status"] == "scored"),
        key=lambda row: (-row["acquisition"], row["rank"], row["candidate_id"]),
    )
    selected = ranked[0]["candidate_id"] if ranked else None
    replay["replayed_ranking"] = [
        dict(candidate_id=row["candidate_id"], acquisition=row["acquisition"])
        for row in ranked
    ]
    replay["replay_verified"] = selected == result.get("selected_candidate_id")
    replay["replayed_selected_candidate_id"] = selected
    return replay
