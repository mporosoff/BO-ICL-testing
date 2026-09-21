"""Explicitly synthetic offline provider and walkthrough fixtures, never lab data."""
from types import SimpleNamespace
import hashlib
import numpy as np


class DemoChat:
    def __init__(self, procedure, bounds=(0, 100)):
        self.procedure = procedure
        self.bounds = bounds or (-100, 100)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        if request["n"] == 1:
            outputs = [self.procedure]
        else:
            # Stable canned responses; they are NOT a fitted model or an oracle.
            n = int(
                hashlib.sha256(request["messages"][-1]["content"].encode()).hexdigest()[
                    :6
                ],
                16,
            )
            lower, upper = self.bounds
            span = upper - lower
            base = lower + span * (0.55 + (n % 32) / 100)
            offsets = [-0.04, 0, 0.04, 0, 0]
            outputs = [
                str(min(upper, base + span * offsets[index % len(offsets)]))
                for index in range(request["n"])
            ]
        return {
            "id": "synthetic-demo-request",
            "model": "synthetic-demo-no-provider",
            "choices": [{"message": {"content": value}} for value in outputs],
        }


def demo_runner(snapshot, eligible, cancel, progress):
    from .campaign import CampaignService, run_engine
    from .structured_gp import transform_features
    from .llm_engine import LLMEngine

    if snapshot["config"]["engine"] == "gpr_features":
        result = run_engine(snapshot, eligible, None, cancel, progress)
        result["synthetic_demo"] = True
        return result
    candidates = snapshot["candidates"]
    lookup = {r["candidate_id"]: r for r in candidates}
    generic = snapshot["config"].get("data_schema") == "generic"
    if generic:
        spec = snapshot["config"]["structured_gp"]["feature_spec"]
        if spec:
            from .generic_import import FeatureTransform

            features = FeatureTransform(spec).transform(candidates)
        else:
            features = (
                np.array(
                    [
                        list(hashlib.sha256(r["procedure"].encode()).digest()[:4])
                        for r in candidates
                    ],
                    dtype=float,
                )
                / 255
            )
    else:
        features = transform_features(candidates)
    # Distinct demonstration vector space; never written to production embedding cache.
    vectors = np.column_stack((np.ones(len(features)), features))
    if snapshot["config"]["engine"] == "gpr_embeddings":
        # Browser walkthrough mock only. Real model numerical behavior is separately tested.
        chosen = eligible[0]
        lower, upper = snapshot["config"]["bounds"] or (-100, 100)
        span = upper - lower
        return {
            "status": "suggested",
            "candidate_id": chosen["candidate_id"],
            "score": 1.0,
            "acquisition_units": "synthetic EI",
            "selection_reason": "SYNTHETIC browser fixture for embedding-GP reservation workflow",
            "prediction": {
                "mean": lower + 0.7 * span,
                "std": 0.05 * span,
                "lower95": lower + 0.6 * span,
                "upper95": lower + 0.8 * span,
                "uncertainty_type": "mock latent interval",
            },
            "synthetic_demo": True,
        }
    progress("Synthetic inverse query and forward completions; no network")
    observations = [
        {
            **r,
            "value": r[CampaignService.objective_field(snapshot)],
            "procedure": lookup[r["candidate_id"]]["procedure"],
        }
        for r in CampaignService.active(snapshot)
    ]
    selected = eligible[0]
    config = snapshot["config"]
    engine = LLMEngine(
        {
            **config["llm"],
            "seed": snapshot["step_seed"],
            "objective_name": config["objective"],
            "objective_units": config["units"],
            "prompt_style": "generic" if generic else "moc",
        },
        client=DemoChat(selected["procedure"], config["bounds"]),
    )
    result = engine.suggest(
        candidates,
        observations,
        excluded_ids=set(lookup) - {r["candidate_id"] for r in eligible},
        candidate_vectors=vectors,
        query_embedder=lambda _: vectors[
            next(
                i
                for i, c in enumerate(candidates)
                if c["candidate_id"] == selected["candidate_id"]
            )
        ],
        cancel=cancel.is_set,
    )
    result["synthetic_demo"] = True
    result[
        "embedding_provenance"
    ] = f"synthetic {vectors.shape[1]}-dimensional vectors; not provider embeddings"
    return result
