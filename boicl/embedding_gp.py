"""Separate finite-corpus embedding GP; learned homoskedastic observation noise.

Projection is outcome-free and fixed for the complete candidate corpus. The
BoTorch public posterior unstandardizes exactly once and uses its own fitted
likelihood. Acquisitions use latent-function uncertainty in raw objective units.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import ndtr
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph

from .embedding_cache import sha256_text

_PROJECTION_RANDOM_LOCK = threading.RLock()


def fit_projection(vectors, dimensions=32, neighbors=5, seed=616):
    """Return Isomap model, fixed coordinates, and visible small-space adjustments."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Projection requires a finite nonempty embedding matrix")
    if dimensions < 1 or neighbors < 1:
        raise ValueError("Projection dimensions and neighbors must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Projection seed must be a nonnegative integer")
    seed = int(seed)
    arpack_seed = seed % (2**32)
    n = len(values)
    diagnostics = {
        "method": "Isomap",
        "requested_dimensions": int(dimensions),
        "requested_neighbors": int(neighbors),
        "corpus_rows": n,
        "seed": seed,
        "arpack_seed": arpack_seed,
        "adjustments": [],
    }
    if n == 1 or np.all(values == values[0]):
        diagnostics.update(dimensions=1, neighbors=0, connected_components=1)
        diagnostics["adjustments"].append(
            "Single or coincident vectors: fixed zero coordinate; no manifold fit"
        )
        return None, np.zeros((n, 1)), diagnostics
    effective_dimensions = min(int(dimensions), n - 1, values.shape[1])
    effective_neighbors = min(int(neighbors), n - 1)
    if effective_dimensions != dimensions:
        diagnostics["adjustments"].append(
            f"Dimensions capped to {effective_dimensions} for corpus size"
        )
    if effective_neighbors != neighbors:
        diagnostics["adjustments"].append(
            f"Neighbors capped to {effective_neighbors} for corpus size"
        )
    while True:
        graph = kneighbors_graph(values, effective_neighbors, mode="connectivity")
        count = connected_components(
            graph.maximum(graph.T), directed=False, return_labels=False
        )
        if count == 1:
            break
        effective_neighbors = min(
            n - 1, max(effective_neighbors + 1, effective_neighbors * 2)
        )
    if effective_neighbors != min(int(neighbors), n - 1):
        diagnostics["adjustments"].append(
            f"Disconnected neighbor graph: increased neighbors deterministically to {effective_neighbors}"
        )
    model = Isomap(
        n_components=effective_dimensions,
        n_neighbors=effective_neighbors,
        eigen_solver="arpack",
    )
    # Isomap/ARPACK initialization must not make a saved corpus projection random.
    # Isomap does not expose KernelPCA's random_state. Serialize this short-lived
    # use of NumPy's legacy RNG so concurrent campaign projections cannot race.
    with _PROJECTION_RANDOM_LOCK:
        random_state = np.random.get_state()
        try:
            np.random.seed(arpack_seed)
            coordinates = model.fit_transform(values)
        finally:
            np.random.set_state(random_state)
    if not np.isfinite(coordinates).all():
        raise ValueError("Isomap returned nonfinite coordinates")
    diagnostics.update(
        dimensions=effective_dimensions,
        neighbors=effective_neighbors,
        connected_components=int(count),
    )
    return model, coordinates, diagnostics


class EmbeddingGPEngine:
    def __init__(self, candidates, vectors, config=None, projection_cache=None):
        self.candidates = list(candidates)
        self.config = dict(config or {})
        self.candidate_ids = [str(c["candidate_id"]) for c in self.candidates]
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Embedding GP corpus contains duplicate candidate IDs")
        self._indices = {cid: index for index, cid in enumerate(self.candidate_ids)}
        vectors = np.asarray(vectors, dtype=float)
        if (
            vectors.ndim != 2
            or len(vectors) != len(self.candidates)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError(
                "Embedding GP requires one finite vector for every full-corpus candidate"
            )
        metadata = {
            "version": "isomap-fixed-corpus-v2",
            "embedding_model": self.config.get(
                "embedding_model", "text-embedding-ada-002"
            ),
            "embedding_dimensions": vectors.shape[1],
            "input_template": "{procedure}",
            "dimensions": self.config.get("dimensions", 32),
            "neighbors": self.config.get("neighbors", 5),
            "seed": self.config.get("seed", 616),
            "corpus": [
                (c["candidate_id"], sha256_text(c["procedure"]))
                for c in self.candidates
            ],
            "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        }
        self.projection_fingerprint = sha256_text(json.dumps(metadata, sort_keys=True))
        cache = Path(projection_cache) if projection_cache else None
        self.projection_loaded = False
        if cache and cache.exists():
            with np.load(cache, allow_pickle=False) as loaded:
                stored_metadata = json.loads(str(loaded["metadata"].item()))
                coordinates = loaded["coordinates"]
                if stored_metadata.get("fingerprint") == self.projection_fingerprint:
                    coordinate_hash = hashlib.sha256(coordinates.tobytes()).hexdigest()
                    if coordinate_hash != stored_metadata.get("coordinates_sha256"):
                        raise ValueError("Projection coordinate checksum mismatch")
                    if (
                        coordinates.ndim != 2
                        or len(coordinates) != len(vectors)
                        or not np.isfinite(coordinates).all()
                    ):
                        raise ValueError("Invalid persisted projection")
                    self.coordinates = coordinates
                    self.projection_diagnostics = stored_metadata["diagnostics"]
                    self.projection_loaded = True
        if not self.projection_loaded:
            _, self.coordinates, self.projection_diagnostics = fit_projection(
                vectors, metadata["dimensions"], metadata["neighbors"], metadata["seed"]
            )
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache.with_name(
                    cache.name + "." + uuid.uuid4().hex + ".tmp"
                )
                stored_metadata = {
                    "fingerprint": self.projection_fingerprint,
                    "source": metadata,
                    "coordinates_sha256": hashlib.sha256(
                        self.coordinates.tobytes()
                    ).hexdigest(),
                    "diagnostics": self.projection_diagnostics,
                }
                with temporary.open("wb") as handle:
                    np.savez(
                        handle,
                        coordinates=self.coordinates,
                        metadata=json.dumps(stored_metadata),
                    )
                os.replace(temporary, cache)
        self.model = None
        self.observations = []
        self.noise_description = "Learned homoskedastic observation noise; quality metadata retained, not used as fixed heteroskedastic noise"

    def fit(self, observations):
        import torch
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim.fit import fit_gpytorch_mll_torch
        from gpytorch.mlls import ExactMarginalLogLikelihood

        self.observations = [
            dict(o) for o in observations if o.get("training_included", True)
        ]
        if not self.observations:
            self.model = None
            return self
        indices, outcomes = [], []
        for observation in self.observations:
            candidate_id = str(observation["candidate_id"])
            if candidate_id not in self._indices:
                raise ValueError(
                    "Observed candidate is absent from fixed embedding projection corpus"
                )
            outcome = float(observation.get("moc_wt_pct", observation.get("value")))
            if not np.isfinite(outcome):
                raise ValueError("Embedding GP observations must be finite")
            indices.append(self._indices[candidate_id])
            outcomes.append(outcome)
        train_x = torch.as_tensor(self.coordinates[indices], dtype=torch.double)
        train_y = torch.as_tensor(outcomes, dtype=torch.double).unsqueeze(-1)
        self.model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll_torch(mll, step_limit=int(self.config.get("fit_steps", 100)))
        self.model.eval()
        self.model.likelihood.eval()
        return self

    def predict(self, candidate_ids, observation_noise=False):
        import torch

        if self.model is None:
            raise ValueError(
                "Fit at least one measured observation before GP prediction"
            )
        indices = [self._indices[str(cid)] for cid in candidate_ids]
        means, variances = [], []
        with torch.no_grad():
            for start in range(0, len(indices), 512):
                x = torch.as_tensor(
                    self.coordinates[indices[start : start + 512]], dtype=torch.double
                )
                posterior = self.model.posterior(x, observation_noise=observation_noise)
                means.extend(posterior.mean.squeeze(-1).cpu().numpy())
                variances.extend(posterior.variance.squeeze(-1).cpu().numpy())
        return np.asarray(means), np.sqrt(np.maximum(variances, 0))

    def suggest(self, eligible_candidates):
        eligible = [
            c if isinstance(c, str) else str(c["candidate_id"])
            for c in eligible_candidates
        ]
        if not eligible:
            return {"status": "exhausted", "candidate_id": None, "records": []}
        if self.model is None:
            return {
                "status": "success",
                "candidate_id": eligible[0],
                "selection_rule": "initial_design",
                "selection_reason": "Stable first eligible candidate: no observed outcomes",
                "records": [],
            }
        mean, std = self.predict(eligible, observation_noise=False)
        maximize = self.config.get(
            "maximize", self.config.get("direction", "maximize") != "minimize"
        )
        values = [float(o.get("moc_wt_pct", o.get("value"))) for o in self.observations]
        best = max(values) if maximize else min(values)
        xi = float(self.config.get("xi", 0.0))
        improvement = (1 if maximize else -1) * (mean - best) - xi
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        ei = np.where(
            std > 0,
            improvement * ndtr(z) + std * np.exp(-z * z / 2) / np.sqrt(2 * np.pi),
            np.maximum(improvement, 0),
        )
        records = [
            {
                "candidate_id": cid,
                "mean": float(mu),
                "std": float(sigma),
                "acquisition": float(max(score, 0)),
                "acquisition_units": "raw objective units",
                "uncertainty_type": "GP latent-function posterior",
                "noise_model": self.noise_description,
            }
            for cid, mu, sigma, score in zip(eligible, mean, std, ei)
        ]
        selected = int(np.argmax(ei))
        return {
            "status": "success",
            "candidate_id": eligible[selected],
            "selection_rule": "EI",
            "selection_reason": "Maximum latent expected improvement over full eligible corpus",
            "prediction": records[selected],
            "records": records,
            "projection_fingerprint": self.projection_fingerprint,
            "projection_diagnostics": self.projection_diagnostics,
        }
