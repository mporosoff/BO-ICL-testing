"""Embedding GP compatibility API; use EmbeddingGPEngine for campaign workflows."""
import ast
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np
import pandas as pd
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from .asktell import AskTellFewShot
from .embedding_cache import EmbeddingCache, EmbeddingSpec, sha256_text
from .embedding_gp import fit_projection
from .llm_model import GaussDist
from .pool import Pool


class AskTellGPR(AskTellFewShot):
    def __init__(
        self,
        n_components=32,
        pool=None,
        cache_path=None,
        n_neighbors=5,
        embedding_model="text-embedding-ada-002",
        embedding_dimensions=None,
        embedder=None,
        cancelled=None,
        fit_steps=100,
        request_settings=None,
        seed=616,
        **kwargs,
    ):
        super().__init__(embedding_model=embedding_model, **kwargs)
        self._selector_k = None
        self.examples = []
        self.pool = pool
        self.n_components, self.n_neighbors = n_components, n_neighbors
        self.projection_seed = seed
        self.isomap = None
        self._projection_corpus = None
        self._projection_lookup = {}
        self.fit_steps = fit_steps
        self._embedder = embedder
        self._cancel_event = cancelled
        self._cancelled = (
            cancelled.is_set if hasattr(cancelled, "is_set") else cancelled
        )
        self._request_settings = request_settings
        self._cache_path = Path(cache_path) if cache_path else None
        dimensions = embedding_dimensions or {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
        }.get(embedding_model)
        if dimensions is None:
            raise ValueError(
                "Declare embedding_dimensions for an unfamiliar embedding model"
            )
        self.embedding_spec = EmbeddingSpec(embedding_model, dimensions)
        safe_directory = (
            self._cache_path.with_name(
                self._cache_path.name + ".safe-" + self.embedding_spec.fingerprint[:12]
            )
            if self._cache_path
            else Path(tempfile.mkdtemp(prefix="boicl-embedding-"))
        )
        self.embedding_cache = EmbeddingCache(safe_directory, self.embedding_spec)
        self.cache_import_errors = []
        self._embeddings_cache = self._get_cache(cache_path)
        self._set_regressor()

    def _get_cache(self, cache_path=None):
        columns = ["x", "embedding", "embedding_model", "input_sha256", "dimensions"]
        if not cache_path or not Path(cache_path).exists():
            return pd.DataFrame(columns=columns)
        cache = pd.read_csv(cache_path)
        if not {"x", "embedding", "embedding_model"}.issubset(cache.columns):
            self.cache_import_errors.append(
                "Legacy CSV lacks explicit input or model identity; no rows reused"
            )
            return pd.DataFrame(columns=columns)
        accepted = []
        for _, row in cache.iterrows():
            try:
                if row["embedding_model"] != self.embedding_model:
                    continue
                exact = self.embedding_spec.format(row["x"])
                digest = sha256_text(exact)
                if (
                    "input_sha256" in row
                    and pd.notna(row["input_sha256"])
                    and row["input_sha256"] != digest
                ):
                    raise ValueError("Legacy CSV exact-input hash mismatch")
                if (
                    "dimensions" in row
                    and pd.notna(row["dimensions"])
                    and int(row["dimensions"]) != self.embedding_spec.dimensions
                ):
                    raise ValueError("Legacy CSV declared dimension mismatch")
                vector = self.embedding_cache._vector(
                    self._parse_embedding(row["embedding"])
                )
                record = {
                    "candidate_id": digest,
                    "input_text": exact,
                    "input_sha256": digest,
                    "namespace": "candidate",
                    "source": "validated legacy CSV; exact stored text",
                }
                self.embedding_cache._entries[self.embedding_cache._key(record)] = (
                    record,
                    vector,
                )
                accepted.append(
                    dict(
                        x=row["x"],
                        embedding=vector.tolist(),
                        embedding_model=self.embedding_model,
                        input_sha256=digest,
                        dimensions=self.embedding_spec.dimensions,
                    )
                )
            except (TypeError, ValueError) as error:
                self.cache_import_errors.append(str(error))
        return pd.DataFrame(accepted, columns=columns)

    @staticmethod
    def _parse_embedding(value):
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as error:
                raise ValueError("Invalid numeric embedding list in CSV") from error
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        raise ValueError("Embedding must be a numeric list")

    def _sync_legacy_table(self):
        self._embeddings_cache = pd.DataFrame(
            [
                {
                    "x": row["input_text"],
                    "embedding": vector.tolist(),
                    "embedding_model": self.embedding_model,
                    "input_sha256": row["input_sha256"],
                    "dimensions": self.embedding_spec.dimensions,
                }
                for row, vector in self.embedding_cache._entries.values()
                if row.get("namespace") == "candidate"
            ],
            columns=["x", "embedding", "embedding_model", "input_sha256", "dimensions"],
        )

    def save_cache(self, cache_path):
        self._sync_legacy_table()
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        self._embeddings_cache.to_csv(temporary, index=False)
        os.replace(temporary, path)
        self.embedding_cache._checkpoint()

    def _provider_embeddings(self, inputs):
        if self._embedder:
            return self._embedder(inputs)
        # Construct remote clients only for an intentional cache miss. The
        # shared wrapper owns total attempts, cancellation and request spacing.
        from openai import OpenAI
        from .request_policy import ReliableClient, RequestPolicy

        if not hasattr(self, "_embedding_client"):
            policy = RequestPolicy(self._request_settings, cancelled=self._cancel_event)
            self._embedding_client = ReliableClient(OpenAI(max_retries=0), policy)
        return self._embedding_client.embeddings.create(
            input=inputs, model=self.embedding_model, encoding_format="float"
        )

    def _query_cache(self, X):
        values = list(X)
        if any(not isinstance(x, str) or not x.strip() for x in values):
            raise ValueError("Embedding inputs must be nonempty procedure strings")
        unique = list(dict.fromkeys(values))
        records = [{"candidate_id": sha256_text(x), "procedure": x} for x in unique]
        report = self.embedding_cache.prepare(
            records, self._provider_embeddings, batch_size=64, cancelled=self._cancelled
        )
        self._sync_legacy_table()
        if self._cache_path:
            self.save_cache(self._cache_path)
        if report["missing_ids"]:
            raise ValueError(
                f"{len(report['missing_ids'])} embeddings remain missing; successful batches saved for resume"
            )
        matrix = self.embedding_cache.matrix(records)
        by_input = dict(zip(unique, matrix))
        return [by_input[x].tolist() for x in values]

    def _initialize_isomap(self):
        if self.pool is None:
            raise ValueError(
                "Pass the fixed full candidate pool, including measured designs, before fitting embedding GP"
            )
        original = self.pool._pool if isinstance(self.pool, Pool) else list(self.pool)
        corpus = list(dict.fromkeys(self.format_x(x) for x in original))
        if not corpus:
            raise ValueError("Embedding projection corpus is empty")
        if self._projection_corpus == corpus:
            return
        vectors = self._query_cache(corpus)
        self.isomap, coordinates, self.projection_diagnostics = fit_projection(
            vectors, self.n_components, self.n_neighbors, self.projection_seed
        )
        self._projection_corpus = corpus
        self._projection_lookup = dict(zip(corpus, coordinates))
        self.projection_fingerprint = sha256_text(
            str(
                (
                    corpus,
                    self.embedding_spec.fingerprint,
                    self.n_components,
                    self.n_neighbors,
                    self.projection_seed,
                )
            )
        )

    def _project(self, X):
        self._initialize_isomap()
        if any(x not in self._projection_lookup for x in X):
            raise ValueError(
                "Procedure is absent from the fixed full projection corpus"
            )
        return np.asarray([self._projection_lookup[x] for x in X], dtype=float)

    def _set_regressor(self):
        self.regressor = None
        self.likelihood = None

    def _train(self, X, y):
        train_x = torch.as_tensor(self._project(X), dtype=torch.double)
        train_y = torch.as_tensor(list(map(float, y)), dtype=torch.double).unsqueeze(-1)
        if not torch.isfinite(train_y).all():
            raise ValueError("Observed outcomes must be finite")
        self.regressor = SingleTaskGP(
            train_x, train_y, outcome_transform=Standardize(m=1)
        )
        self.likelihood = self.regressor.likelihood
        mll = ExactMarginalLogLikelihood(self.likelihood, self.regressor)
        fit_gpytorch_mll_torch(mll, step_limit=self.fit_steps)

    def _predict(self, X, observation_noise=False):
        if not X or self.regressor is None:
            raise ValueError("Fit measured observations before requesting predictions")
        query = torch.as_tensor(self._project(X), dtype=torch.double)
        with torch.no_grad():
            self.regressor.eval()
            self.regressor.likelihood.eval()
            posterior = self.regressor.posterior(
                query, observation_noise=observation_noise
            )
            means = posterior.mean.squeeze(-1)
            stds = posterior.variance.clamp_min(0).sqrt().squeeze(-1)
        return [GaussDist(mean.item(), std.item()) for mean, std in zip(means, stds)], 0

    def tell(self, x, y, alt_ys=None, train=True):
        if alt_ys is not None:
            raise ValueError("Alternative completion responses are not GP measurements")
        if self.use_quantiles:
            raise ValueError(
                "Embedding GP uses fitted outcome standardization; external quantile transformation is unsupported"
            )
        procedure, outcome = self.format_x(x), float(y)
        if not np.isfinite(outcome):
            raise ValueError("Observed outcomes must be finite")
        self.examples.append({"x": procedure, "y": outcome, "y_name": self._y_name})
        self._observed_x.add(procedure)
        self._ys.append(outcome)
        self._example_count += 1
        self._ready = True
        if train:
            self._train(
                [e["x"] for e in self.examples], [e["y"] for e in self.examples]
            )

    def predict(self, x, system_message=None, observation_noise=False):
        single = not isinstance(x, list)
        values = [x] if single else x
        inputs = [self.format_x(item) for item in values]
        if observation_noise:
            results, tokens = self._predict(inputs, observation_noise=True)
        else:
            results, tokens = self._predict(inputs)
        self.tokens_used += tokens
        return results[0] if single else results

    def _ask(self, possible_x, best, aq_fxn, k, system_message):
        results = self.predict(possible_x)
        results = results if isinstance(results, list) else [results]
        records = [
            (x, dist, float(aq_fxn(dist, best)))
            for x, dist in zip(possible_x, results)
            if len(dist) > 0
        ]
        records = [record for record in records if np.isfinite(record[2])]
        records.sort(key=lambda record: -record[2])
        records = records[:k]
        return (
            [r[0] for r in records],
            [r[2] for r in records],
            [r[1].mean() for r in records],
            [r[1].std() for r in records],
        )

    def ask(
        self,
        possible_x,
        aq_fxn="expected_improvement",
        k=1,
        inv_filter=None,
        aug_random_filter=None,
        lambda_mult=0.5,
        _lambda=0.5,
        system_message="",
        inv_system_message="",
    ):
        return super().ask(
            possible_x,
            aq_fxn,
            k,
            inv_filter=0,
            aug_random_filter=len(possible_x),
            lambda_mult=lambda_mult,
            _lambda=_lambda,
            system_message=system_message,
            inv_system_message=inv_system_message,
        )
