"""utilities for building and selecting from a pool"""
from typing import List, Any, Callable
import numpy as np
import hashlib
import json
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import OpenAIEmbeddings


class Pool:
    """Class for sampling from pool of possible data points

    Example:
        >>> pool = Pool(['a', 'b', 'c', 'd', 'e'])
        >>> pool.sample(3)
        ['a', 'd', 'c']
        >>> pool.choose('a')
        >>> pool.sample(3)
        ['b', 'c', 'd']
        >>> pool.approx_sample('a', 3)
        ['b', 'c', 'd']
    """

    def __init__(
        self,
        pool: List[Any],
        formatter: Callable = lambda x: str(x),
        embedding_model: str = "text-embedding-3-large",
    ) -> None:
        if type(pool) is not list:
            raise TypeError("Pool must be a list")
        self._pool = list({formatter(item): item for item in pool}.values())
        self._selected = []
        self._available = self._pool[:]
        self.format = formatter
        self.embedding_model = embedding_model
        self._db = None
        self._db_fingerprint = None

    def _get_db(self):
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "texts": [self.format(x) for x in self._available],
                    "model": self.embedding_model,
                    "representation": "float32-l2-normalized-cosine-v1",
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if self._db is None or self._db_fingerprint != fingerprint:
            self._db = FAISS.from_texts(
                [self.format(x) for x in self._available],
                OpenAIEmbeddings(model=self.embedding_model),
                metadatas=[dict(data=p) for p in self._available],
                normalize_L2=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
            self._db_fingerprint = fingerprint
        return self._db

    def sample(self, n: int) -> List[str]:
        """Sample n items from the pool"""
        if n > len(self._available):
            raise ValueError("Not enough items in pool")
        indices = np.random.choice(len(self._available), size=n, replace=False)
        return [self._available[i] for i in indices]

    def choose(self, x: str) -> None:
        """Choose a specific item from the pool"""
        if x not in self._available:
            raise ValueError("Item not in pool")
        self._selected.append(x)
        self._available.remove(x)

    def approx_sample(
        self, x: str, k: int, lambda_mult: float = 0.5, fetch_k: int = 100
    ) -> List[str]:
        """Given an approximation of x, return k similar"""

        if not self._available or k == 0:
            return []
        if not 0 <= lambda_mult <= 1:
            raise ValueError("MMR lambda must lie between zero and one")
        docs = self._get_db().max_marginal_relevance_search(
            self.format(x),
            k=min(k, len(self._available)),
            fetch_k=min(fetch_k, len(self._available)),
            lambda_mult=lambda_mult,
        )
        docs = [d.metadata["data"] for d in docs]
        # remove previously chosen
        docs = [d for d in docs if d not in self._selected]
        # select k
        return docs[:k]

    def reset(self) -> None:
        """Reset the pool"""
        self._selected = []
        self._available = self._pool[:]

    def __len__(self) -> int:
        return len(self._available)

    def __repr__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __str__(self) -> str:
        return f"Pool of {len(self)} items with {len(self._selected)} selected"

    def __iter__(self):
        return iter(self._available)
