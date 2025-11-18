from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from time import perf_counter_ns
from app.config import SETTINGS


try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional
    faiss = None


@dataclass
class Doc:
    ts_ms: int
    text: str


class VectorIndex:
    """Cosine similarity via inner product on L2-normalized vectors.

    Uses FAISS if available, otherwise a simple numpy index.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vecs: List[np.ndarray] = []
        self.docs: List[Doc] = []
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexFlatIP(dim)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype("float32")

    def add(self, vecs: np.ndarray, metas: List[Doc]) -> None:
        v = np.vstack([self._normalize(x) for x in vecs])
        self.vecs.extend(list(v))
        self.docs.extend(metas)
        if self._faiss is not None:
            self._faiss.add(v)

    def search(self, q: np.ndarray, top_k: int = 6) -> List[Tuple[Doc, float]]:
        qn = self._normalize(q.reshape(1, -1))
        if self._faiss is not None:
            scores, idxs = self._faiss.search(qn, top_k)
            out: List[Tuple[Doc, float]] = []
            for i, s in zip(idxs[0], scores[0]):
                if i == -1:
                    continue
                out.append((self.docs[i], float(s)))
            return out
        # numpy fallback
        if not self.vecs:
            return []
        t0 = perf_counter_ns()
        mat = np.vstack(self.vecs)
        sims = mat @ qn.T
        order = np.argsort(-sims.flatten())[:top_k]
        if (perf_counter_ns() - t0) / 1e6 > SETTINGS.search_timeout_ms:
            order = order[: max(1, top_k // 2)]
        return [(self.docs[int(i)], float(sims[int(i)])) for i in order]
