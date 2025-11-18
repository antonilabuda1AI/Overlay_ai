from __future__ import annotations

from typing import List, Tuple

import numpy as np


def mmr(query: np.ndarray, cands: List[np.ndarray], lambda_mult: float = 0.5, top_k: int = 6) -> List[int]:
    """Maximal Marginal Relevance selection over normalized vectors."""
    if not cands:
        return []
    sims = np.array([(query @ c).item() for c in cands], dtype=float)
    selected: List[int] = []
    candidates = list(range(len(cands)))
    while candidates and len(selected) < top_k:
        if not selected:
            i = int(np.argmax(sims[candidates]))
            selected.append(candidates[i])
            candidates.pop(i)
            continue
        # diversity penalty
        max_scores: List[Tuple[int, float]] = []
        for idx in candidates:
            div = max((cands[idx] @ cands[s]).item() for s in selected)
            score = lambda_mult * sims[idx] - (1 - lambda_mult) * div
            max_scores.append((idx, score))
        best = max(max_scores, key=lambda x: x[1])[0]
        selected.append(best)
        candidates.remove(best)
    return selected

