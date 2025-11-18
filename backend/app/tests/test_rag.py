import numpy as np

from app.rag.index import Doc, VectorIndex
from app.rag.select import mmr


def test_vector_index_numpy_fallback():
    idx = VectorIndex(dim=8)
    vecs = np.eye(8, dtype="float32")[:3]
    docs = [Doc(1, "a"), Doc(2, "b"), Doc(3, "c")]
    idx.add(vecs, docs)
    q = vecs[0]
    res = idx.search(q, top_k=2)
    assert len(res) == 2
    assert res[0][0].text == "a"


def test_mmr_selection():
    q = np.ones((8,), dtype="float32") / np.sqrt(8)
    cands = [np.eye(8, dtype="float32")[i] for i in range(4)]
    picks = mmr(q, cands, top_k=2)
    assert len(picks) == 2
    assert len(set(picks)) == 2

