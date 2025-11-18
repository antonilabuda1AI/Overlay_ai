from __future__ import annotations

from typing import List

import numpy as np

from app.config import SETTINGS
import logging


class OpenAIClient:
    """Thin wrapper around OpenAI embeddings + chat.

    In test/dev without key, falls back to deterministic local stubs.
    """

    def __init__(self) -> None:
        self.key = SETTINGS.openai_api_key
        self._client = None
        self._force_local = SETTINGS.llm_force_local
        if self.key and not self._force_local:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.key)
            except Exception:
                self._client = None

    @staticmethod
    def _local_embed(texts: List[str], dim: int = 256) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            v = rng.normal(0, 1, dim).astype("float32")
            n = float(np.linalg.norm(v) + 1e-9)
            vecs.append(v / n)
        return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype="float32")

    def embed(self, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
        if not self._client:
            return self._local_embed(texts)
        # batch by config
        out: list[np.ndarray] = []
        try:
            for i in range(0, len(texts), SETTINGS.embed_batch_size):
                chunk = texts[i:i + SETTINGS.embed_batch_size]
                resp = self._client.embeddings.create(model=model, input=chunk)
                data = [np.array(d.embedding, dtype="float32") for d in resp.data]
                out.append(np.vstack(data))
            return np.vstack(out) if out else np.zeros((0, 0), dtype="float32")
        except Exception as e:
            logging.getLogger(__name__).warning("OpenAI embeddings failed, using local stub: %s", e)
            return self._local_embed(texts)

    def chat(self, prompt: str, contexts: List[str], model: str = "gpt-4o-mini") -> str:
        if not self._client:
            return "\n".join([prompt] + [f"- {c}" for c in contexts])[:1000]
        try:
            content = "Answer the question based on contexts.\n\n" + "\n".join([f"- {c}" for c in contexts])
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful study assistant."},
                    {"role": "user", "content": prompt + "\n\n" + content},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logging.getLogger(__name__).warning("OpenAI chat failed, using local stub: %s", e)
            return "\n".join([prompt] + [f"- {c}" for c in contexts])[:1000]
