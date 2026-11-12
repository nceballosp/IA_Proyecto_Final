"""Recuperación local TF-IDF de fragmentos de movilidad con metadatos de fuente."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "Datos" / "corpus_movilidad.json"


class MobilityRetriever:
    def __init__(self, corpus_path: Path = CORPUS_PATH):
        with corpus_path.open("r", encoding="utf-8") as file:
            self.documents = json.load(file)
        if not self.documents:
            raise ValueError(f"El corpus está vacío: {corpus_path}")
        self.vectorizer = TfidfVectorizer(
            strip_accents="unicode",
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        texts = [f"{doc['title']} {doc['text']}" for doc in self.documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.025):
        query = query.strip()
        if not query:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1]
        results = []
        for index in ranked:
            score = float(scores[index])
            if score < min_score:
                break
            doc = self.documents[int(index)]
            results.append({**doc, "score": round(score, 4)})
            if len(results) >= top_k:
                break
        return results
