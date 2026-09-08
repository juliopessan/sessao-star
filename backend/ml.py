"""Modelo preditivo de cobertura STAR, treinado a partir do histórico local.

Início "frio" (poucas respostas no banco): usa a heurística de palavras-chave
de `star_coverage()` (definida em main.py) como único juiz.

Depois de MIN_SAMPLES_TO_TRAIN respostas reais acumuladas em `db.py`, treina
um classificador por elemento do STAR (Situação/Tarefa/Ação/Resultado) sobre
o texto das respostas — usando a heurística como rótulo de treino (o professor
que ensina o modelo) — e passa a usar esse modelo para prever cobertura em
respostas novas, capturando padrões de texto que a lista fixa de palavras-chave
não cobre. Reaprende a cada sessão salva, então melhora com o uso.
"""

from pathlib import Path
from typing import List, Optional

import joblib

import db

MODEL_PATH = Path(__file__).resolve().parent / "data" / "star_model.joblib"
MIN_SAMPLES_TO_TRAIN = 20

STAR_LABELS = ["situação", "tarefa", "ação", "resultado"]
_COLUMN_BY_LABEL = {
    "situação": "cov_situacao",
    "tarefa": "cov_tarefa",
    "ação": "cov_acao",
    "resultado": "cov_resultado",
}

_model_cache = None
_model_loaded = False


def _load_model():
    global _model_cache, _model_loaded
    if not _model_loaded:
        _model_loaded = True
        if MODEL_PATH.exists():
            try:
                _model_cache = joblib.load(MODEL_PATH)
            except Exception:
                _model_cache = None
    return _model_cache


def model_status() -> dict:
    return {
        "trained": _load_model() is not None,
        "answers_available": db.answer_count(),
        "min_samples_to_train": MIN_SAMPLES_TO_TRAIN,
    }


def train() -> Optional[dict]:
    """Retreina o modelo com tudo que já está no banco. Chamado após cada sessão salva."""
    rows = db.all_answers()
    if len(rows) < MIN_SAMPLES_TO_TRAIN:
        return None

    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    texts = [r["answer"] for r in rows]
    classifiers = {}
    for label, column in _COLUMN_BY_LABEL.items():
        y = [r[column] for r in rows]
        if len(set(y)) < 2:
            # sem exemplos das duas classes ainda — não dá pra aprender esse
            # elemento; usa um classificador "burro" que sempre prevê o que
            # foi observado, em vez de travar o treino dos outros três.
            clf = DummyClassifier(strategy="constant", constant=int(y[0]))
        else:
            clf = Pipeline([
                ("tfidf", TfidfVectorizer(max_features=400, ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
        clf.fit(texts, y)
        classifiers[label] = clf

    bundle = {"classifiers": classifiers, "n_samples": len(rows)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    global _model_cache, _model_loaded
    _model_cache = bundle
    _model_loaded = True
    return bundle


def predict_coverage(answer: str) -> Optional[List[str]]:
    """Elementos STAR que o modelo treinado prevê que a resposta cobre.
    Retorna None (sinal para o chamador usar a heurística) se o modelo ainda
    não foi treinado ou a resposta está vazia."""
    model = _load_model()
    if model is None or not (answer or "").strip():
        return None
    return [label for label, clf in model["classifiers"].items() if int(clf.predict([answer])[0]) == 1]


def weak_theme_profile(limit: int = 3) -> List[dict]:
    """Temas (categorias de pergunta) onde o candidato historicamente cobre
    menos elementos do STAR, dos mais fracos pra menos fracos."""
    stats = db.theme_stats()
    scored = []
    for s in stats:
        avg_cov = ((s["situacao"] or 0) + (s["tarefa"] or 0) + (s["acao"] or 0) + (s["resultado"] or 0)) / 4.0
        scored.append({
            "theme": s["theme"],
            "n": s["n"],
            "avg_coverage": round(avg_cov, 2),
            "avg_words": round(s["avg_words"] or 0, 1),
        })
    scored.sort(key=lambda x: x["avg_coverage"])
    return scored[:limit]
