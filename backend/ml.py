"""Predictive model for STAR coverage, trained from the local history.

Cold start (few answers in the database): uses the keyword heuristic from
`star_coverage()` (defined in main.py) as the sole judge.

Once MIN_SAMPLES_TO_TRAIN real answers have accumulated in `db.py`, trains a
classifier per STAR element (Situation/Task/Action/Result) over the answer
text — using the heuristic as the training label (the teacher that trains the
model) — and starts using that model to predict coverage on new answers,
picking up text patterns the fixed keyword list doesn't. Retrains on every
saved session, so it improves with use.
"""

from pathlib import Path
from typing import List, Optional

import joblib

import db

MODEL_PATH = Path(__file__).resolve().parent / "data" / "star_model.joblib"
MIN_SAMPLES_TO_TRAIN = 20

STAR_LABELS = ["situation", "task", "action", "result"]
_COLUMN_BY_LABEL = {
    "situation": "cov_situation",
    "task": "cov_task",
    "action": "cov_action",
    "result": "cov_result",
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
    """Retrains the model with everything already in the database. Called after every saved session."""
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
            # no examples of both classes yet — this element can't be learned;
            # fall back to a "dumb" classifier that always predicts whatever
            # was observed, instead of blocking training for the other three.
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
    """STAR elements the trained model predicts this answer covers.
    Returns None (a signal for the caller to use the heuristic instead) if the
    model hasn't been trained yet or the answer is empty."""
    model = _load_model()
    if model is None or not (answer or "").strip():
        return None
    return [label for label, clf in model["classifiers"].items() if int(clf.predict([answer])[0]) == 1]


def weak_theme_profile(limit: int = 3) -> List[dict]:
    """Themes (question categories) where the candidate historically covers
    fewer STAR elements, weakest first."""
    stats = db.theme_stats()
    scored = []
    for s in stats:
        avg_cov = ((s["situation"] or 0) + (s["task"] or 0) + (s["action"] or 0) + (s["result"] or 0)) / 4.0
        scored.append({
            "theme": s["theme"],
            "n": s["n"],
            "avg_coverage": round(avg_cov, 2),
            "avg_words": round(s["avg_words"] or 0, 1),
        })
    scored.sort(key=lambda x: x["avg_coverage"])
    return scored[:limit]
