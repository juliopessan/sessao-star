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

MODEL_DIR = Path(__file__).resolve().parent / "data"
MIN_SAMPLES_TO_TRAIN = 20

STAR_LABELS = ["situation", "task", "action", "result"]
_COLUMN_BY_LABEL = {
    "situation": "cov_situation",
    "task": "cov_task",
    "action": "cov_action",
    "result": "cov_result",
}

_model_cache = {}
_model_loaded = set()


def _normalise_lang(lang: str) -> str:
    return "en" if lang == "en" else "pt"


def model_path(lang: str) -> Path:
    return MODEL_DIR / f"star_model_{_normalise_lang(lang)}.joblib"


def _load_model(lang: str = "pt"):
    lang = _normalise_lang(lang)
    if lang not in _model_loaded:
        _model_loaded.add(lang)
        path = model_path(lang)
        if path.exists():
            try:
                _model_cache[lang] = joblib.load(path)
            except Exception:
                _model_cache[lang] = None
    return _model_cache.get(lang)


def model_status(lang: str = "pt") -> dict:
    lang = _normalise_lang(lang)
    return {
        "language": lang,
        "trained": _load_model(lang) is not None,
        "answers_available": db.answer_count(lang),
        "min_samples_to_train": MIN_SAMPLES_TO_TRAIN,
    }


def train(lang: str = "pt") -> Optional[dict]:
    """Retrains the model with everything already in the database. Called after every saved session."""
    lang = _normalise_lang(lang)
    rows = db.all_answers(lang)
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
    model_path(lang).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path(lang))

    _model_cache[lang] = bundle
    _model_loaded.add(lang)
    return bundle


def predict_coverage(answer: str, lang: str = "pt") -> Optional[List[str]]:
    """STAR elements the trained model predicts this answer covers.
    Returns None (a signal for the caller to use the heuristic instead) if the
    model hasn't been trained yet or the answer is empty."""
    model = _load_model(lang)
    if model is None or not (answer or "").strip():
        return None
    return [label for label, clf in model["classifiers"].items() if int(clf.predict([answer])[0]) == 1]


def weak_theme_profile(limit: int = 3, lang: str = "pt") -> List[dict]:
    """Themes (question categories) where the candidate historically covers
    fewer STAR elements, weakest first."""
    stats = db.theme_stats(lang)
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
