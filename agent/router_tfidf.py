"""A local, learned router: TF-IDF + logistic regression over the question bank (docs/test_questions.md) and the
organiser workbook. It predicts the question CATEGORY only; entities (stations, dates, lines) are still extracted
deterministically by router.py. Used by the experiment suite to compare rules vs a learned classifier.

It is trained at start-up (a few ms), never on the questions under test (`exclude` removes them), so the
comparison is not contaminated by leakage. With ~85 examples it is a *small-data* baseline, not a claim about
what a properly trained classifier could do.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_MODEL = {}


def _training_set(exclude: tuple[str, ...]) -> tuple[list[str], list[str]]:
    sys.path.insert(0, str(REPO / "evaluation"))
    import dataset

    cat_of_training = {"T01": "A", "T02": "B", "T03": "C", "T04": "D", "T05": "E", "T06": "F", "T07": "B",
                       "T08": "G", "T09": "H", "T10": "X", "T11": "X"}
    texts, labels = [], []
    for it in dataset.load_workbook_items():
        if it.stage == "TRAINING":
            texts.append(it.question); labels.append(cat_of_training[it.id])
    for line in dataset.BANK.read_text().splitlines():           # keep follow-ups too: they are labelled with their category
        m = re.match(r"\|\s*([A-HX]\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*.+\|\s*$", line)
        if m:
            texts.append(re.sub(r"^\(after [A-Z]\d+\)\s*|[*`]", "", m.group(3))); labels.append(m.group(1)[0])
    keep = [(t, l) for t, l in zip(texts, labels) if not any(t.strip().lower().startswith(e.strip().lower()[:60]) for e in exclude)]
    return [t for t, _ in keep], [l for _, l in keep]


def fit(exclude: tuple[str, ...] = ()):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts, labels = _training_set(exclude)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True, min_df=1),
                         LogisticRegression(max_iter=2000, C=10.0))
    pipe.fit(texts, labels)
    _MODEL["pipe"], _MODEL["n"] = pipe, len(texts)
    return pipe


def classify(question: str, exclude: tuple[str, ...] = ()) -> tuple[str, float]:
    if "pipe" not in _MODEL:
        fit(exclude)
    pipe = _MODEL["pipe"]
    proba = pipe.predict_proba([question])[0]
    i = int(proba.argmax())
    return str(pipe.classes_[i]), float(proba[i])
