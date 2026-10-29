"""Red wine quality: can lab measurements pick out the good wines?

A wine counts as "good" if tasters scored it 7 or more out of 10 (the same
cut-off as the original notebook). Only about 1 in 7 wines are good, so the
models are judged on how many good wines they find, not just accuracy.

    python src/wine.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "winequality-red.csv"
RESULTS, IMAGES = ROOT / "results", ROOT / "images"
GOOD_FROM = 7


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["good"] = (df["quality"] >= GOOD_FROM).astype(int)
    return df


def features(df: pd.DataFrame):
    return df.drop(columns=["quality", "good"]), df["good"]


def leakage_check(df: pd.DataFrame) -> dict:
    """Reproduce the original split and count test wines that also appear in training."""
    X, y = features(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    seen = set(map(tuple, X_tr.to_numpy()))
    dup = np.array([tuple(row) in seen for row in X_te.to_numpy()])
    rf = RandomForestClassifier(200, random_state=0).fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    return {
        "duplicate_rows_in_data": int(df.duplicated().sum()),
        "test_rows": len(X_te),
        "test_rows_also_in_training": int(dup.sum()),
        "rf_accuracy_on_repeated_wines": round(accuracy_score(y_te[dup], pred[dup]), 3),
        "rf_accuracy_on_new_wines": round(accuracy_score(y_te[~dup], pred[~dup]), 3),
        "always_not_good_accuracy": round(float(1 - y_te.mean()), 3),
        "rf_good_wines_found": round(recall_score(y_te, pred), 3),
    }


def models() -> dict:
    return {
        "Always 'not good'": None,
        "Random forest": RandomForestClassifier(300, random_state=0),
        "Random forest, balanced": RandomForestClassifier(300, random_state=0, min_samples_leaf=2,
                                                          class_weight="balanced_subsample"),
        "SVM, balanced": make_pipeline(StandardScaler(), SVC(class_weight="balanced")),
        "Logistic regression, balanced": make_pipeline(StandardScaler(),
                                                       LogisticRegression(max_iter=2000, class_weight="balanced")),
    }


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """5-fold stratified cross-validation on de-duplicated wines; scaling happens inside each fold."""
    X, y = features(df.drop_duplicates())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    rows = []
    for name, model in models().items():
        if model is None:
            rows.append({"model": name, "accuracy": round(float(1 - y.mean()), 3), "good_found_recall": 0.0,
                         "precision": None, "f1": 0.0, "roc_auc": 0.5})
            continue
        s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "precision", "f1", "roc_auc"])
        rows.append({"model": name,
                     "accuracy": round(s["test_accuracy"].mean(), 3),
                     "good_found_recall": round(s["test_recall"].mean(), 3),
                     "precision": round(s["test_precision"].mean(), 3),
                     "f1": round(s["test_f1"].mean(), 3),
                     "roc_auc": round(s["test_roc_auc"].mean(), 3)})
    return pd.DataFrame(rows)


def plot_tradeoff(results: pd.DataFrame):
    r = results.dropna(subset=["precision"])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    y = np.arange(len(r))
    ax.barh(y - 0.2, r["good_found_recall"], 0.4, color="#7a2e3a", label="Good wines found (recall)")
    ax.barh(y + 0.2, r["precision"], 0.4, color="#d9a5ae", label="Flagged wines that really are good (precision)")
    for yy, a, b in zip(y, r["good_found_recall"], r["precision"]):
        ax.text(a + 0.01, yy - 0.2, f"{a:.0%}", va="center", fontsize=9)
        ax.text(b + 0.01, yy + 0.2, f"{b:.0%}", va="center", fontsize=9)
    ax.set_yticks(y, r["model"])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Finding good wines: catch more, or be right more often", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "recall_precision_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_drivers(df: pd.DataFrame):
    X, y = features(df.drop_duplicates())
    rf = RandomForestClassifier(300, random_state=0).fit(X, y)
    imp = pd.Series(rf.feature_importances_, X.columns).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].barh(imp.index, imp.values, color=["#7a2e3a" if v >= imp.nlargest(3).min() else "#c9b3b8" for v in imp])
    axes[0].set_title("What the random forest relies on", loc="left", fontsize=11)
    axes[0].set_xlabel("Importance")
    d = df.drop_duplicates()
    bins = np.arange(8, 15.25, 0.25)
    axes[1].hist(d.loc[d["good"] == 0, "alcohol"], bins=bins, color="#c9b3b8", label="Not good (score 3-6)", density=True)
    axes[1].hist(d.loc[d["good"] == 1, "alcohol"], bins=bins, color="#7a2e3a", alpha=0.75, label="Good (score 7-8)", density=True)
    axes[1].set_title("Good wines tend to have more alcohol", loc="left", fontsize=11)
    axes[1].set_xlabel("Alcohol (% vol)")
    axes[1].set_yticks([])
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMAGES / "quality_drivers.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    RESULTS.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    leak = leakage_check(df)
    (RESULTS / "leakage_check.json").write_text(json.dumps(leak, indent=2))
    results = evaluate(df)
    results.to_csv(RESULTS / "results.csv", index=False)
    plot_tradeoff(results)
    plot_drivers(df)
    print("Leakage check on the original split:", leak)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()
