import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import wine  # noqa: E402


def test_data_and_label():
    df = wine.load()
    assert len(df) == 1599 and df.isna().sum().sum() == 0
    assert set(df.loc[df["good"] == 1, "quality"]) == {7, 8}
    assert set(df.loc[df["good"] == 0, "quality"]) == {3, 4, 5, 6}


def test_leakage_check_finds_the_repeated_wines():
    leak = wine.leakage_check(wine.load())
    assert leak["duplicate_rows_in_data"] == 240
    assert leak["test_rows_also_in_training"] == 64


def test_fair_evaluation_has_no_duplicates():
    df = wine.load().drop_duplicates()
    X, _ = wine.features(df)
    assert not X.duplicated().any()


def test_balanced_models_find_more_good_wines():
    r = wine.evaluate(wine.load()).set_index("model")
    assert r.loc["Logistic regression, balanced", "good_found_recall"] > r.loc["Random forest", "good_found_recall"]
