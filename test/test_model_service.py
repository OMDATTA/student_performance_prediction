from services.model_service import _category


def test_notebook_category_thresholds():
    assert _category(15) == "High Performance"
    assert _category(10) == "Average Performance"
    assert _category(9.99) == "Low Performance"


def test_training_bundle_uses_expected_feature_set(monkeypatch):
    import pandas as pd
    from services import model_service

    # Small local fixture keeps this test offline and fast.
    n = 50
    df = pd.DataFrame({
        "age": [15 + (i % 8) for i in range(n)],
        "studytime": [1 + (i % 4) for i in range(n)],
        "failures": [i % 4 for i in range(n)],
        "absences": [i % 20 for i in range(n)],
        "G1": [5 + (i % 15) for i in range(n)],
        "G2": [4 + ((i * 2) % 16) for i in range(n)],
        "G3": [float(min(20, max(0, 0.15 * (5 + i % 15) + 0.65 * (4 + ((i * 2) % 16)) + i % 3))) for i in range(n)],
    })
    monkeypatch.setattr(model_service, "_load_training_data", lambda: (df, "test fixture"))
    bundle = model_service._train_bundle()
    assert set(bundle["features"]) == {"age", "studytime", "failures", "absences", "G1", "G2"}
    assert set(bundle["models"]) == {"linear_regression", "random_forest"}
    assert set(bundle["metrics"]) == {"linear_regression", "random_forest"}
    assert all(metric in bundle["metrics"]["random_forest"] for metric in ("mae", "rmse", "r2"))
    assert bundle["sample_count"] == 50


def test_dataframe_loader_reads_semicolon_csv_bytes():
    from services.model_service import _read_dataframe_from_bytes
    raw = (
        b"age;studytime;failures;absences;G1;G2;G3\n"
        b"17;3;0;5;11;11;12\n"
        b"16;2;1;8;9;10;10\n"
    )
    frame = _read_dataframe_from_bytes(raw)
    assert frame.shape == (2, 7)
    assert list(frame.columns)[-1] == "G3"
