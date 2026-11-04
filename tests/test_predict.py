import json

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from dsrv.features import build_pipeline
from dsrv.config import load_yaml
from dsrv.hashing import sha256_file
from dsrv.predict import load_packaged_model, predict_file, predict_frame
from dsrv.recovery_reference import select_fixed_rows

FEATURES = ["dur", "proto", "service", "state"]


def make_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, rows)
    return pd.DataFrame(
        {
            "row_id": [f"train:{i}" for i in range(rows)],
            "dur": rng.random(rows) + label,
            "proto": rng.choice(["tcp", "udp"], rows),
            "service": rng.choice(["-", "dns"], rows),
            "state": rng.choice(["FIN", "CON"], rows),
            "label": label,
        }
    )


@pytest.fixture
def packaged(tmp_path):
    frame = make_frame()
    pipeline = build_pipeline(load_yaml("configs/e1_lr_base.yaml"), frame[FEATURES])
    pipeline.fit(frame[FEATURES], frame["label"])
    joblib.dump(pipeline, tmp_path / "model.joblib")
    config = {
        "threshold": 0.5,
        "positive_class": {"label": 1, "name": "ataque"},
        "input_columns": FEATURES,
        "sklearn_version": sklearn.__version__,
        "model_sha256": sha256_file(tmp_path / "model.joblib"),
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, frame, config


def test_output_has_row_id_score_and_decision(packaged):
    directory, frame, _ = packaged
    model, config = load_packaged_model(directory)
    result = predict_frame(model, config, frame)

    assert list(result.columns) == ["row_id", "score_attack", "y_pred"]
    assert (result["row_id"] == frame["row_id"]).all()
    assert (result["y_pred"] == (result["score_attack"] >= 0.5).astype(int)).all()


def test_threshold_comes_from_the_config(packaged):
    directory, frame, _ = packaged
    model, config = load_packaged_model(directory)

    all_attack = predict_frame(model, {**config, "threshold": 0.0}, frame)
    no_attack = predict_frame(model, {**config, "threshold": 1.01}, frame)

    assert all_attack["y_pred"].eq(1).all()
    assert no_attack["y_pred"].eq(0).all()


def test_extra_columns_and_order_do_not_change_the_result(packaged):
    directory, frame, _ = packaged
    model, config = load_packaged_model(directory)
    reordered = frame[["label", "state", "row_id", "service", "dur", "proto"]].copy()
    reordered["label"] = 1 - reordered["label"]

    expected = predict_frame(model, config, frame)
    pd.testing.assert_frame_equal(predict_frame(model, config, reordered), expected)


@pytest.mark.parametrize("column", ["dur", "row_id"])
def test_missing_column_fails(packaged, column):
    directory, frame, _ = packaged
    model, config = load_packaged_model(directory)

    with pytest.raises(ValueError, match=column):
        predict_frame(model, config, frame.drop(columns=column))


def test_refuses_to_load_a_model_with_another_hash(packaged):
    directory, _, config = packaged
    (directory / "config.json").write_text(
        json.dumps({**config, "model_sha256": "0" * 64}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_packaged_model(directory)


def test_refuses_another_sklearn_version(packaged):
    directory, _, config = packaged
    (directory / "config.json").write_text(
        json.dumps({**config, "sklearn_version": "0.0"}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="scikit-learn"):
        load_packaged_model(directory)


def test_predict_file_matches_predict_frame(packaged, tmp_path):
    directory, frame, _ = packaged
    input_csv = tmp_path / "entrada.csv"
    output_csv = tmp_path / "salida.csv"
    frame.to_csv(input_csv, index=False)

    predict_file(directory, input_csv, output_csv)

    model, config = load_packaged_model(directory)
    pd.testing.assert_frame_equal(pd.read_csv(output_csv), predict_frame(model, config, frame))


def test_fixed_rows_are_balanced_ordered_and_reproducible():
    frame = make_frame(200)
    first = select_fixed_rows(frame, FEATURES)
    second = select_fixed_rows(frame, FEATURES)

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["row_id", *FEATURES, "y_true"]
    assert first["y_true"].value_counts().to_dict() == {0: 20, 1: 20}
    assert first.index.is_monotonic_increasing
