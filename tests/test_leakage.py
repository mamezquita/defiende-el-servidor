import numpy as np
import pandas as pd
import pytest

from dsrv.config import load_yaml
from dsrv.features import (
    FORBIDDEN_COLUMNS,
    build_pipeline,
    use_single_thread_inference,
    validate_config,
)

CONFIGS = [
    "configs/e1_lr_base.yaml",
    "configs/e2_lr_log_balanced.yaml",
    "configs/e3_rf.yaml",
    "configs/e4_hgb.yaml",
]
FEATURES = ["dur", "sbytes", "proto", "service", "state"]


def make_frame(rows: int = 120, seed: int = 0) -> pd.DataFrame:
    """Datos sintéticos con las columnas de trazabilidad que traen los parquet."""
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, rows)
    return pd.DataFrame(
        {
            "row_id": [f"train:{i}" for i in range(rows)],
            "id": np.arange(rows),
            "group_id": [f"g{i}" for i in range(rows)],
            "dur": rng.random(rows) + label,
            "sbytes": rng.integers(0, 1000, rows) * (label + 1),
            "proto": rng.choice(["tcp", "udp"], rows),
            "service": rng.choice(["-", "dns"], rows),
            "state": rng.choice(["FIN", "CON"], rows),
            "attack_cat": np.where(label == 1, "Generic", "Normal"),
            "label": label,
        }
    )


def fit(config_path: str, frame: pd.DataFrame):
    pipeline = build_pipeline(load_yaml(config_path), frame[FEATURES])
    pipeline.fit(frame[FEATURES], frame["label"])
    use_single_thread_inference(pipeline)
    return pipeline


@pytest.mark.parametrize("config_path", CONFIGS)
def test_configs_are_valid(config_path):
    validate_config(load_yaml(config_path))


@pytest.mark.parametrize("config_path", CONFIGS)
@pytest.mark.parametrize("column", sorted(FORBIDDEN_COLUMNS))
def test_rejects_forbidden_column(config_path, column):
    frame = make_frame()

    with pytest.raises(ValueError, match=column):
        build_pipeline(load_yaml(config_path), frame[[*FEATURES, column]])


@pytest.mark.parametrize("config_path", CONFIGS)
def test_ignores_forbidden_columns_when_predicting(config_path):
    frame = make_frame()
    pipeline = fit(config_path, frame)

    altered = frame.copy()
    altered["label"] = 1 - altered["label"]
    altered["attack_cat"] = "Otro"
    altered["id"] = -1
    altered["row_id"] = "otro"
    altered["group_id"] = "otro"

    assert np.array_equal(pipeline.predict_proba(frame), pipeline.predict_proba(altered))


@pytest.mark.parametrize("config_path", CONFIGS)
def test_preprocessing_uses_exactly_the_features(config_path):
    pipeline = fit(config_path, make_frame())
    preprocess = pipeline.named_steps["preprocess"]

    used = [
        column
        for name, _, columns in preprocess.transformers_
        if name != "remainder"
        for column in columns
    ]
    assert sorted(used) == sorted(FEATURES)
    assert preprocess.remainder == "drop"


@pytest.mark.parametrize("config_path", CONFIGS)
def test_unknown_category_does_not_fail_and_is_not_learned(config_path):
    frame = make_frame()
    pipeline = fit(config_path, frame)

    unseen = frame.head(5).copy()
    unseen["proto"] = "solo_en_val"

    assert pipeline.predict_proba(unseen[FEATURES]).shape == (5, 2)
    encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"]
    assert "solo_en_val" not in encoder.categories_[0]


def test_scaler_is_fitted_only_with_the_training_frame():
    train, val = make_frame(seed=0), make_frame(seed=1)
    pipeline = fit(CONFIGS[0], train)
    pipeline.predict_proba(val[FEATURES])

    numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["scale"].mean_ == pytest.approx(train[["dur", "sbytes"]].mean())


def test_log1p_only_applies_to_columns_non_negative_in_train():
    frame = make_frame()
    frame["dur"] = frame["dur"] - 5
    pipeline = build_pipeline(load_yaml(CONFIGS[1]), frame[FEATURES])
    preprocess = pipeline.named_steps["preprocess"]
    transformers = {name: columns for name, _, columns in preprocess.transformers}

    assert transformers["numeric_log1p"] == ["sbytes"]
    assert transformers["numeric"] == ["dur"]


def test_negative_value_outside_train_fails_visibly():
    frame = make_frame()
    pipeline = fit(CONFIGS[1], frame)
    invalid = frame.head(5).copy()
    invalid["dur"] = -5.0

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="NaN"):
        pipeline.predict_proba(invalid[FEATURES])


def test_single_thread_inference_only_changes_estimators_with_n_jobs():
    frame = make_frame()
    forest = fit(CONFIGS[2], frame)
    linear = fit(CONFIGS[0], frame)

    assert forest.named_steps["model"].n_jobs == 1
    assert linear.named_steps["model"].get_params()["n_jobs"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("hypothesis"),
        lambda c: c.update(extra=1),
        lambda c: c.update(threshold=1.5),
        lambda c: c["model"].update({"class": "os.system"}),
        lambda c: c["model"]["params"].update(random_state=1),
        lambda c: c["preprocessing"].update(categorical_encoder="otro"),
    ],
)
def test_invalid_config_is_rejected(change):
    config = load_yaml(CONFIGS[0])
    change(config)

    with pytest.raises(ValueError):
        validate_config(config)
