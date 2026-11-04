"""Construcción del Pipeline completo: preprocesamiento + estimador, ajustado solo con train."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from dsrv.config import EXCLUDED_COLUMNS

CATEGORICAL_COLUMNS = ("proto", "service", "state")
FORBIDDEN_COLUMNS = frozenset({*EXCLUDED_COLUMNS, "row_id", "group_id"})

ESTIMATORS = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
}
CATEGORICAL_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
}

CONFIG_FIELDS = {
    "config_id",
    "family",
    "hypothesis",
    "change",
    "seed",
    "preprocessing",
    "model",
    "threshold",
}
PREPROCESSING_FIELDS = {"categorical_encoder", "log1p", "scale"}
MODEL_FIELDS = {"class", "params"}


def _check_fields(section: dict[str, Any], expected: set[str], name: str) -> None:
    if set(section) != expected:
        missing = sorted(expected - set(section))
        unknown = sorted(set(section) - expected)
        raise ValueError(f"{name}: campos faltantes {missing}, campos desconocidos {unknown}")


def validate_config(config: dict[str, Any]) -> None:
    """Lanza ValueError si la configuración no cumple el esquema común de los experimentos."""
    _check_fields(config, CONFIG_FIELDS, "config")
    _check_fields(config["preprocessing"], PREPROCESSING_FIELDS, "preprocessing")
    _check_fields(config["model"], MODEL_FIELDS, "model")

    if config["preprocessing"]["categorical_encoder"] not in CATEGORICAL_ENCODERS:
        raise ValueError(f"encoder no permitido: {config['preprocessing']['categorical_encoder']}")
    if config["model"]["class"] not in ESTIMATORS:
        raise ValueError(f"estimador no permitido: {config['model']['class']}")
    if "random_state" in config["model"]["params"]:
        raise ValueError("random_state sale de seed; no debe ir en model.params")
    if not 0 <= config["threshold"] <= 1:
        raise ValueError(f"threshold fuera de [0, 1]: {config['threshold']}")


def _numeric_steps(log1p: bool, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if log1p:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_pipeline(config: dict[str, Any], X_train: pd.DataFrame) -> Pipeline:
    """Construye el Pipeline sin ajustarlo. Las columnas salen de X_train, que debe ser train."""
    validate_config(config)
    forbidden = FORBIDDEN_COLUMNS & set(X_train.columns)
    if forbidden:
        raise ValueError(f"columnas prohibidas en las entradas: {sorted(forbidden)}")
    missing = set(CATEGORICAL_COLUMNS) - set(X_train.columns)
    if missing:
        raise ValueError(f"faltan columnas categóricas: {sorted(missing)}")

    preprocessing = config["preprocessing"]
    numeric = [col for col in X_train.columns if col not in CATEGORICAL_COLUMNS]
    log_columns = (
        [col for col in numeric if X_train[col].min() >= 0] if preprocessing["log1p"] else []
    )
    plain_columns = [col for col in numeric if col not in log_columns]

    encoder = CATEGORICAL_ENCODERS[preprocessing["categorical_encoder"]]()
    transformers = [("categorical", encoder, list(CATEGORICAL_COLUMNS))]
    if log_columns:
        steps = _numeric_steps(log1p=True, scale=preprocessing["scale"])
        transformers.append(("numeric_log1p", steps, log_columns))
    if plain_columns:
        steps = _numeric_steps(log1p=False, scale=preprocessing["scale"])
        transformers.append(("numeric", steps, plain_columns))

    estimator = ESTIMATORS[config["model"]["class"]](
        **config["model"]["params"], random_state=config["seed"]
    )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("model", estimator),
        ]
    )


def use_single_thread_inference(pipeline: Pipeline) -> None:
    """Fija n_jobs=1 en el estimador ya entrenado.

    Con n_jobs=-1, RandomForest suma las probabilidades en el orden en que terminan los hilos, y
    los scores de una misma entrada varían por redondeo. Con un hilo son siempre idénticos.
    """
    if pipeline.named_steps["model"].get_params().get("n_jobs") is not None:
        pipeline.set_params(model__n_jobs=1)
