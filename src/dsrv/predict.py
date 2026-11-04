"""Predice con un modelo empaquetado.

Se ejecuta con `python -m dsrv.predict --model-dir <dir> --input <csv> --output <csv>`.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from dsrv.config import resolve_path
from dsrv.hashing import sha256_file
from dsrv.metrics import predict_at_threshold

MODEL_FILE = "model.joblib"
CONFIG_FILE = "config.json"


def load_packaged_model(model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Carga el modelo tras comprobar su SHA-256, porque cargar un pickle puede ejecutar código."""
    directory = resolve_path(model_dir)
    config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))

    actual = sha256_file(directory / MODEL_FILE)
    if actual != config["model_sha256"]:
        raise ValueError(
            f"{MODEL_FILE}: el SHA-256 {actual} no coincide con {config['model_sha256']}"
        )
    if sklearn.__version__ != config["sklearn_version"]:
        raise RuntimeError(
            f"scikit-learn {sklearn.__version__} instalado; el modelo requiere "
            f"{config['sklearn_version']}"
        )
    return joblib.load(directory / MODEL_FILE), config


def predict_frame(model: Any, config: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Devuelve row_id, score_attack e y_pred. Ignora las columnas que no son de entrada."""
    columns = config["input_columns"]
    missing = [column for column in ["row_id", *columns] if column not in frame.columns]
    if missing:
        raise ValueError(f"faltan columnas de entrada: {missing}")

    positive = list(model.classes_).index(config["positive_class"]["label"])
    score: np.ndarray = model.predict_proba(frame[columns])[:, positive]
    return pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "score_attack": score,
            "y_pred": predict_at_threshold(score, config["threshold"]),
        }
    )


def predict_file(model_dir: str | Path, input_csv: str | Path, output_csv: str | Path) -> None:
    model, config = load_packaged_model(model_dir)
    frame = pd.read_csv(resolve_path(input_csv))
    predictions = predict_frame(model, config, frame)
    predictions.to_csv(resolve_path(output_csv), index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice con un modelo empaquetado")
    parser.add_argument("--model-dir", required=True, help="Carpeta con model.joblib y config.json")
    parser.add_argument("--input", required=True, help="CSV con row_id y las columnas de entrada")
    parser.add_argument("--output", required=True, help="CSV de salida")
    args = parser.parse_args()
    predict_file(args.model_dir, args.input, args.output)
    print(f"Escrito {args.output}")


if __name__ == "__main__":
    main()
