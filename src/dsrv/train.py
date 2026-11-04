"""Entrena una configuración y la registra en MLflow.

Se ejecuta con `python -m dsrv.train --config configs/e1_lr_base.yaml`. No lee el archivo de test.
"""

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from threadpoolctl import threadpool_limits

from dsrv.config import PREPARED_DIR, REPO_ROOT, TARGET_COLUMN, load_yaml, resolve_path
from dsrv.datainfo import assert_data_matches_version, get_data_version, load_manifest
from dsrv.features import build_pipeline, use_single_thread_inference, validate_config
from dsrv.gitinfo import assert_clean_worktree, get_git_commit
from dsrv.hashing import sha256_file
from dsrv.metrics import METRIC_NAMES, compute_metrics, predict_at_threshold, threshold_sweep

EXPERIMENT_NAME = "defiende-el-servidor"
TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACT_LOCATION = "./mlartifacts"
MODELS_DIR = Path("models")
REQUIREMENTS_FILE = Path("requirements.txt")
POSITIVE_CLASS = 1

# Tipos que skops no considera seguros por defecto y que E1 a E4 necesitan para cargarse.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]

MATRIX_COLORS = LinearSegmentedColormap.from_list("matriz", ["#cde2fb", "#0d366b"])
TEXT_ON_LIGHT = "#0b0b0b"
TEXT_ON_DARK = "#fcfcfb"
FIGURE_SURFACE = "#fcfcfb"


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Aplana el YAML con claves separadas por punto, como model.params.C."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def write_confusion_png(matrix: list[list[int]], config_id: str, threshold: float, path: Path) -> None:
    """Dibuja la matriz con su valor en cada celda, para no depender solo del color."""
    values = np.array(matrix)
    fig = Figure(figsize=(5, 4.4), facecolor=FIGURE_SURFACE, layout="constrained")
    ax = fig.add_subplot()
    ax.imshow(values, cmap=MATRIX_COLORS, vmin=0, vmax=max(int(values.max()), 1))

    names = [["TN", "FP"], ["FN", "TP"]]
    midpoint = values.max() / 2
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{names[row][col]}\n{values[row, col]:,}".replace(",", "."),
                ha="center",
                va="center",
                fontsize=14,
                color=TEXT_ON_DARK if values[row, col] > midpoint else TEXT_ON_LIGHT,
            )

    labels = ["Normal (0)", "Ataque (1)"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicción", color=TEXT_ON_LIGHT)
    ax.set_ylabel("Valor real", color=TEXT_ON_LIGHT)
    ax.tick_params(length=0, colors=TEXT_ON_LIGHT)
    ax.spines[:].set_visible(False)
    ax.set_title(f"{config_id}: validación, umbral {threshold}", color=TEXT_ON_LIGHT, fontsize=12)
    fig.savefig(path, dpi=150)


def _score_attack(pipeline: Any, X: pd.DataFrame) -> np.ndarray:
    positive = list(pipeline.classes_).index(POSITIVE_CLASS)
    return pipeline.predict_proba(X)[:, positive]


def _log_results(
    config: dict[str, Any],
    config_path: str | Path,
    pipeline: Any,
    val: pd.DataFrame,
    X_val: pd.DataFrame,
    score: np.ndarray,
    metrics: dict[str, Any],
    workdir: Path,
) -> Path:
    """Registra los artefactos del run activo y devuelve la ruta de model.joblib."""
    mlflow.log_artifact(str(resolve_path(config_path)))

    matrix_json = workdir / "confusion_matrix.json"
    matrix_json.write_text(
        json.dumps(
            {
                "orden": "[[TN, FP], [FN, TP]]",
                "threshold": config["threshold"],
                "matriz": metrics["confusion_matrix"],
                "metricas": {name: metrics[name] for name in METRIC_NAMES},
                "zero_division": metrics["zero_division"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_png = workdir / "confusion_matrix.png"
    write_confusion_png(
        metrics["confusion_matrix"], config["config_id"], config["threshold"], matrix_png
    )

    scores_path = workdir / "val_scores.parquet"
    pd.DataFrame(
        {
            "row_id": val["row_id"].to_numpy(),
            "y_true": val[TARGET_COLUMN].to_numpy(),
            "score_attack": score,
        }
    ).to_parquet(scores_path, index=False)

    sweep_path = workdir / "threshold_sweep.csv"
    threshold_sweep(val[TARGET_COLUMN], score).to_csv(sweep_path, index=False, lineterminator="\n")

    model_path = workdir / "model.joblib"
    joblib.dump(pipeline, model_path)
    mlflow.set_tag("model_sha256", sha256_file(model_path))

    for path in (matrix_json, matrix_png, scores_path, sweep_path, model_path):
        mlflow.log_artifact(str(path))

    sample = X_val.head(100)
    model_info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        signature=infer_signature(sample, pipeline.predict_proba(sample)),
        pip_requirements=str(resolve_path(REQUIREMENTS_FILE)),
        pyfunc_predict_fn="predict_proba",
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )

    reloaded = mlflow.sklearn.load_model(model_info.model_uri)
    if not np.array_equal(_score_attack(reloaded, X_val), score):
        raise RuntimeError("el modelo registrado no reproduce los scores de validación")
    return model_path


def run_training(config_path: str | Path) -> str:
    """Entrena la configuración, registra el run y devuelve su run_id."""
    os.chdir(REPO_ROOT)
    assert_clean_worktree()

    config = load_yaml(config_path)
    validate_config(config)
    git_commit = get_git_commit()
    assert_data_matches_version()
    data_version = get_data_version()

    features = load_manifest()["variables_predictoras"]
    train = pd.read_parquet(resolve_path(PREPARED_DIR) / "train.parquet")
    val = pd.read_parquet(resolve_path(PREPARED_DIR) / "val.parquet")
    X_train, y_train = train[features], train[TARGET_COLUMN]
    X_val, y_val = val[features], val[TARGET_COLUMN]

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=config["config_id"]) as run:
        try:
            mlflow.set_tags(
                {
                    "hypothesis": config["hypothesis"],
                    "change": config["change"],
                    "family": config["family"],
                    "config_id": config["config_id"],
                    "git_commit": git_commit,
                    "data_version": data_version,
                    "seed": config["seed"],
                }
            )
            mlflow.log_params(flatten_config(config))

            pipeline = build_pipeline(config, X_train)
            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start
            use_single_thread_inference(pipeline)

            # Un solo hilo en todos los modelos, para que el tiempo de inferencia sea comparable.
            with threadpool_limits(limits=1):
                start = time.perf_counter()
                score = _score_attack(pipeline, X_val)
                predict_seconds = time.perf_counter() - start

            metrics = compute_metrics(y_val, predict_at_threshold(score, config["threshold"]))
            mlflow.log_metrics(
                {
                    **{name: metrics[name] for name in METRIC_NAMES},
                    "train_seconds": train_seconds,
                    "val_predict_seconds": predict_seconds,
                    "zero_division": float(any(metrics["zero_division"].values())),
                }
            )
            with tempfile.TemporaryDirectory() as tmp:
                model_path = _log_results(
                    config, config_path, pipeline, val, X_val, score, metrics, Path(tmp)
                )
                model_dir = resolve_path(MODELS_DIR) / config["config_id"]
                model_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_path, model_dir / "model.joblib")
                (model_dir / "run_id.txt").write_text(run.info.run_id + "\n", encoding="utf-8")
        except Exception:
            mlflow.set_tag("status", "failed")
            raise
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una configuración y la registra en MLflow")
    parser.add_argument("--config", required=True, help="Ruta del YAML, relativa a la raíz del repo")
    args = parser.parse_args()
    print(f"run_id: {run_training(args.config)}")


if __name__ == "__main__":
    main()
