"""Evaluación única de v2 en el archivo de test. Se ejecuta con `python -m dsrv.evaluate_test`.

Solo se ejecuta si la selección está congelada y si no existe ya una evaluación.
"""

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from dsrv.audit import SOURCE_CONFIG
from dsrv.config import EVIDENCE_DIR, PREPARED_DIR, TARGET_COLUMN, load_yaml, resolve_path
from dsrv.gitinfo import _git, assert_clean_worktree, get_git_commit
from dsrv.hashing import sha256_dataframe_rows, sha256_file
from dsrv.metrics import METRIC_NAMES, compute_metrics
from dsrv.predict import CONFIG_FILE, MODEL_FILE, load_packaged_model, predict_frame
from dsrv.prepare import SPLITS_NAME, read_raw_csv
from dsrv.select import SELECTION_JSON

FREEZE_TAG = "freeze-v2"
EVALUATED_VERSION = "v2"
METRICS_JSON = EVIDENCE_DIR / "metricas-test.json"
PREDICTIONS_CSV = EVIDENCE_DIR / "predicciones-test.csv"
SEEN_NOTE = "Solo descriptivo: estas cifras no se usaron para decidir nada."


class EvaluationAborted(RuntimeError):
    """No se cumple una condición para evaluar el test."""


def check_preconditions() -> None:
    """Aborta si la evaluación ya existe o si la selección no está congelada."""
    for path in (METRICS_JSON, PREDICTIONS_CSV):
        if resolve_path(path).exists():
            raise EvaluationAborted(f"ya existe {Path(path).as_posix()}: el test se evalúa una vez")
    assert_clean_worktree()
    try:
        _git("merge-base", "--is-ancestor", FREEZE_TAG, "HEAD")
    except subprocess.CalledProcessError as error:
        raise EvaluationAborted(f"{FREEZE_TAG} no existe o no es ancestro de HEAD") from error
    if _git("diff", FREEZE_TAG, "--", Path(SELECTION_JSON).as_posix()):
        raise EvaluationAborted(f"seleccion.json cambió desde {FREEZE_TAG}")


def load_selection_document() -> dict[str, Any]:
    selection = json.loads(resolve_path(SELECTION_JSON).read_text(encoding="utf-8"))
    return next(item for item in selection["versiones"] if item["version"] == EVALUATED_VERSION)


def download_model(document: dict[str, Any], directory: Path) -> tuple[Any, dict[str, Any]]:
    """Descarga la revisión seleccionada y comprueba que sea el modelo y el umbral congelados."""
    for name in (MODEL_FILE, CONFIG_FILE):
        hf_hub_download(
            document["hf_repo"], name, revision=document["hf_sha"], local_dir=str(directory)
        )
    actual = sha256_file(directory / MODEL_FILE)
    if actual != document["model_sha256"]:
        raise EvaluationAborted(f"SHA-256 del modelo {actual}; se esperaba {document['model_sha256']}")

    model, config = load_packaged_model(directory)
    if config["model_sha256"] != document["model_sha256"]:
        raise EvaluationAborted("config.json declara otro SHA-256 de modelo")
    if config["threshold"] != document["umbral"]:
        raise EvaluationAborted("config.json declara otro umbral")
    return model, config


def read_test_file() -> tuple[pd.DataFrame, str]:
    """Lee el test completo, sin deduplicar ni filtrar, tras comprobar su SHA-256."""
    expected = load_yaml(SOURCE_CONFIG)["archivos"]["testing"]
    digest = sha256_file(resolve_path(expected["ruta"]))
    if digest != expected["sha256"]:
        raise EvaluationAborted(f"SHA-256 del test {digest}; se esperaba {expected['sha256']}")

    frame = read_raw_csv(expected["ruta"])
    if len(frame) != expected["filas"]:
        raise EvaluationAborted(f"el test tiene {len(frame)} filas; se esperaban {expected['filas']}")
    frame.insert(0, "row_id", [f"test:{position}" for position in range(len(frame))])
    return frame, digest


def seen_masks(frame: pd.DataFrame, features: list[str]) -> dict[str, pd.Series]:
    """Marca las filas de test cuyas entradas aparecen en train, en val o en ninguna."""
    splits = pd.read_csv(
        resolve_path(PREPARED_DIR) / SPLITS_NAME,
        usecols=["group_id", "split"],
        keep_default_na=False,
    )
    hashes = sha256_dataframe_rows(frame, features)
    in_train = hashes.isin(set(splits.loc[splits["split"] == "train", "group_id"]))
    in_val = hashes.isin(set(splits.loc[splits["split"] == "val", "group_id"]))
    return {
        "vistas_en_train_o_val": in_train | in_val,
        "vistas_en_train": in_train,
        "vistas_en_val": in_val,
        "no_vistas": ~(in_train | in_val),
    }


def metrics_section(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_pred)
    return {
        **{name: metrics[name] for name in METRIC_NAMES},
        "matriz_confusion": metrics["confusion_matrix"],
        "orden_matriz": "[[TN, FP], [FN, TP]]",
        "zero_division": metrics["zero_division"],
    }


def descriptive_sections(
    y_true: pd.Series, y_pred: pd.Series, masks: dict[str, pd.Series]
) -> dict[str, Any]:
    """Métricas por grupo de filas. Devuelve solo cifras que no intervienen en ninguna decisión."""
    sections: dict[str, Any] = {"nota": SEEN_NOTE}
    for name, mask in masks.items():
        sections[name] = {
            "filas": int(mask.sum()),
            **metrics_section(y_true[mask], y_pred[mask]),
        }
    return sections


def build_report(
    document: dict[str, Any],
    config: dict[str, Any],
    dataset_sha256: str,
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    git_commit: str,
    date: str,
) -> dict[str, Any]:
    y_true = frame[TARGET_COLUMN]
    y_pred = predictions["y_pred"]
    masks = seen_masks(frame, config["input_columns"])
    return {
        "version": document["version"],
        "hf_repo": document["hf_repo"],
        "hf_sha": document["hf_sha"],
        "model_sha256": document["model_sha256"],
        "dataset_sha256": dataset_sha256,
        "filas_evaluadas": len(frame),
        "umbral": config["threshold"],
        "clase_positiva": config["positive_class"],
        "regla_decision": config["decision_rule"],
        **metrics_section(y_true, y_pred),
        "git_commit": git_commit,
        "freeze_tag": FREEZE_TAG,
        "fecha": date,
        "descriptivo_filas_vistas": descriptive_sections(y_true, y_pred, masks),
    }


def write_evidence(frame: pd.DataFrame, predictions: pd.DataFrame, report: dict[str, Any]) -> None:
    """Escribe las predicciones y, al final, las métricas, que marcan la evaluación como hecha."""
    output = pd.DataFrame(
        {
            "row_id": predictions["row_id"],
            "id": frame["id"].to_numpy(),
            "y_true": frame[TARGET_COLUMN].to_numpy(),
            "score_attack": predictions["score_attack"],
            "y_pred": predictions["y_pred"],
        }
    )
    output.to_csv(resolve_path(PREDICTIONS_CSV), index=False, lineterminator="\n")
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    resolve_path(METRICS_JSON).write_text(text + "\n", encoding="utf-8")


def main() -> None:
    check_preconditions()
    document = load_selection_document()
    git_commit = get_git_commit()

    with tempfile.TemporaryDirectory() as tmp:
        model, config = download_model(document, Path(tmp))

    frame, dataset_sha256 = read_test_file()
    predictions = predict_frame(model, config, frame)
    date = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = build_report(document, config, dataset_sha256, frame, predictions, git_commit, date)
    write_evidence(frame, predictions, report)
    print(f"Evaluadas {len(frame)} filas; escrito {METRICS_JSON.as_posix()}")


if __name__ == "__main__":
    main()
