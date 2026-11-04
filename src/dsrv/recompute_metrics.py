"""Recalcula las métricas desde predicciones-test.csv. Se ejecuta con `python -m dsrv.recompute_metrics`.

Termina con error si algún valor difiere de metricas-test.json. No carga el modelo.
"""

import json
import sys
from typing import Any

import pandas as pd

from dsrv.audit import SOURCE_CONFIG
from dsrv.config import TARGET_COLUMN, load_yaml, resolve_path
from dsrv.datainfo import load_manifest
from dsrv.evaluate_test import (
    METRICS_JSON,
    PREDICTIONS_CSV,
    descriptive_sections,
    metrics_section,
    seen_masks,
)
from dsrv.metrics import predict_at_threshold
from dsrv.prepare import read_raw_csv

SUMMARY_FIELDS = (
    "recall_attack",
    "precision_attack",
    "fpr",
    "macro_f1",
    "accuracy",
    "matriz_confusion",
    "orden_matriz",
    "zero_division",
)


def find_differences(
    predictions: pd.DataFrame, report: dict[str, Any], masks: dict[str, pd.Series] | None = None
) -> list[str]:
    """Devuelve una descripción de cada valor que no coincide exactamente con el reporte."""
    differences = []
    expected_ids = [f"test:{position}" for position in range(len(predictions))]
    if predictions["row_id"].tolist() != expected_ids:
        differences.append("row_id no es la secuencia test:0, test:1, ...")
    if len(predictions) != report["filas_evaluadas"]:
        differences.append(
            f"filas_evaluadas: {len(predictions)} frente a {report['filas_evaluadas']}"
        )

    decisions = predict_at_threshold(predictions["score_attack"], report["umbral"])
    if (decisions != predictions["y_pred"].to_numpy()).any():
        differences.append("y_pred no coincide con score_attack >= umbral")

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    recomputed = metrics_section(y_true, y_pred)
    for field in SUMMARY_FIELDS:
        if recomputed[field] != report[field]:
            differences.append(f"{field}: {recomputed[field]} frente a {report[field]}")

    if masks is not None:
        sections = descriptive_sections(y_true, y_pred, masks)
        if sections != report["descriptivo_filas_vistas"]:
            differences.append("descriptivo_filas_vistas no coincide")
    return differences


def main() -> int:
    report = json.loads(resolve_path(METRICS_JSON).read_text(encoding="utf-8"))
    predictions = pd.read_csv(resolve_path(PREDICTIONS_CSV))

    testing = load_yaml(SOURCE_CONFIG)["archivos"]["testing"]
    frame = read_raw_csv(testing["ruta"])
    if (frame[TARGET_COLUMN].to_numpy() != predictions["y_true"].to_numpy()).any():
        print("y_true no coincide con el archivo de test", file=sys.stderr)
        return 1
    if (frame["id"].to_numpy() != predictions["id"].to_numpy()).any():
        print("id no coincide con el archivo de test", file=sys.stderr)
        return 1
    masks = seen_masks(frame, load_manifest()["variables_predictoras"])

    differences = find_differences(predictions, report, masks)
    for difference in differences:
        print(f"DIFERENCIA: {difference}", file=sys.stderr)
    if differences:
        return 1
    print(f"Las {len(predictions)} predicciones reproducen exactamente {METRICS_JSON.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
