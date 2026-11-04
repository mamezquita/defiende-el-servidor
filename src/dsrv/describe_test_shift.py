"""Describe las diferencias entre test y training. Se ejecuta con `python -m dsrv.describe_test_shift`.

Análisis solo descriptivo, posterior a la evaluación única: no decide ni cambia nada. Escribe
evidencias/desplazamiento-test.json, que respalda las cifras de evidencias/notas-test.md.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from dsrv.audit import SOURCE_CONFIG
from dsrv.config import EVIDENCE_DIR, PREPARED_DIR, TARGET_COLUMN, load_yaml, resolve_path
from dsrv.datainfo import load_manifest
from dsrv.evaluate_test import METRICS_JSON, PREDICTIONS_CSV, load_selection_document
from dsrv.features import CATEGORICAL_COLUMNS
from dsrv.hashing import sha256_dataframe_rows
from dsrv.metrics import METRIC_NAMES, compute_metrics
from dsrv.prepare import read_raw_csv

SHIFT_JSON = EVIDENCE_DIR / "desplazamiento-test.json"
TOP_COLUMNS = 5


def ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    """Estadístico de Kolmogorov-Smirnov: mayor distancia entre dos distribuciones acumuladas."""
    a, b = np.sort(first), np.sort(second)
    points = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, points, side="right") / len(a)
    cdf_b = np.searchsorted(b, points, side="right") / len(b)
    return float(np.abs(cdf_a - cdf_b).max())


def total_variation(first: pd.Series, second: pd.Series) -> float:
    """Variación total entre dos variables categóricas: 0 si son iguales, 1 si no se solapan."""
    p = first.value_counts(normalize=True)
    q = second.value_counts(normalize=True)
    return float(p.sub(q, fill_value=0).abs().sum() / 2)


def _ks_summary(
    reference: pd.DataFrame, control: pd.DataFrame, target: pd.DataFrame, numeric: list[str]
) -> dict[str, Any]:
    rows = [
        {
            "columna": column,
            "ks_control": ks_statistic(reference[column].to_numpy(), control[column].to_numpy()),
            "ks_test": ks_statistic(reference[column].to_numpy(), target[column].to_numpy()),
        }
        for column in numeric
    ]
    table = pd.DataFrame(rows).sort_values("ks_test", ascending=False)
    return {
        "mediana_ks_control": float(table["ks_control"].median()),
        "maximo_ks_control": float(table["ks_control"].max()),
        "mediana_ks_test": float(table["ks_test"].median()),
        "maximo_ks_test": float(table["ks_test"].max()),
        "columnas_con_ks_test_mayor_que_maximo_control": int(
            (table["ks_test"] > table["ks_control"].max()).sum()
        ),
        "columnas_analizadas": len(table),
        "mayores": table.head(TOP_COLUMNS).to_dict(orient="records"),
    }


def _short_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = compute_metrics(frame[TARGET_COLUMN], frame["y_pred"])
    return {
        "filas": len(frame),
        "proporcion_ataques": float(frame[TARGET_COLUMN].mean()),
        **{name: metrics[name] for name in METRIC_NAMES},
    }


def describe(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> dict[str, Any]:
    """Calcula las cifras descriptivas. `test` debe traer y_pred, ya evaluado."""
    numeric = [column for column in features if column not in CATEGORICAL_COLUMNS]
    duplicate = sha256_dataframe_rows(test, features).duplicated(keep="first")
    normal = test[TARGET_COLUMN] == 0
    unique_test = test[~duplicate]

    train_normal = train[train[TARGET_COLUMN] == 0]
    val_normal = val[val[TARGET_COLUMN] == 0]
    train_attack = train[train[TARGET_COLUMN] == 1]
    val_attack = val[val[TARGET_COLUMN] == 1]
    test_normal = unique_test[unique_test[TARGET_COLUMN] == 0]
    test_attack = unique_test[unique_test[TARGET_COLUMN] == 1]

    categorical = {}
    for column in CATEGORICAL_COLUMNS:
        unseen = set(test[column]) - set(train[column])
        categorical[column] = {
            "variacion_total_normales_control": total_variation(
                train_normal[column], val_normal[column]
            ),
            "variacion_total_normales_test": total_variation(
                train_normal[column], test_normal[column]
            ),
            "categorias_de_test_no_vistas_en_train": sorted(unseen),
            "filas_de_test_con_categorias_no_vistas": int(test[column].isin(unseen).sum()),
        }

    by_state = (
        test[normal]
        .groupby("state")["y_pred"]
        .agg(tasa_falsos_positivos="mean", filas="size")
        .sort_values("filas", ascending=False)
        .head(TOP_COLUMNS)
    )
    state_share = {
        state: {
            "proporcion_normales_train": float((train_normal["state"] == state).mean()),
            "proporcion_normales_test": float((test[normal]["state"] == state).mean()),
            "tasa_falsos_positivos_test": float(row["tasa_falsos_positivos"]),
        }
        for state, row in by_state.iterrows()
    }

    return {
        "nota": "Solo descriptivo y posterior a la evaluación única: no se usó para decidir nada.",
        "control": "train frente a validación, que salen del mismo archivo.",
        "duplicados_en_test": {
            "filas_sobrantes": int(duplicate.sum()),
            "proporcion_filas": float(duplicate.mean()),
            "proporcion_entre_normales": float(duplicate[normal].mean()),
            "proporcion_entre_ataques": float(duplicate[~normal].mean()),
        },
        "metricas_test_completo": _short_metrics(test),
        "metricas_test_filas_unicas": _short_metrics(unique_test),
        "desplazamiento_normales_filas_unicas": _ks_summary(
            train_normal, val_normal, test_normal, numeric
        ),
        "desplazamiento_ataques_filas_unicas": _ks_summary(
            train_attack, val_attack, test_attack, numeric
        ),
        "categoricas": categorical,
        "normales_por_state": state_share,
    }


def main() -> None:
    if not resolve_path(METRICS_JSON).exists():
        raise RuntimeError("este análisis solo se ejecuta después de la evaluación única")

    features = load_manifest()["variables_predictoras"]
    train = pd.read_parquet(resolve_path(PREPARED_DIR) / "train.parquet")
    val = pd.read_parquet(resolve_path(PREPARED_DIR) / "val.parquet")
    test = read_raw_csv(load_yaml(SOURCE_CONFIG)["archivos"]["testing"]["ruta"])
    predictions = pd.read_csv(resolve_path(PREDICTIONS_CSV))
    test["y_pred"] = predictions["y_pred"].to_numpy()

    result = describe(train, val, test, features)

    metrics = load_selection_document()["metricas_validacion"]
    prevalence = result["metricas_test_completo"]["proporcion_ataques"]
    true_positives = metrics["recall_attack"] * prevalence
    false_positives = metrics["fpr"] * (1 - prevalence)
    result["precision_esperada_solo_por_proporcion_de_clases"] = true_positives / (
        true_positives + false_positives
    )

    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    resolve_path(SHIFT_JSON).write_text(text + "\n", encoding="utf-8")
    print(f"Escrito {SHIFT_JSON.as_posix()}")


if __name__ == "__main__":
    main()
