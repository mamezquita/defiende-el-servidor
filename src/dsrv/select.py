"""Aplica la política de umbral en validación y elige v1 y v2. Se ejecuta con `python -m dsrv.select`.

Solo usa val_scores: el archivo de test no interviene en la selección.
"""

import json
import tempfile
from typing import Any

import pandas as pd
from mlflow.entities import Run

from dsrv.config import CONFIGS_DIR, EVIDENCE_DIR, load_yaml, resolve_path
from dsrv.metrics import METRIC_NAMES, threshold_sweep
from dsrv.mlflowinfo import download_artifact, latest_finished_runs, open_tracking

POLICY_PATH = CONFIGS_DIR / "threshold_policy.yaml"
THRESHOLDS_CSV = EVIDENCE_DIR / "umbrales.csv"
SELECTION_JSON = EVIDENCE_DIR / "seleccion.json"

RULE_CONSTRAINT = "restriccion"
RULE_FALLBACK = "respaldo_menor_fpr"
RULE_DEFAULT = "respaldo_umbral_por_defecto"
RULE_PRIORITY = (RULE_CONSTRAINT, RULE_FALLBACK, RULE_DEFAULT)

V1_CONFIG = "e1_lr_base"
V1_THRESHOLD = 0.5

HIGHER_IS_BETTER = ("recall_attack", "precision_attack", "macro_f1", "accuracy")


def _sort_keys(rule: str, policy: dict[str, Any]) -> tuple[list[str], list[bool]]:
    """Columnas y sentido de orden con que la política prefiere un resultado sobre otro."""
    tie_break = policy["desempate"]
    if rule == RULE_FALLBACK:
        return [policy["respaldo"]["minimizar"], tie_break], [True, False]
    return [policy["objetivo"]["maximizar"], tie_break], [False, False]


def apply_policy(sweep: pd.DataFrame, policy: dict[str, Any]) -> tuple[pd.Series, str]:
    """Elige el umbral de un barrido según la política y devuelve su fila y la regla aplicada."""
    constraint = policy["restriccion"]
    fallback = policy["respaldo"]

    eligible = sweep[sweep[constraint["metrica"]] <= constraint["maximo"]]
    rule = RULE_CONSTRAINT
    if eligible.empty:
        eligible = sweep[sweep["recall_attack"] >= fallback["recall_attack_minimo"]]
        rule = RULE_FALLBACK
    if eligible.empty:
        row = sweep[sweep["threshold"] == fallback["umbral_por_defecto"]].iloc[0]
        return row, RULE_DEFAULT

    columns, ascending = _sort_keys(rule, policy)
    ordered = eligible.sort_values([*columns, "threshold"], ascending=[*ascending, True])
    return ordered.iloc[0], rule


def choose_best(table: pd.DataFrame, policy: dict[str, Any]) -> pd.Series:
    """Elige la mejor fila de umbrales.csv: primero la regla de mayor prioridad, luego la política."""
    best_rule = next(rule for rule in RULE_PRIORITY if (table["regla"] == rule).any())
    columns, ascending = _sort_keys(best_rule, policy)
    candidates = table[table["regla"] == best_rule]
    return candidates.sort_values(columns, ascending=ascending, kind="stable").iloc[0]


def _sweep_for_run(run: Run) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path = download_artifact(run.info.run_id, "val_scores.parquet", tmp)
        scores = pd.read_parquet(path)
    return threshold_sweep(scores["y_true"], scores["score_attack"])


def _row_metrics(row: pd.Series) -> dict[str, Any]:
    return {
        **{name: float(row[name]) for name in METRIC_NAMES},
        "confusion_matrix": [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]],
    }


def _version_document(version: str, run: Run, threshold: float, row: pd.Series, reason: str):
    return {
        "version": version,
        "config_id": run.data.tags["config_id"],
        "run_id": run.info.run_id,
        "git_commit": run.data.tags["git_commit"],
        "data_version": run.data.tags["data_version"],
        "umbral": threshold,
        "metricas_validacion": _row_metrics(row),
        "razon_seleccion": reason,
        "hf_repo": "",
        "hf_sha": "",
        "model_sha256": "",
    }


def _selection_reason(best: pd.Series, policy: dict[str, Any]) -> str:
    objective = policy["objetivo"]["maximizar"]
    constraint = policy["restriccion"]
    limit = f"{constraint['metrica']} <= {constraint['maximo']}"
    if best["regla"] == RULE_CONSTRAINT:
        return (
            f"Mayor {objective} ({best[objective]:.4f}) entre las configuraciones que cumplen "
            f"{limit} en validación, con su mejor umbral."
        )
    if best["regla"] == RULE_FALLBACK:
        minimum = policy["respaldo"]["recall_attack_minimo"]
        return (
            f"Ninguna configuración cumple {limit}. Se aplica el respaldo: menor fpr "
            f"({best['fpr']:.4f}) con recall_attack >= {minimum}."
        )
    return f"Ninguna configuración cumple {limit} ni el respaldo: se usa el umbral por defecto."


def compare_with_v1(v1: pd.Series, v2: pd.Series) -> dict[str, list[str]]:
    """Clasifica cada métrica según mejore o empeore frente a v1, con su diferencia."""
    comparison: dict[str, list[str]] = {"mejora": [], "empeora": [], "igual": []}
    for name in METRIC_NAMES:
        difference = float(v2[name]) - float(v1[name])
        text = f"{name}: {float(v1[name]):.4f} -> {float(v2[name]):.4f} ({difference:+.4f})"
        better = difference > 0 if name in HIGHER_IS_BETTER else difference < 0
        key = "igual" if difference == 0 else "mejora" if better else "empeora"
        comparison[key].append(text)
    for name, label in (("fp", "falsos positivos"), ("fn", "falsos negativos")):
        difference = int(v2[name]) - int(v1[name])
        text = f"{label}: {int(v1[name])} -> {int(v2[name])} ({difference:+d})"
        key = "igual" if difference == 0 else "mejora" if difference < 0 else "empeora"
        comparison[key].append(text)
    return comparison


def accepted_errors(row: pd.Series) -> list[str]:
    attacks = int(row["fn"]) + int(row["tp"])
    normals = int(row["fp"]) + int(row["tn"])
    return [
        f"{int(row['fn'])} ataques no detectados en validación ({int(row['fn']) / attacks:.2%} "
        "de los ataques): pasan como tráfico normal.",
        f"{int(row['fp'])} falsas alarmas en validación ({int(row['fp']) / normals:.2%} del "
        "tráfico normal): conexiones normales marcadas como ataque.",
    ]


def _assert_not_published() -> None:
    """Evita vaciar hf_repo, hf_sha y model_sha256 de una selección ya publicada."""
    path = resolve_path(SELECTION_JSON)
    if not path.exists():
        return
    selection = json.loads(path.read_text(encoding="utf-8"))
    if any(item["hf_sha"] for item in selection["versiones"]):
        raise RuntimeError(f"{SELECTION_JSON.as_posix()} ya está publicada; no se sobrescribe")


def main() -> None:
    _assert_not_published()
    client = open_tracking()
    policy = load_yaml(POLICY_PATH)
    runs = latest_finished_runs(client)
    sweeps = {config_id: _sweep_for_run(run) for config_id, run in runs.items()}

    rows = []
    for config_id, sweep in sweeps.items():
        row, rule = apply_policy(sweep, policy)
        rows.append(
            {
                "config_id": config_id,
                "run_id": runs[config_id].info.run_id,
                "umbral": row["threshold"],
                **{name: row[name] for name in METRIC_NAMES},
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
                "zero_division": row["zero_division"],
                "regla": rule,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(resolve_path(THRESHOLDS_CSV), index=False, lineterminator="\n")

    best = choose_best(table, policy)
    v2_sweep = sweeps[best["config_id"]]
    v2_row = v2_sweep[v2_sweep["threshold"] == best["umbral"]].iloc[0]
    v1_sweep = sweeps[V1_CONFIG]
    v1_row = v1_sweep[v1_sweep["threshold"] == V1_THRESHOLD].iloc[0]

    selection = {
        "estado": "borrador",
        "politica": POLICY_PATH.as_posix(),
        "conjunto_de_seleccion": "validacion",
        "versiones": [
            _version_document(
                "v1",
                runs[V1_CONFIG],
                V1_THRESHOLD,
                v1_row,
                f"Referencia fijada de antemano: {V1_CONFIG} con umbral {V1_THRESHOLD}.",
            ),
            {
                **_version_document(
                    "v2",
                    runs[best["config_id"]],
                    float(best["umbral"]),
                    v2_row,
                    _selection_reason(best, policy),
                ),
                "regla_aplicada": best["regla"],
                "comparacion_con_v1": compare_with_v1(v1_row, v2_row),
                "errores_aceptados": accepted_errors(v2_row),
            },
        ],
    }
    text = json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False)
    resolve_path(SELECTION_JSON).write_text(text + "\n", encoding="utf-8")
    print(f"v2 = {best['config_id']} con umbral {best['umbral']} ({best['regla']})")


if __name__ == "__main__":
    main()
