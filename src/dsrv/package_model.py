"""Arma hf_staging/<version>/ sin reentrenar. Se ejecuta con `python -m dsrv.package_model --version v1`."""

import argparse
import json
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import yaml

from dsrv.config import CONFIGS_DIR, EVIDENCE_DIR, load_yaml, resolve_path
from dsrv.datainfo import DATA_TAG, load_manifest
from dsrv.features import FORBIDDEN_COLUMNS
from dsrv.gitinfo import _git
from dsrv.hashing import sha256_file
from dsrv.mlflowinfo import download_artifact, open_tracking
from dsrv.predict import CONFIG_FILE, MODEL_FILE
from dsrv.metrics import SWEEP_THRESHOLDS
from dsrv.select import POLICY_PATH, SELECTION_JSON, THRESHOLDS_CSV

STAGING_DIR = Path("hf_staging")
SHA_FILE = "model.joblib.sha256"
REQUIREMENTS_FILE = "requirements.txt"
README_FILE = "README.md"
STAGING_FILES = frozenset({MODEL_FILE, SHA_FILE, CONFIG_FILE, REQUIREMENTS_FILE, README_FILE})

HF_REPO = "MiguelAngelAB/unsw-nb15-defiende-servidor"
GITHUB_REPO = "https://github.com/mamezquita/defiende-el-servidor"
LICENSE_LINK = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
PREDICT_PACKAGES = ("scikit-learn", "pandas", "numpy", "joblib")
AUDIT_JSON = EVIDENCE_DIR / "auditoria.json"


def _version_document(selection: dict[str, Any], version: str) -> dict[str, Any]:
    return next(item for item in selection["versiones"] if item["version"] == version)


def pinned_requirements(requirements_text: str) -> str:
    """Toma de requirements.txt las versiones exactas de los paquetes necesarios para predecir."""
    pins = dict(
        line.strip().split("==")
        for line in requirements_text.splitlines()
        if "==" in line and "[" not in line
    )
    return "".join(f"{package}=={pins[package]}\n" for package in PREDICT_PACKAGES)


def build_config(
    document: dict[str, Any], columns: list[str], model_sha256: str, tag_commit: str
) -> dict[str, Any]:
    return {
        "version": document["version"],
        "config_id": document["config_id"],
        "threshold": document["umbral"],
        "decision_rule": "y_pred = 1 si score_attack >= threshold",
        "positive_class": {"label": 1, "name": "ataque"},
        "input_columns": columns,
        "excluded_columns": sorted(FORBIDDEN_COLUMNS),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "git_commit": document["git_commit"],
        "git_tag": document["version"],
        "git_tag_commit": tag_commit,
        "github_commit_url": f"{GITHUB_REPO}/commit/{document['git_commit']}",
        "github_tag_commit_url": f"{GITHUB_REPO}/commit/{tag_commit}",
        "data_version": document["data_version"],
        "mlflow_run_id": document["run_id"],
        "model_sha256": model_sha256,
        "validation_metrics": document["metricas_validacion"],
    }


def _number(value: float, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def _decimal_comma(text: str) -> str:
    return re.sub(r"(\d)\.(\d)", r"\1,\2", text)


def _percent(part: int, total: int) -> str:
    return f"{_number(100 * part / total, 1)} %"


def _count(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _front_matter() -> str:
    metadata = {
        "library_name": "sklearn",
        "pipeline_tag": "tabular-classification",
        "language": "es",
        "tags": ["tabular-classification", "intrusion-detection", "unsw-nb15", "mlops"],
        "license": "other",
        "license_name": "unsw-nb15-academic-use",
        "license_link": LICENSE_LINK,
    }
    return yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)


def _metrics_table(v1: dict[str, Any], v2: dict[str, Any]) -> str:
    rows = [
        ("Recall de ataques", "recall_attack"),
        ("Precision de ataques", "precision_attack"),
        ("FPR", "fpr"),
        ("Macro F1", "macro_f1"),
        ("Accuracy", "accuracy"),
    ]
    lines = ["| Métrica | v1 | v2 |", "|---|---:|---:|"]
    lines.append(f"| Umbral | {_number(v1['umbral'], 2)} | {_number(v2['umbral'], 2)} |")
    for label, key in rows:
        first = _number(v1["metricas_validacion"][key])
        second = _number(v2["metricas_validacion"][key])
        lines.append(f"| {label} | {first} | {second} |")
    return "\n".join(lines)


def _threshold_section(
    document: dict[str, Any], version: str, policy: dict[str, Any]
) -> str:
    if version == "v1":
        return (
            f"Umbral **{_number(document['umbral'], 2)}**, fijado de antemano. No se optimizó: "
            "v1 es la referencia con la que se compara v2."
        )
    limit = policy["restriccion"]
    return (
        f"Umbral **{_number(document['umbral'], 2)}**, elegido en validación con una política "
        "fijada antes de los experimentos (`configs/threshold_policy.yaml`):\n\n"
        f"1. Maximizar `{policy['objetivo']['maximizar']}` con "
        f"`{limit['metrica']} <= {limit['maximo']}`, probando 91 umbrales de 0,05 a 0,95.\n"
        f"2. Desempatar por `{policy['desempate']}`.\n"
        f"3. Si ningún umbral cumple: el de menor `{policy['respaldo']['minimizar']}` con "
        f"recall >= {_number(policy['respaldo']['recall_attack_minimo'], 2)}; si tampoco hay, "
        f"{_number(policy['respaldo']['umbral_por_defecto'], 2)}.\n\n"
        f"{_decimal_comma(document['razon_seleccion'])}"
    )


def _differences_section(version: str, v1: dict[str, Any], v2: dict[str, Any]) -> str:
    comparison = v2["comparacion_con_v1"]
    lines = [
        f"- v1 es `{v1['config_id']}` (regresión logística, umbral {_number(v1['umbral'], 2)}).",
        f"- v2 es `{v2['config_id']}` (gradient boosting, umbral {_number(v2['umbral'], 2)}).",
        "",
        "**Frente a v1, v2 mejora:**",
        "",
        *(f"- {_decimal_comma(item)}" for item in comparison["mejora"]),
        "",
        "**Frente a v1, v2 empeora:**",
        "",
        *(f"- {_decimal_comma(item)}" for item in comparison["empeora"]),
    ]
    if version == "v2":
        lines += ["", "**Errores que acepta v2:**", ""]
        lines += [f"- {_decimal_comma(item)}" for item in v2["errores_aceptados"]]
    return "\n".join(lines)


def _version_limitations(
    version: str, document: dict[str, Any], thresholds: pd.DataFrame
) -> str:
    if version == "v2":
        chosen = thresholds[thresholds["config_id"] == document["config_id"]].iloc[0]
        others = thresholds[thresholds["config_id"] != document["config_id"]]
        runner_up = others.sort_values("recall_attack", ascending=False).iloc[0]
        points = 100 * (chosen["recall_attack"] - runner_up["recall_attack"])
        attacks = int(runner_up["fn"]) - int(chosen["fn"])
        return (
            f"- **FPR cerca del límite:** {_number(100 * document['metricas_validacion']['fpr'], 2)} "
            f"% en validación, y el umbral se eligió entre {len(thresholds)} x "
            f"{len(SWEEP_THRESHOLDS)} combinaciones: en datos nuevos puede superar el 5 %.\n"
            f"- **Ventaja pequeña sobre `{runner_up['config_id']}`:** {_number(points, 1)} puntos "
            f"de recall ({attacks} ataques), sin intervalo de confianza. El criterio decisivo fue "
            "la política; el menor tamaño y la menor inferencia son ventajas secundarias.\n"
        )
    return (
        "- **FPR alta:** v1 es solo una referencia. Con su umbral marca como ataque una de cada "
        f"{round(1 / document['metricas_validacion']['fpr'])} conexiones normales.\n"
    )


def build_model_card(
    version: str,
    selection: dict[str, Any],
    audit: dict[str, Any],
    policy: dict[str, Any],
    training: dict[str, Any],
    config: dict[str, Any],
    thresholds: pd.DataFrame,
) -> str:
    """Redacta la model card con las cifras de las evidencias."""
    document = _version_document(selection, version)
    v1 = _version_document(selection, "v1")
    v2 = _version_document(selection, "v2")
    other = "v2" if version == "v1" else "v1"
    matrix = document["metricas_validacion"]["confusion_matrix"]
    metrics = document["metricas_validacion"]
    sizes = audit["splits"]["tamanos"]
    removed = audit["splits"]["filas_eliminadas"]
    proportions = audit["splits"]["proporcion_clases"]
    test_rows = audit["archivos"]["testing"]["filas"]
    crosses = audit["cruces_particiones"]
    params = ", ".join(f"`{key}={value}`" for key, value in training["model"]["params"].items())
    prep = training["preprocessing"]
    numeric_steps = "imputación con la mediana"
    numeric_steps += ", `log1p`" if prep["log1p"] else ""
    numeric_steps += ", `StandardScaler`" if prep["scale"] else ""
    policy_rules = "\n".join(f"- {rule}" for rule in audit["politica_limpieza"]["reglas"])

    return f"""---
{_front_matter()}---

# Defiende el servidor: {version}

Clasificador binario de conexiones de red: normal (0) o ataque (1), entrenado con UNSW-NB15.

> Esta revisión contiene solo la versión **{version}** (`{document['config_id']}`). La otra versión,
> {other}, está en el tag `{other}` de este repositorio. El tag `{version}` apunta a esta revisión.

## Propósito

Modelo de un proyecto académico de MLOps, cuyo objetivo es la **trazabilidad** de código, datos,
experimentos y modelos, y la recuperación de cada versión en un entorno nuevo. La métrica alta no
es el objetivo y **no está pensado para producción**.

## Datos

- **Dataset:** UNSW-NB15, particiones oficiales de training y testing.
- **Versión:** `{DATA_TAG}`, gestionada con DVC.
- **Entradas:** {len(config['input_columns'])} variables. Quedan fuera `label`, `attack_cat` e `id`.
- **Partición:** train y validación salen solo del archivo de training.

| Partición | Filas | Ataques |
|---|---:|---:|
| train | {_count(sizes['train'])} | {_number(100 * proportions['train']['1'], 1)} % |
| validación | {_count(sizes['val'])} | {_number(100 * proportions['val']['1'], 1)} % |
| excluidas | {_count(sizes['excluded'])} | {_number(100 * proportions['excluded']['1'], 1)} % |

**Política de limpieza:**

{policy_rules}

Se eliminaron {_count(removed['duplicado_exacto'])} duplicados exactos y
{_count(removed['grupo_contradictorio'])} filas de grupos contradictorios.

## Entrenamiento

- **Configuración:** `{document['config_id']}` (`configs/{document['config_id']}.yaml`).
- **Estimador:** `{training['model']['class']}` con {params}, semilla {training['seed']}.
- **Categóricas (`proto`, `service`, `state`):** `{prep['categorical_encoder']}`.
- **Numéricas:** {numeric_steps}.
- Todo el preprocesamiento se ajusta solo con train y va dentro del `Pipeline` de `model.joblib`.

## Métricas de validación

Con el umbral de esta versión ({_number(document['umbral'], 2)}), sobre {_count(sizes['val'])} filas:

| Métrica | Valor |
|---|---:|
| Recall de ataques | {_number(metrics['recall_attack'])} |
| Precision de ataques | {_number(metrics['precision_attack'])} |
| FPR | {_number(metrics['fpr'])} |
| Macro F1 | {_number(metrics['macro_f1'])} |
| Accuracy | {_number(metrics['accuracy'])} |

Matriz de confusión:

| | Predicho normal | Predicho ataque |
|---|---:|---:|
| **Real normal** | {_count(matrix[0][0])} (TN) | {_count(matrix[0][1])} (FP) |
| **Real ataque** | {_count(matrix[1][0])} (FN) | {_count(matrix[1][1])} (TP) |

**Todavía no hay métricas de test:** esta revisión se publicó antes de evaluar el archivo de test.

## Umbral

{_threshold_section(document, version, policy)}

Regla de decisión: `{config['decision_rule']}`.

## Diferencias con la otra versión

{_metrics_table(v1, v2)}

{_differences_section(version, v1, v2)}

## Limitaciones

- **Dataset histórico:** UNSW-NB15 es tráfico generado en 2015 en un laboratorio. No representa el
  tráfico ni los ataques actuales.
- **Solapamiento con test:** el {_percent(crosses['train']['filas_test'], test_rows)} de las filas
  de test tiene entradas idénticas a train y el {_percent(crosses['val']['filas_test'], test_rows)}
  a validación, así que las métricas sobre esas filas pueden ser optimistas.
- **Proporción de clases:** la deduplicación deja train con
  {_number(100 * proportions['train']['1'], 1)} % de ataques, y test tiene
  {_percent(audit['archivos']['testing']['conteo_label']['1'], test_rows)}. La precision depende de
  esa proporción y puede diferir en test.
{_version_limitations(version, document, thresholds)}- **Riesgo de cargar pickles de terceros:** `model.joblib` es un pickle, y cargarlo puede ejecutar
  código arbitrario. Fija la revisión por SHA, comprueba el SHA-256 y cárgalo solo si confías en el
  origen.

## Uso

Requiere Python {config['python_version'].rsplit('.', 1)[0]} y las versiones de `requirements.txt`, más
`huggingface_hub` para descargar. Sustituye `<SHA>` por el SHA completo de esta revisión: es el
commit al que apunta el tag `{version}`.

```python
import hashlib
import json

import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "{HF_REPO}"
REVISION = "<SHA>"

model_path = hf_hub_download(REPO_ID, "{MODEL_FILE}", revision=REVISION)
config_path = hf_hub_download(REPO_ID, "{CONFIG_FILE}", revision=REVISION)
with open(config_path, encoding="utf-8") as file:
    config = json.load(file)

with open(model_path, "rb") as file:
    if hashlib.sha256(file.read()).hexdigest() != config["model_sha256"]:
        raise ValueError("model.joblib no coincide con el SHA-256 esperado")

model = joblib.load(model_path)
X = pd.read_csv("entradas.csv")[config["input_columns"]]
score_attack = model.predict_proba(X)[:, 1]
y_pred = (score_attack >= config["threshold"]).astype(int)
```

## Referencias

- **SHA-256 del modelo:** `{config['model_sha256']}`
- **Commit de entrenamiento:** [`{document['git_commit']}`]({config['github_commit_url']})
- **Tag de código:** [`{version}`]({GITHUB_REPO}/tree/{version}), en el commit `{config['git_tag_commit']}`
- **Tag de datos (DVC):** [`{DATA_TAG}`]({GITHUB_REPO}/tree/{DATA_TAG})
- **`data_version`:** `{document['data_version']}`
- **Run de MLflow:** `{document['run_id']}`
- **Dataset:** [UNSW-NB15]({LICENSE_LINK}). Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive
  data set for network intrusion detection systems. *Military Communications and Information Systems
  Conference (MilCIS)*.
"""


def package(version: str) -> Path:
    """Rehace el staging de la versión con el modelo del run, sin reentrenar."""
    client = open_tracking()
    selection = json.loads(resolve_path(SELECTION_JSON).read_text(encoding="utf-8"))
    document = _version_document(selection, version)
    run = client.get_run(document["run_id"])
    expected_sha = run.data.tags["model_sha256"]
    columns = load_manifest()["variables_predictoras"]

    staging = resolve_path(STAGING_DIR) / version
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = download_artifact(document["run_id"], MODEL_FILE, tmp)
        if sha256_file(downloaded) != expected_sha:
            raise ValueError(f"{MODEL_FILE} no coincide con el tag model_sha256 del run")
        shutil.copy2(downloaded, staging / MODEL_FILE)

    model = joblib.load(staging / MODEL_FILE)
    if list(model.feature_names_in_) != columns:
        raise ValueError("las columnas del modelo no coinciden con variables_predictoras")

    tag_commit = _git("rev-list", "-n", "1", version)
    config = build_config(document, columns, expected_sha, tag_commit)
    text = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
    (staging / CONFIG_FILE).write_text(text + "\n", encoding="utf-8")
    (staging / SHA_FILE).write_text(f"{expected_sha}  {MODEL_FILE}\n", encoding="utf-8")

    requirements = resolve_path(REQUIREMENTS_FILE).read_text(encoding="utf-8")
    (staging / REQUIREMENTS_FILE).write_text(pinned_requirements(requirements), encoding="utf-8")

    audit = json.loads(resolve_path(AUDIT_JSON).read_text(encoding="utf-8"))
    policy = load_yaml(POLICY_PATH)
    training = load_yaml(CONFIGS_DIR / f"{document['config_id']}.yaml")
    thresholds = pd.read_csv(resolve_path(THRESHOLDS_CSV))
    card = build_model_card(version, selection, audit, policy, training, config, thresholds)
    (staging / README_FILE).write_text(card, encoding="utf-8")
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma hf_staging/<version>/ sin reentrenar")
    parser.add_argument("--version", required=True, choices=["v1", "v2"])
    args = parser.parse_args()
    staging = package(args.version)
    print(f"Empaquetado en {staging.relative_to(resolve_path('.'))}")


if __name__ == "__main__":
    main()
