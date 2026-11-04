"""Añade la evaluación en test a la model card de v2. Se ejecuta con `python -m dsrv.document_test_hf`.

Sube solo README.md: el modelo, el tag v2 y la revisión evaluada no cambian.
"""

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from dsrv.config import load_yaml, resolve_path
from dsrv.evaluate_test import EVALUATED_VERSION, METRICS_JSON
from dsrv.hashing import sha256_file
from dsrv.package_model import (
    GITHUB_REPO,
    HF_REPO,
    README_FILE,
    STAGING_FILES,
    _count,
    _number,
)
from dsrv.predict import CONFIG_FILE, MODEL_FILE
from dsrv.publish_hf import (
    FORBIDDEN_CONTENT,
    PRESERVED_FILE,
    REVISIONS_JSON,
    PublishError,
    _read_json,
    _write_json,
)
from dsrv.select import POLICY_PATH

SECTION_TITLE = "## Evaluación final en test"
PENDING_NOTICE = (
    "**Todavía no hay métricas de test:** esta revisión se publicó antes de evaluar el archivo de "
    "test."
)
PUBLISHED_NOTICE = "Las métricas de test están en la sección «Evaluación final en test»."
INSERT_BEFORE = "## Umbral"


def _metric_row(label: str, key: str, validation: dict[str, Any], report: dict[str, Any]) -> str:
    return f"| {label} | {_number(validation[key])} | {_number(report[key])} |"


def _group_row(label: str, group: dict[str, Any]) -> str:
    return (
        f"| {label} | {_count(group['filas'])} | {_number(group['recall_attack'])} | "
        f"{_number(group['precision_attack'])} | {_number(group['fpr'])} | "
        f"{_number(group['macro_f1'])} |"
    )


def _limit_notice(report: dict[str, Any], limit: float) -> str:
    """Avisa de forma explícita si la FPR de test supera el límite de la política."""
    if report["fpr"] <= limit:
        return ""
    return (
        f"**La FPR en test ({_number(100 * report['fpr'], 2)} %) supera el límite de "
        f"{_number(100 * limit, 0)} % con el que se eligió el umbral en validación.** No se cambió "
        "ni el modelo ni el umbral. Las causas medidas están en `evidencias/notas-test.md`.\n\n"
    )


def build_section(report: dict[str, Any], validation: dict[str, Any], limit: float) -> str:
    """Redacta la sección de test con las cifras de metricas-test.json."""
    matrix = report["matriz_confusion"]
    groups = report["descriptivo_filas_vistas"]
    return f"""{SECTION_TITLE}

Evaluación única del archivo de test completo ({_count(report['filas_evaluadas'])} filas, sin
deduplicar ni filtrar), hecha después de congelar la selección con el tag
[`{report['freeze_tag']}`]({GITHUB_REPO}/tree/{report['freeze_tag']}). El test no se usó para elegir
el modelo ni el umbral, y nada cambió después de ver estos resultados.

{_limit_notice(report, limit)}- **Revisión evaluada:** `{report['hf_sha']}`
- **SHA-256 del modelo:** `{report['model_sha256']}`
- **SHA-256 del archivo de test:** `{report['dataset_sha256']}`
- **Umbral:** {_number(report['umbral'], 2)}
- **Código de la evaluación:** [`{report['git_commit']}`]({GITHUB_REPO}/commit/{report['git_commit']})

Esta revisión de la model card solo cambia `README.md`: su `model.joblib` es idéntico al evaluado.

| Métrica | Validación | Test |
|---|---:|---:|
{_metric_row('Recall de ataques', 'recall_attack', validation, report)}
{_metric_row('Precision de ataques', 'precision_attack', validation, report)}
{_metric_row('FPR', 'fpr', validation, report)}
{_metric_row('Macro F1', 'macro_f1', validation, report)}
{_metric_row('Accuracy', 'accuracy', validation, report)}

Matriz de confusión en test:

| | Predicho normal | Predicho ataque |
|---|---:|---:|
| **Real normal** | {_count(matrix[0][0])} (TN) | {_count(matrix[0][1])} (FP) |
| **Real ataque** | {_count(matrix[1][0])} (FN) | {_count(matrix[1][1])} (TP) |

Por grupo de filas de test. Es solo descriptivo: estas cifras no se usaron para decidir nada.

| Filas de test | Cantidad | Recall | Precision | FPR | Macro F1 |
|---|---:|---:|---:|---:|---:|
{_group_row('Con entradas vistas en train o validación', groups['vistas_en_train_o_val'])}
{_group_row('Con entradas no vistas', groups['no_vistas'])}

Detalle en `evidencias/metricas-test.json` y `evidencias/predicciones-test.csv` del
[repositorio de código]({GITHUB_REPO}).

"""


def update_card(card: str, section: str) -> str:
    """Sustituye el aviso de que no hay test e inserta la sección antes de la de umbral."""
    if card.count(PENDING_NOTICE) != 1 or card.count(INSERT_BEFORE) != 1:
        raise PublishError("la model card no tiene la estructura esperada")
    if SECTION_TITLE in card:
        raise PublishError("la model card ya tiene la sección de test")
    updated = card.replace(PENDING_NOTICE, PUBLISHED_NOTICE)
    return updated.replace(INSERT_BEFORE, f"{section}{INSERT_BEFORE}")


def publish_test_section(
    api: HfApi | None = None, download: Callable[..., str] = hf_hub_download
) -> str:
    """Sube la card actualizada, verifica el modelo y registra ambas revisiones."""
    api = api or HfApi()
    report = json.loads(resolve_path(METRICS_JSON).read_text(encoding="utf-8"))
    revisions = _read_json(REVISIONS_JSON, {})
    entry = revisions["versiones"][EVALUATED_VERSION]
    evaluated = report["hf_sha"]
    if evaluated != entry["hf_sha"] or report["model_sha256"] != entry["model_sha256"]:
        raise PublishError("metricas-test.json no corresponde a la revisión publicada de v2")
    if "revision_documentacion" in entry:
        raise PublishError("la revisión de documentación ya está registrada")

    with tempfile.TemporaryDirectory() as tmp:
        card_path = download(HF_REPO, README_FILE, revision=evaluated, local_dir=tmp)
        config_path = download(HF_REPO, CONFIG_FILE, revision=evaluated, local_dir=tmp)
        card = Path(card_path).read_text(encoding="utf-8")
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    validation = config["validation_metrics"]

    limit = load_yaml(POLICY_PATH)["restriccion"]["maximo"]
    updated = update_card(card, build_section(report, validation, limit))
    content = updated.encode("utf-8")
    if FORBIDDEN_CONTENT.search(content):
        raise PublishError("la model card contiene un posible secreto o una ruta personal")

    commit = api.upload_file(
        path_or_fileobj=content,
        path_in_repo=README_FILE,
        repo_id=HF_REPO,
        commit_message="v2: añade la evaluación final en test a la model card",
        parent_commit=evaluated,
    )
    documentation = commit.oid

    files = sorted(api.list_repo_files(HF_REPO, revision=documentation))
    if files != sorted({*STAGING_FILES, PRESERVED_FILE}):
        raise PublishError(f"la revisión {documentation} contiene {files}")
    with tempfile.TemporaryDirectory() as tmp:
        model_path = download(HF_REPO, MODEL_FILE, revision=documentation, local_dir=tmp)
        model_sha256 = sha256_file(model_path)
    if model_sha256 != report["model_sha256"]:
        raise PublishError(f"SHA-256 en la nueva revisión {model_sha256}")
    if api.repo_info(HF_REPO, revision=EVALUATED_VERSION).sha != evaluated:
        raise PublishError("el tag v2 dejó de apuntar a la revisión evaluada")

    entry["revision_evaluada"] = evaluated
    entry["revision_documentacion"] = documentation
    entry["revision_documentacion_url"] = commit.commit_url
    entry["modelo_identico_en_documentacion"] = True
    _write_json(REVISIONS_JSON, revisions)
    return documentation


def main() -> None:
    documentation = publish_test_section()
    print(f"Revisión de documentación de v2: {documentation}")


if __name__ == "__main__":
    main()
