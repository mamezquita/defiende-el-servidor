"""Valida seleccion.json y lo marca como congelado. Se ejecuta con `python -m dsrv.freeze_selection`."""

import json
from typing import Any

from huggingface_hub import HfApi

from dsrv.config import resolve_path
from dsrv.datainfo import DATA_TAG
from dsrv.gitinfo import _git
from dsrv.package_model import HF_REPO
from dsrv.publish_hf import REVISIONS_JSON
from dsrv.select import SELECTION_JSON

REQUIRED_FIELDS = (
    "version",
    "config_id",
    "run_id",
    "git_commit",
    "data_version",
    "umbral",
    "metricas_validacion",
    "razon_seleccion",
    "hf_repo",
    "hf_sha",
    "model_sha256",
)
SHA_LENGTHS = {"git_commit": 40, "hf_sha": 40, "model_sha256": 64, "run_id": 32}
STATE_FROZEN = "congelada"


def incomplete_fields(document: dict[str, Any]) -> list[str]:
    """Devuelve los campos ausentes, vacíos o con un SHA de longitud incorrecta."""
    problems = [field for field in REQUIRED_FIELDS if document.get(field) in (None, "", {})]
    for field, length in SHA_LENGTHS.items():
        if document.get(field) and len(document[field]) != length:
            problems.append(field)
    return problems


def freeze(selection: dict[str, Any], revisions: dict[str, Any]) -> dict[str, Any]:
    """Comprueba cada versión contra hf_revisiones.json y añade sus referencias."""
    for document in selection["versiones"]:
        version = document["version"]
        problems = incomplete_fields(document)
        if problems:
            raise ValueError(f"{version}: campos incompletos en la selección: {problems}")

        revision = revisions["versiones"][version]
        for field in ("hf_sha", "model_sha256"):
            if document[field] != revision[field]:
                raise ValueError(f"{version}: {field} no coincide con hf_revisiones.json")
        if not document["data_version"].startswith(f"{DATA_TAG}|"):
            raise ValueError(f"{version}: data_version no corresponde a {DATA_TAG}")

        document.update(
            {
                "git_tag": version,
                "git_tag_commit": revision["git_tag_commit"],
                "dvc_tag": DATA_TAG,
                "hf_tag": revision["hf_tag"],
            }
        )
    selection["estado"] = STATE_FROZEN
    return selection


def _check_remote_references(selection: dict[str, Any], api: HfApi) -> None:
    """Comprueba que los tags de Git y de Hugging Face resuelvan a los commits declarados."""
    for document in selection["versiones"]:
        version = document["version"]
        if _git("rev-list", "-n", "1", version) != document["git_tag_commit"]:
            raise ValueError(f"el tag de Git {version} no apunta a {document['git_tag_commit']}")
        if api.repo_info(HF_REPO, revision=document["hf_tag"]).sha != document["hf_sha"]:
            raise ValueError(f"el tag de Hugging Face {version} no apunta a {document['hf_sha']}")


def main() -> None:
    path = resolve_path(SELECTION_JSON)
    selection = json.loads(path.read_text(encoding="utf-8"))
    revisions = json.loads(resolve_path(REVISIONS_JSON).read_text(encoding="utf-8"))

    frozen = freeze(selection, revisions)
    _check_remote_references(frozen, HfApi())

    text = json.dumps(frozen, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"{SELECTION_JSON.as_posix()}: estado {STATE_FROZEN}")


if __name__ == "__main__":
    main()
