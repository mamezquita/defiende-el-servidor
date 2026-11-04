"""Publica hf_staging/<version>/ en Hugging Face. Se ejecuta con `python -m dsrv.publish_hf --version v1`.

Usa la sesión de `hf auth login`; este módulo nunca lee ni imprime el token.
"""

import argparse
import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from dsrv.config import EVIDENCE_DIR, resolve_path
from dsrv.hashing import sha256_file
from dsrv.package_model import HF_REPO, STAGING_DIR, STAGING_FILES
from dsrv.predict import CONFIG_FILE, MODEL_FILE
from dsrv.select import SELECTION_JSON

REVISIONS_JSON = EVIDENCE_DIR / "hf_revisiones.json"
PUBLISH_ORDER = ("v1", "v2")
PRESERVED_FILE = ".gitattributes"

FORBIDDEN_CONTENT = re.compile(
    rb"hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|AKIA[0-9A-Z]{12,}"
    rb"|access_key_id|secret_access_key|BEGIN [A-Z ]*PRIVATE KEY"
    rb"|/home/[a-z]|C:\\Users"
)


class PublishError(RuntimeError):
    """La publicación no cumple una condición de seguridad o de trazabilidad."""


def check_staging(staging: Path) -> None:
    """Exige exactamente los archivos esperados, sin secretos ni rutas personales."""
    found = {path.relative_to(staging).as_posix() for path in staging.rglob("*") if path.is_file()}
    if found != STAGING_FILES:
        raise PublishError(
            f"staging incorrecto; sobran {sorted(found - STAGING_FILES)}, "
            f"faltan {sorted(STAGING_FILES - found)}"
        )
    for name in sorted(found):
        if FORBIDDEN_CONTENT.search((staging / name).read_bytes()):
            raise PublishError(f"{name} contiene un posible secreto o una ruta personal")


def _read_json(path: str | Path, default: dict[str, Any]) -> dict[str, Any]:
    real_path = resolve_path(path)
    if not real_path.exists():
        return default
    return json.loads(real_path.read_text(encoding="utf-8"))


def _write_json(path: str | Path, content: dict[str, Any]) -> None:
    text = json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False)
    resolve_path(path).write_text(text + "\n", encoding="utf-8")


def _check_can_publish(api: HfApi, version: str, revisions: dict[str, Any]) -> None:
    earlier = PUBLISH_ORDER[: PUBLISH_ORDER.index(version)]
    missing = [item for item in earlier if item not in revisions["versiones"]]
    if missing:
        raise PublishError(f"publica antes {missing}")
    if version in {tag.name for tag in api.list_repo_refs(HF_REPO).tags}:
        raise PublishError(f"el tag {version} ya existe en Hugging Face y no se borran tags")


def _verify_revision(
    api: HfApi, sha: str, model_sha256: str, download: Callable[..., str]
) -> list[str]:
    """Comprueba el contenido de la revisión y el SHA-256 del modelo descargado de nuevo."""
    files = sorted(api.list_repo_files(HF_REPO, revision=sha))
    expected = sorted({*STAGING_FILES, PRESERVED_FILE})
    if files != expected:
        raise PublishError(f"la revisión {sha} contiene {files}; se esperaba {expected}")

    with tempfile.TemporaryDirectory() as tmp:
        path = download(HF_REPO, MODEL_FILE, revision=sha, local_dir=tmp)
        downloaded = sha256_file(path)
    if downloaded != model_sha256:
        raise PublishError(f"SHA-256 descargado {downloaded}; se esperaba {model_sha256}")
    return files


def _complete_selection(version: str, sha: str, model_sha256: str) -> None:
    selection = _read_json(SELECTION_JSON, {})
    document = next(item for item in selection["versiones"] if item["version"] == version)
    document.update({"hf_repo": HF_REPO, "hf_sha": sha, "model_sha256": model_sha256})
    if all(item["hf_sha"] for item in selection["versiones"]):
        selection["estado"] = "publicada"
    _write_json(SELECTION_JSON, selection)


def publish(
    version: str, api: HfApi | None = None, download: Callable[..., str] = hf_hub_download
) -> str:
    """Sube el staging, etiqueta el commit, verifica la descarga y devuelve el SHA de la revisión."""
    api = api or HfApi()
    staging = resolve_path(STAGING_DIR) / version
    check_staging(staging)
    config = json.loads((staging / CONFIG_FILE).read_text(encoding="utf-8"))
    if sha256_file(staging / MODEL_FILE) != config["model_sha256"]:
        raise PublishError(f"{MODEL_FILE} no coincide con el SHA-256 de {CONFIG_FILE}")

    revisions = _read_json(REVISIONS_JSON, {"hf_repo": HF_REPO, "versiones": {}})
    _check_can_publish(api, version, revisions)

    commit = api.upload_folder(
        repo_id=HF_REPO,
        folder_path=str(staging),
        commit_message=f"{version}: {config['config_id']}, umbral {config['threshold']}",
        delete_patterns="*",
    )
    sha = commit.oid
    if len(sha) != 40:
        raise PublishError(f"SHA de commit inesperado: {sha}")
    api.create_tag(
        HF_REPO, tag=version, revision=sha, tag_message=f"{version}: {config['config_id']}"
    )
    files = _verify_revision(api, sha, config["model_sha256"], download)

    revisions["versiones"][version] = {
        "hf_sha": sha,
        "hf_tag": version,
        "hf_commit_url": commit.commit_url,
        "archivos": files,
        "config_id": config["config_id"],
        "umbral": config["threshold"],
        "model_sha256": config["model_sha256"],
        "descarga_verificada": True,
        "git_commit": config["git_commit"],
        "git_tag": config["git_tag"],
        "git_tag_commit": config["git_tag_commit"],
        "data_version": config["data_version"],
        "mlflow_run_id": config["mlflow_run_id"],
    }
    _write_json(REVISIONS_JSON, revisions)
    _complete_selection(version, sha, config["model_sha256"])
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description="Publica hf_staging/<version>/ en Hugging Face")
    parser.add_argument("--version", required=True, choices=list(PUBLISH_ORDER))
    args = parser.parse_args()
    sha = publish(args.version)
    print(f"{args.version} publicada en {HF_REPO} con SHA {sha}")


if __name__ == "__main__":
    main()
