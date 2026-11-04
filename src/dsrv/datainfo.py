"""Versión de los datos usados para entrenar y comprobación de que coinciden con data-v2."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from dsrv.config import PREPARED_DIR, RAW_DIR, REPO_ROOT, resolve_path
from dsrv.gitinfo import _git
from dsrv.hashing import sha256_file
from dsrv.prepare import MANIFEST_NAME

DATA_TAG = "data-v2"
DVC_LOCK = Path("dvc.lock")
PREPARE_STAGE = "prepare"


class DataVersionError(RuntimeError):
    """Los datos locales no corresponden a la versión declarada."""


def _dvc_files() -> list[Path]:
    return sorted((resolve_path(RAW_DIR)).glob("*.dvc"))


def _md5_from_dvc_file(text: str) -> str:
    return yaml.safe_load(text)["outs"][0]["md5"]


def _md5_from_lock(text: str) -> str:
    outs = yaml.safe_load(text)["stages"][PREPARE_STAGE]["outs"]
    return next(out["md5"] for out in outs if out["path"] == PREPARED_DIR.as_posix())


def get_data_version() -> str:
    """Devuelve el tag y los md5 de los datos en data/raw/*.dvc y de data/prepared en dvc.lock."""
    parts = [DATA_TAG]
    for path in _dvc_files():
        parts.append(f"{path.stem}={_md5_from_dvc_file(path.read_text(encoding='utf-8'))}")
    lock_text = resolve_path(DVC_LOCK).read_text(encoding="utf-8")
    parts.append(f"{PREPARED_DIR.as_posix()}={_md5_from_lock(lock_text)}")
    return "|".join(parts)


def load_manifest() -> dict[str, Any]:
    path = resolve_path(PREPARED_DIR) / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_same_as_tag() -> None:
    try:
        _git("merge-base", "--is-ancestor", DATA_TAG, "HEAD")
    except subprocess.CalledProcessError as error:
        raise DataVersionError(f"{DATA_TAG} no existe o no es ancestro de HEAD") from error

    for path in _dvc_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        tagged = _md5_from_dvc_file(_git("show", f"{DATA_TAG}:{relative}"))
        if tagged != _md5_from_dvc_file(path.read_text(encoding="utf-8")):
            raise DataVersionError(f"{relative} cambió respecto a {DATA_TAG}")

    tagged = _md5_from_lock(_git("show", f"{DATA_TAG}:{DVC_LOCK.as_posix()}"))
    if tagged != _md5_from_lock(resolve_path(DVC_LOCK).read_text(encoding="utf-8")):
        raise DataVersionError(f"data/prepared en dvc.lock cambió respecto a {DATA_TAG}")


def _assert_dvc_status_clean() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "--json"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(result.stdout)
    if status:
        raise DataVersionError(f"dvc status no está limpio: {status}")


def _assert_files_match_manifest() -> None:
    prepared = resolve_path(PREPARED_DIR)
    for name, expected in load_manifest()["archivos_sha256"].items():
        if sha256_file(prepared / name) != expected:
            raise DataVersionError(f"{name} no coincide con el SHA-256 del manifest")


def assert_data_matches_version() -> None:
    """Lanza DataVersionError si los datos locales no son exactamente los de data-v2."""
    _assert_same_as_tag()
    _assert_dvc_status_clean()
    _assert_files_match_manifest()
