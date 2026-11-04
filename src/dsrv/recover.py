"""Recupera una versión desde Hugging Face y la compara con sus predicciones originales.

Se ejecuta con:
`python -m dsrv.recover --version v1 --sha <SHA> --inputs <csv> --reference <csv> --out <dir>`
"""

import argparse
import json
import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from dsrv.config import EVIDENCE_DIR, resolve_path
from dsrv.hashing import sha256_file
from dsrv.predict import CONFIG_FILE, MODEL_FILE, load_packaged_model, predict_frame

REVISIONS_JSON = EVIDENCE_DIR / "hf_revisiones.json"
SCORE_TOLERANCE = 1e-9
FULL_SHA = re.compile(r"[0-9a-f]{40}")
RESULT_OK = "OK"
RESULT_FAILED = "FALLA"


class RecoveryError(RuntimeError):
    """La revisión o el modelo no corresponden a lo registrado."""


def registered_version(version: str, sha: str, hf_repo: str | None = None) -> dict[str, Any]:
    """Comprueba el SHA y el repo contra hf_revisiones.json y devuelve el registro de la versión."""
    if not FULL_SHA.fullmatch(sha):
        raise RecoveryError(f"se requiere el SHA completo de 40 caracteres, no {sha!r}")

    revisions = json.loads(resolve_path(REVISIONS_JSON).read_text(encoding="utf-8"))
    entry = revisions["versiones"][version]
    if sha != entry["hf_sha"]:
        raise RecoveryError(f"{version}: el SHA {sha} no es el registrado, {entry['hf_sha']}")
    if hf_repo is not None and hf_repo != revisions["hf_repo"]:
        raise RecoveryError(f"el repo {hf_repo} no es el registrado, {revisions['hf_repo']}")
    return {**entry, "hf_repo": revisions["hf_repo"]}


def download_and_load(
    entry: dict[str, Any], directory: Path, download: Callable[..., str] = hf_hub_download
) -> tuple[Any, dict[str, Any], str]:
    """Descarga la revisión y comprueba el SHA-256 del modelo antes de cargar el pickle."""
    for name in (MODEL_FILE, CONFIG_FILE):
        download(entry["hf_repo"], name, revision=entry["hf_sha"], local_dir=str(directory))

    model_sha256 = sha256_file(directory / MODEL_FILE)
    config = json.loads((directory / CONFIG_FILE).read_text(encoding="utf-8"))
    if model_sha256 != config["model_sha256"]:
        raise RecoveryError(f"SHA-256 del modelo {model_sha256}; config.json declara otro")
    if model_sha256 != entry["model_sha256"]:
        raise RecoveryError(f"SHA-256 del modelo {model_sha256}; hf_revisiones.json declara otro")

    model, config = load_packaged_model(directory)
    return model, config, model_sha256


def compare_predictions(recovered: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    """Compara filas en orden: mismo row_id, y_pred idéntico y score_attack dentro de la tolerancia."""
    if len(recovered) != len(reference):
        return {
            "filas": len(recovered),
            "filas_referencia": len(reference),
            "coincidencias_row_id": 0,
            "coincidencias_y_pred": 0,
            "diferencia_maxima_score": None,
            "tolerancia_score": SCORE_TOLERANCE,
            "resultado": RESULT_FAILED,
        }

    same_id = recovered["row_id"].to_numpy() == reference["row_id"].to_numpy()
    same_pred = recovered["y_pred"].to_numpy() == reference["y_pred"].to_numpy()
    difference = (recovered["score_attack"] - reference["score_attack"]).abs()
    largest = float(difference.max())
    passed = bool(same_id.all() and same_pred.all() and largest <= SCORE_TOLERANCE)
    return {
        "filas": len(recovered),
        "filas_referencia": len(reference),
        "coincidencias_row_id": int(same_id.sum()),
        "coincidencias_y_pred": int(same_pred.sum()),
        "diferencia_maxima_score": largest,
        "tolerancia_score": SCORE_TOLERANCE,
        "resultado": RESULT_OK if passed else RESULT_FAILED,
    }


def recover(
    version: str,
    sha: str,
    inputs: str | Path,
    reference: str | Path,
    output_dir: str | Path,
    hf_repo: str | None = None,
    download: Callable[..., str] = hf_hub_download,
) -> dict[str, Any]:
    """Recupera la versión, escribe sus predicciones y su comparación, y devuelve la comparación."""
    entry = registered_version(version, sha, hf_repo)
    with tempfile.TemporaryDirectory() as tmp:
        model, config, model_sha256 = download_and_load(entry, Path(tmp), download)

    frame = pd.read_csv(resolve_path(inputs))
    recovered = predict_frame(model, config, frame)
    expected = pd.read_csv(resolve_path(reference))

    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / f"predicciones_recuperadas_{version}.csv"
    recovered.to_csv(predictions_path, index=False, lineterminator="\n")

    comparison = {
        "version": version,
        "hf_repo": entry["hf_repo"],
        "hf_sha": sha,
        **compare_predictions(recovered, expected),
        "umbral": config["threshold"],
        "sha256_entradas": sha256_file(resolve_path(inputs)),
        "sha256_referencia": sha256_file(resolve_path(reference)),
        "sha256_modelo": model_sha256,
        "sha256_salida": sha256_file(predictions_path),
    }
    text = json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False)
    (output / f"comparacion_{version}.json").write_text(text + "\n", encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Recupera una versión desde Hugging Face")
    parser.add_argument("--version", required=True, choices=["v1", "v2"])
    parser.add_argument("--sha", required=True, help="SHA completo de la revisión de Hugging Face")
    parser.add_argument("--inputs", required=True, help="CSV de entradas fijas")
    parser.add_argument("--reference", required=True, help="CSV de predicciones originales")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    parser.add_argument("--hf-repo", help="Repo de Hugging Face; debe ser el registrado")
    args = parser.parse_args()

    comparison = recover(
        args.version, args.sha, args.inputs, args.reference, args.out, args.hf_repo
    )
    print(
        f"{args.version}: {comparison['resultado']}, "
        f"{comparison['coincidencias_y_pred']}/{comparison['filas']} y_pred iguales, "
        f"diferencia máxima de score {comparison['diferencia_maxima_score']}"
    )
    return 0 if comparison["resultado"] == RESULT_OK else 1


if __name__ == "__main__":
    sys.exit(main())
