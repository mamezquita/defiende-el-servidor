"""Limpieza y partición agrupada del training original. Se ejecuta con `python -m dsrv.prepare`.

Solo limpia y parte: no ajusta codificadores ni escaladores, y no lee el archivo de test.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dsrv.config import (
    EXCLUDED_COLUMNS,
    PREPARED_DIR,
    SEED,
    TARGET_COLUMN,
    TRAINING_CSV,
    resolve_path,
)
from dsrv.hashing import sha256_dataframe_rows, sha256_file

EXPECTED_TRAINING_ROWS = 175_341
VAL_SIZE = 0.2
ID_COLUMN = "id"

REASON_DUPLICATE = "duplicado_exacto"
REASON_CONTRADICTORY = "grupo_contradictorio"
SPLIT_NAMES = ("train", "val", "excluded")

MANIFEST_NAME = "manifest.json"
SPLITS_NAME = "splits.csv"

CLEANING_POLICY = (
    "Duplicados exactos (mismas entradas y misma etiqueta) en training: se conserva una fila por "
    "grupo; se registra cuántas se eliminaron.",
    "Grupos contradictorios (mismas entradas, etiquetas 0 y 1): se excluyen de train y validación.",
    "Split agrupado: GroupShuffleSplit(test_size=0.2, random_state=42) con group = hash de "
    "entradas.",
    "El archivo de test no se modifica ni se deduplica.",
)


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV original.

    Los CSV traen BOM: sin utf-8-sig, la columna id podría no reconocerse como excluida.
    """
    return pd.read_csv(resolve_path(path), encoding="utf-8-sig", low_memory=False)


def assign_splits(
    df: pd.DataFrame, features: Sequence[str], seed: int = SEED
) -> pd.DataFrame:
    """Asigna a cada fila su partición y motivo de exclusión, en el orden del CSV original.

    De cada grupo sin contradicción se conserva la primera fila. Como queda una fila por grupo y
    GroupShuffleSplit ordena los grupos, la partición no depende del orden de las filas.
    """
    frame = pd.DataFrame(
        {
            "row_id": [f"train:{position}" for position in range(len(df))],
            "group_id": sha256_dataframe_rows(df, features).to_numpy(),
            "split": "excluded",
            "motivo_exclusion": "",
            TARGET_COLUMN: df[TARGET_COLUMN].to_numpy(),
        }
    )
    contradictory = frame.groupby("group_id")[TARGET_COLUMN].transform("nunique") > 1
    duplicate = frame["group_id"].duplicated(keep="first") & ~contradictory
    frame.loc[contradictory, "motivo_exclusion"] = REASON_CONTRADICTORY
    frame.loc[duplicate, "motivo_exclusion"] = REASON_DUPLICATE

    kept = frame[~contradictory & ~duplicate]
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=seed)
    train_positions, val_positions = next(splitter.split(kept, groups=kept["group_id"]))
    frame.loc[kept.index[train_positions], "split"] = "train"
    frame.loc[kept.index[val_positions], "split"] = "val"
    return frame


def validate_partitions(
    frame: pd.DataFrame, features: Sequence[str], expected_rows: int = EXPECTED_TRAINING_ROWS
) -> None:
    """Lanza ValueError si la partición incumple alguna regla de datos."""
    train_groups = set(frame.loc[frame["split"] == "train", "group_id"])
    val_groups = set(frame.loc[frame["split"] == "val", "group_id"])
    contradictory_groups = set(
        frame.loc[frame["motivo_exclusion"] == REASON_CONTRADICTORY, "group_id"]
    )

    if train_groups & val_groups:
        raise ValueError(f"{len(train_groups & val_groups)} grupos están en train y en val")
    if contradictory_groups & (train_groups | val_groups):
        raise ValueError("hay grupos contradictorios dentro de train o val")
    if frame["row_id"].duplicated().any():
        raise ValueError("hay row_id repetidos")
    if len(frame) != expected_rows:
        raise ValueError(f"train + val + excluded = {len(frame)}; se esperaban {expected_rows}")

    forbidden = {*EXCLUDED_COLUMNS, "row_id", "group_id"} & set(features)
    if forbidden:
        raise ValueError(f"columnas prohibidas entre las predictoras: {sorted(forbidden)}")


def split_fingerprint(row_ids: pd.Series) -> str:
    """SHA-256 de los row_id ordenados como texto y unidos con salto de línea."""
    return hashlib.sha256("\n".join(sorted(row_ids)).encode("utf-8")).hexdigest()


def _class_counts(labels: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in labels.value_counts().sort_index().items()}


def build_manifest(
    frame: pd.DataFrame,
    features: Sequence[str],
    source_path: str | Path,
    file_hashes: dict[str, str],
    seed: int = SEED,
) -> dict[str, Any]:
    """Resume la partición sin marcas de tiempo, para que dos corridas den el mismo manifest."""
    parts = {name: frame[frame["split"] == name] for name in SPLIT_NAMES}
    counts = {name: _class_counts(part[TARGET_COLUMN]) for name, part in parts.items()}
    return {
        "semilla": seed,
        "val_size": VAL_SIZE,
        "fuente": {
            "ruta": Path(source_path).as_posix(),
            "sha256": sha256_file(resolve_path(source_path)),
        },
        "variables_predictoras": list(features),
        "tamanos": {**{name: len(part) for name, part in parts.items()}, "total": len(frame)},
        "filas_eliminadas": {
            reason: int((frame["motivo_exclusion"] == reason).sum())
            for reason in (REASON_DUPLICATE, REASON_CONTRADICTORY)
        },
        "conteo_clases": counts,
        "proporcion_clases": {
            name: {
                label: round(count / len(parts[name]), 6)
                for label, count in class_counts.items()
            }
            for name, class_counts in counts.items()
        },
        "archivos_sha256": file_hashes,
        "huellas_split": {name: split_fingerprint(part["row_id"]) for name, part in parts.items()},
        "huellas_split_metodo": "SHA-256 de los row_id ordenados como texto y unidos con \\n",
    }


def write_outputs(
    df: pd.DataFrame,
    frame: pd.DataFrame,
    features: Sequence[str],
    source_path: str | Path,
    output_dir: str | Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Escribe los parquet, splits.csv y manifest.json, y devuelve el manifest."""
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = df[[ID_COLUMN, *features]].reset_index(drop=True)
    rows.insert(0, "row_id", frame["row_id"])
    rows.insert(2, "group_id", frame["group_id"])
    rows[TARGET_COLUMN] = frame[TARGET_COLUMN]

    file_hashes = {}
    for name in ("train", "val"):
        path = output / f"{name}.parquet"
        rows[frame["split"] == name].to_parquet(path, index=False)
        file_hashes[path.name] = sha256_file(path)

    splits_path = output / SPLITS_NAME
    frame[["row_id", "group_id", "split", "motivo_exclusion"]].to_csv(
        splits_path, index=False, lineterminator="\n"
    )
    file_hashes[splits_path.name] = sha256_file(splits_path)

    manifest = build_manifest(frame, features, source_path, file_hashes, seed)
    text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    (output / MANIFEST_NAME).write_text(text + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    df = read_raw_csv(TRAINING_CSV)
    features = [col for col in df.columns if col not in EXCLUDED_COLUMNS]
    frame = assign_splits(df, features)
    validate_partitions(frame, features)
    manifest = write_outputs(df, frame, features, TRAINING_CSV, PREPARED_DIR)
    print(f"Tamaños: {manifest['tamanos']}")
    print(f"Filas eliminadas: {manifest['filas_eliminadas']}")


if __name__ == "__main__":
    main()
