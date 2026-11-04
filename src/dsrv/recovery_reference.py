"""Genera las referencias de recuperación. Se ejecuta con `python -m dsrv.recovery_reference`.

Elige filas fijas de validación y guarda las predicciones de los modelos de hf_staging.
"""

import pandas as pd

from dsrv.config import EVIDENCE_DIR, PREPARED_DIR, SEED, TARGET_COLUMN, resolve_path
from dsrv.datainfo import load_manifest
from dsrv.hashing import sha256_file
from dsrv.package_model import STAGING_DIR
from dsrv.predict import predict_file

RECOVERY_DIR = EVIDENCE_DIR / "recuperacion"
FIXED_INPUTS = RECOVERY_DIR / "entradas_fijas.csv"
FIXED_INPUTS_SHA = RECOVERY_DIR / "entradas_fijas.csv.sha256"
ROWS_PER_CLASS = 20
VERSIONS = ("v1", "v2")


def select_fixed_rows(
    val: pd.DataFrame, features: list[str], rows_per_class: int = ROWS_PER_CLASS, seed: int = SEED
) -> pd.DataFrame:
    """Elige filas por clase, en el orden original, con row_id, entradas e y_true."""
    chosen = val.groupby(TARGET_COLUMN).sample(rows_per_class, random_state=seed).sort_index()
    frame = chosen[["row_id", *features]].copy()
    frame["y_true"] = chosen[TARGET_COLUMN]
    return frame


def main() -> None:
    features = load_manifest()["variables_predictoras"]
    val = pd.read_parquet(resolve_path(PREPARED_DIR) / "val.parquet")
    frame = select_fixed_rows(val, features)

    resolve_path(RECOVERY_DIR).mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolve_path(FIXED_INPUTS), index=False, lineterminator="\n")
    digest = sha256_file(resolve_path(FIXED_INPUTS))
    resolve_path(FIXED_INPUTS_SHA).write_text(
        f"{digest}  {FIXED_INPUTS.name}\n", encoding="utf-8"
    )

    for version in VERSIONS:
        output = RECOVERY_DIR / f"predicciones_originales_{version}.csv"
        predict_file(STAGING_DIR / version, FIXED_INPUTS, output)
    print(f"Escritas {len(frame)} filas en {FIXED_INPUTS.as_posix()}, SHA-256 {digest}")


if __name__ == "__main__":
    main()
