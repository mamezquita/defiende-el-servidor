"""Hashes SHA-256 en hexadecimal minúscula de archivos y de filas de un DataFrame."""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Calcula el SHA-256 de un archivo leyéndolo por bloques de 1 MB."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_text(value: Any) -> str | None:
    """Texto estable de un valor; None para los faltantes.

    Un mismo número da el mismo texto sea cual sea su dtype: 1, 1.0 y -0.0/0 no se distinguen.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return str(int(number)) if number.is_integer() else repr(number)
    return str(value)


def sha256_dataframe_rows(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Devuelve un SHA-256 por fila calculado sobre las columnas dadas.

    Las columnas se recorren ordenadas por nombre, de modo que el hash no depende del orden en
    que se pasen ni del índice de df. Cada fila se serializa como lista JSON, que separa los
    valores sin ambigüedad y distingue un faltante del texto "nan". La Serie resultante conserva
    el índice de df solo para poder alinearla con él.
    """
    if not cols:
        raise ValueError("cols no puede estar vacío")
    if len(set(cols)) != len(cols):
        raise ValueError("cols tiene columnas repetidas")

    columns = [[_stable_text(v) for v in df[col].tolist()] for col in sorted(cols)]
    hashes = [
        hashlib.sha256(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        for row in zip(*columns)
    ]
    return pd.Series(hashes, index=df.index, name="row_hash", dtype="str")
