"""Auditoría de los CSV originales de UNSW-NB15. Se ejecuta con `python -m dsrv.audit`."""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from dsrv.config import (
    CONFIGS_DIR,
    EVIDENCE_DIR,
    EXCLUDED_COLUMNS,
    TARGET_COLUMN,
    load_yaml,
    resolve_path,
)
from dsrv.hashing import sha256_dataframe_rows, sha256_file

SOURCE_CONFIG = CONFIGS_DIR / "data_source.yaml"
AUDIT_JSON = EVIDENCE_DIR / "auditoria.json"
FEATURE_HASH_CSV = EVIDENCE_DIR / "feature_hash_train.csv"
FEATURE_HASH_MAX_BYTES = 5 * 1024 * 1024

CATEGORICAL_COLUMNS = ("proto", "service", "state")
ATTACK_CATEGORY_COLUMN = "attack_cat"
MISSING_KEY = "__faltante__"
EXCLUSION_REASONS = {
    "label": "Variable objetivo.",
    "attack_cat": "Fuga de información: describe el tipo de ataque, es decir, el objetivo.",
    "id": "Identificador de fila sin información de la conexión.",
}


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    """Lee un CSV original.

    Los CSV traen BOM: sin utf-8-sig, la columna id podría no reconocerse como excluida.
    """
    return pd.read_csv(resolve_path(path), encoding="utf-8-sig", low_memory=False)


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    items = ((MISSING_KEY if pd.isna(k) else str(k), int(v)) for k, v in counts.items())
    return dict(sorted(items))


def audit_file(path: str | Path, expected: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Resume un archivo: huella, tamaño, esquema, faltantes y conteos."""
    real_path = resolve_path(path)
    sha256 = sha256_file(real_path)
    return {
        "ruta": Path(path).as_posix(),
        "sha256": sha256,
        "sha256_esperado": expected["sha256"],
        "sha256_coincide": sha256 == expected["sha256"],
        "tamano_bytes": real_path.stat().st_size,
        "filas": len(df),
        "filas_esperadas": expected["filas"],
        "filas_coinciden": len(df) == expected["filas"],
        "columnas": df.shape[1],
        "esquema": [
            {"nombre": col, "dtype": str(df[col].dtype), "faltantes": int(df[col].isna().sum())}
            for col in df.columns
        ],
        "conteo_label": _value_counts(df[TARGET_COLUMN]),
        "conteo_attack_cat": _value_counts(df[ATTACK_CATEGORY_COLUMN]),
        "categorias": {col: _value_counts(df[col]) for col in CATEGORICAL_COLUMNS},
    }


def group_stats(hashes: pd.Series, labels: pd.Series) -> tuple[dict[str, int], dict[str, int]]:
    """Calcula duplicados y grupos contradictorios (mismas entradas, label 0 y 1)."""
    frame = pd.DataFrame({"hash": hashes.to_numpy(), "label": labels.to_numpy()})
    groups = frame.groupby("hash")["label"]
    sizes = groups.size()
    label_kinds = groups.nunique()
    contradictory = label_kinds[label_kinds > 1].index
    duplicates = {
        "filas": len(frame),
        "grupos_distintos": len(sizes),
        "filas_duplicadas_sobrantes": len(frame) - len(sizes),
        "filas_en_grupos_repetidos": int(sizes[sizes > 1].sum()),
        "grupos_con_mas_de_una_fila": int((sizes > 1).sum()),
        "tamano_maximo_grupo": int(sizes.max()) if len(sizes) else 0,
    }
    contradictions = {
        "grupos": len(contradictory),
        "filas": int(sizes[contradictory].sum()),
    }
    return duplicates, contradictions


def cross_stats(
    train_hashes: pd.Series,
    train_labels: pd.Series,
    test_hashes: pd.Series,
    test_labels: pd.Series,
) -> dict[str, int]:
    """Cuenta las filas de testing con entradas idénticas a alguna fila de training."""
    train = pd.DataFrame({"hash": train_hashes.to_numpy(), "label": train_labels.to_numpy()})
    test = pd.DataFrame({"hash": test_hashes.to_numpy(), "label": test_labels.to_numpy()})
    train_range = train.groupby("hash")["label"].agg(["min", "max"])
    overlap = test.join(train_range, on="hash", how="inner")
    differs = (overlap["label"] != overlap["min"]) | (overlap["label"] != overlap["max"])
    no_match = (overlap["min"] == overlap["max"]) & (overlap["label"] != overlap["min"])
    return {
        "filas_testing_en_training": len(overlap),
        "grupos_en_comun": int(overlap["hash"].nunique()),
        "filas_con_alguna_etiqueta_distinta": int(differs.sum()),
        "grupos_con_alguna_etiqueta_distinta": int(overlap.loc[differs, "hash"].nunique()),
        "filas_sin_etiqueta_coincidente": int(no_match.sum()),
    }


def check_content(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_file: dict[str, Any],
    test_file: dict[str, Any],
) -> dict[str, Any]:
    """Comprueba huellas, filas y esquema, y advierte si los archivos parecen invertidos."""
    same_columns = list(train.columns) == list(test.columns)
    different_dtypes = (
        [col for col in train.columns if train[col].dtype != test[col].dtype]
        if same_columns
        else []
    )
    inverted = (
        len(train) < len(test)
        or train_file["sha256"] == test_file["sha256_esperado"]
        or test_file["sha256"] == train_file["sha256_esperado"]
    )

    warnings = []
    for name, section in (("training", train_file), ("testing", test_file)):
        if not section["sha256_coincide"]:
            warnings.append(f"{name}: el SHA-256 no coincide con la huella esperada.")
        if not section["filas_coinciden"]:
            warnings.append(
                f"{name}: {section['filas']} filas; se esperaban {section['filas_esperadas']}."
            )
    if not same_columns:
        warnings.append("training y testing no tienen las mismas columnas en el mismo orden.")
    if inverted:
        warnings.append("Los conteos o las huellas sugieren que train y test están invertidos.")

    return {
        "mismas_columnas": same_columns,
        "columnas_con_dtype_distinto": different_dtypes,
        "posible_inversion_train_test": inverted,
        "advertencias": warnings,
    }


def build_audit(
    source: dict[str, Any],
) -> tuple[dict[str, Any], pd.Series, pd.Series]:
    """Construye el contenido de auditoria.json y devuelve también los hashes de training."""
    files = source["archivos"]
    train = read_raw_csv(files["training"]["ruta"])
    test = read_raw_csv(files["testing"]["ruta"])

    train_file = audit_file(files["training"]["ruta"], files["training"], train)
    test_file = audit_file(files["testing"]["ruta"], files["testing"], test)
    checks = check_content(train, test, train_file, test_file)
    checks["advertencias"].extend(
        f"fuente: falta {field}." for field, value in source["fuente"].items() if not value
    )

    features = [col for col in train.columns if col not in EXCLUDED_COLUMNS]
    if not checks["mismas_columnas"] and set(features) - set(test.columns):
        raise ValueError("testing no contiene todas las variables predictoras de training")

    train_hashes = sha256_dataframe_rows(train, features)
    test_hashes = sha256_dataframe_rows(test, features)
    train_labels = train[TARGET_COLUMN]
    test_labels = test[TARGET_COLUMN]
    union_hashes = pd.concat([train_hashes, test_hashes], ignore_index=True)
    union_labels = pd.concat([train_labels, test_labels], ignore_index=True)

    duplicates: dict[str, Any] = {}
    contradictions: dict[str, Any] = {}
    for name, hashes, labels in (
        ("training", train_hashes, train_labels),
        ("testing", test_hashes, test_labels),
        ("union", union_hashes, union_labels),
    ):
        duplicates[name], contradictions[name] = group_stats(hashes, labels)

    audit = {
        "fuentes": source["fuente"],
        "archivos": {"training": train_file, "testing": test_file},
        "comprobaciones": checks,
        "variables": {
            "objetivo": TARGET_COLUMN,
            "clase_positiva": 1,
            "usadas": features,
            "excluidas": [
                {"nombre": col, "motivo": EXCLUSION_REASONS[col]} for col in EXCLUDED_COLUMNS
            ],
        },
        "duplicados": duplicates,
        "contradicciones": contradictions,
        "cruces_train_test": cross_stats(train_hashes, train_labels, test_hashes, test_labels),
        "politica_limpieza": {
            "estado": "borrador; se completa en la fase 3",
            "grupos_contradictorios": "Excluir de train y validación y documentar aquí.",
            "split": "Agrupado por hash de predictoras: ningún grupo en ambos lados.",
            "testing": "No se limpia ni decide nada; sus cifras son solo informativas.",
        },
    }
    return audit, train_hashes, train_labels


def write_feature_hash_train(
    hashes: pd.Series, labels: pd.Series, path: str | Path
) -> dict[str, Any]:
    """Guarda hash y label por fila de training solo si el CSV pesa menos del límite."""
    frame = pd.DataFrame({"feature_hash": hashes.to_numpy(), TARGET_COLUMN: labels.to_numpy()})
    content = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    saved = len(content) < FEATURE_HASH_MAX_BYTES
    if saved:
        resolve_path(path).write_bytes(content)
    return {
        "ruta": Path(path).as_posix(),
        "guardado": saved,
        "tamano_bytes": len(content),
        "limite_bytes": FEATURE_HASH_MAX_BYTES,
    }


def write_audit(audit: dict[str, Any], path: str | Path) -> None:
    text = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)
    resolve_path(path).write_text(text + "\n", encoding="utf-8")


def main() -> int:
    source = load_yaml(SOURCE_CONFIG)
    audit, train_hashes, train_labels = build_audit(source)
    audit["duplicados"]["feature_hash_train"] = write_feature_hash_train(
        train_hashes, train_labels, FEATURE_HASH_CSV
    )
    write_audit(audit, AUDIT_JSON)

    print(f"Escrito {AUDIT_JSON.as_posix()}")
    warnings = audit["comprobaciones"]["advertencias"]
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}", file=sys.stderr)
    return 1 if warnings else 0


if __name__ == "__main__":
    sys.exit(main())
