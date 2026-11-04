"""Comprueba los datos recuperados con DVC. Se ejecuta con `python -m dsrv.verify_data --out <dir>`.

Compara los CSV originales con auditoria.json y data/prepared con su manifest.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dsrv.config import EVIDENCE_DIR, PREPARED_DIR, resolve_path
from dsrv.hashing import sha256_file

AUDIT_JSON = EVIDENCE_DIR / "auditoria.json"
MANIFEST_NAME = "manifest.json"
REPORT_NAME = "verificacion_datos.json"


def _check(name: str, path: Path, expected: str) -> dict[str, Any]:
    if not path.exists():
        return {"archivo": name, "esperado": expected, "calculado": None, "coincide": False}
    actual = sha256_file(path)
    return {"archivo": name, "esperado": expected, "calculado": actual, "coincide": actual == expected}


def verify(
    audit_path: str | Path = AUDIT_JSON, prepared_dir: str | Path = PREPARED_DIR
) -> dict[str, Any]:
    """Calcula el SHA-256 de cada archivo y lo compara con el valor registrado."""
    audit = json.loads(resolve_path(audit_path).read_text(encoding="utf-8"))
    prepared = resolve_path(prepared_dir)
    manifest = json.loads((prepared / MANIFEST_NAME).read_text(encoding="utf-8"))

    raw = [
        _check(section["ruta"], resolve_path(section["ruta"]), section["sha256_esperado"])
        for section in audit["archivos"].values()
    ]
    derived = [
        _check(f"{Path(prepared_dir).as_posix()}/{name}", prepared / name, expected)
        for name, expected in manifest["archivos_sha256"].items()
    ]
    return {
        "csv_originales_contra_auditoria": raw,
        "datos_preparados_contra_manifest": derived,
        "manifest_sha256": sha256_file(prepared / MANIFEST_NAME),
        "resultado": "OK" if all(item["coincide"] for item in [*raw, *derived]) else "FALLA",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba los datos recuperados con DVC")
    parser.add_argument("--out", required=True, help="Carpeta de salida")
    args = parser.parse_args()

    report = verify()
    output = resolve_path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output / REPORT_NAME).write_text(text + "\n", encoding="utf-8")

    checked = report["csv_originales_contra_auditoria"] + report["datos_preparados_contra_manifest"]
    print(f"Datos: {report['resultado']}, {len(checked)} archivos comprobados")
    return 0 if report["resultado"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
