"""Genera RESULTADO.md de la prueba de recuperación.

Se ejecuta con `python -m dsrv.recovery_report --dir evidencias/recuperacion`.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from dsrv.config import resolve_path

VERSIONS = ("v1", "v2")
REPORT_NAME = "RESULTADO.md"
SUMMARY_NAME = "resumen_ejecucion.json"
TIMES_NAME = "tiempos.csv"
DATA_NAME = "verificacion_datos.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(
    summary: dict[str, Any],
    comparisons: list[dict[str, Any]],
    data: dict[str, Any],
    times: list[dict[str, str]],
) -> str:
    """Redacta el resultado con las cifras de los archivos generados por la prueba."""
    passed = (
        summary["estado"] == "OK"
        and data["resultado"] == "OK"
        and all(item["resultado"] == "OK" for item in comparisons)
    )
    rows = "\n".join(
        f"| {item['version']} | `{item['hf_sha']}` | {item['filas']} | "
        f"{item['coincidencias_row_id']}/{item['filas']} | "
        f"{item['coincidencias_y_pred']}/{item['filas']} | "
        f"{item['diferencia_maxima_score']:.3e} | {item['tolerancia_score']:.0e} | "
        f"{item['umbral']} | `{item['sha256_modelo']}` | {item['resultado']} |"
        for item in comparisons
    )
    files = [
        *data["csv_originales_contra_auditoria"],
        *data["datos_preparados_contra_manifest"],
    ]
    data_rows = "\n".join(
        f"| `{item['archivo']}` | `{item['calculado']}` | {'sí' if item['coincide'] else 'no'} |"
        for item in files
    )
    time_rows = "\n".join(
        f"| {row['paso']} | {row['segundos']} | {row['estado']} |" for row in times
    )
    total = sum(int(row["segundos"]) for row in times)

    return f"""# Resultado de la prueba de recuperación

**Resultado: {'SUPERADA' if passed else 'NO SUPERADA'}.**

Archivo generado por `python -m dsrv.recovery_report` a partir de los archivos de esta carpeta.

- **Repositorio:** {summary['repo_url']}
- **Commit probado:** `{summary['commit']}`
- **Repositorio de modelos:** {summary['hf_repo']}
- **Inicio:** {summary['inicio']}
- **Fin:** {summary['fin']}
- **Entorno:** clon, venv y caché de Hugging Face nuevos, con descarga anónima de los modelos. Misma
  máquina y sistema operativo del entrenamiento (WSL).

## Coincidencias con las predicciones originales

| Versión | SHA de Hugging Face | Filas | `row_id` iguales | `y_pred` iguales | Diferencia máxima de score | Tolerancia | Umbral | SHA-256 del modelo | Resultado |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
{rows}

## Datos recuperados con DVC

Resultado: **{data['resultado']}**. `manifest.json`: `{data['manifest_sha256']}`.

| Archivo | SHA-256 calculado | Coincide |
|---|---|---|
{data_rows}

## Tiempos

Total: {total} s.

| Paso | Segundos | Estado |
|---|---:|---|
{time_rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera RESULTADO.md de la recuperación")
    parser.add_argument("--dir", required=True, help="Carpeta con los resultados de la prueba")
    args = parser.parse_args()
    directory = resolve_path(args.dir)

    with (directory / TIMES_NAME).open(encoding="utf-8", newline="") as file:
        times = list(csv.DictReader(file))
    comparisons = [_read_json(directory / f"comparacion_{version}.json") for version in VERSIONS]
    report = build_report(
        _read_json(directory / SUMMARY_NAME), comparisons, _read_json(directory / DATA_NAME), times
    )
    (directory / REPORT_NAME).write_text(report, encoding="utf-8")
    print(f"Escrito {REPORT_NAME}")


if __name__ == "__main__":
    main()
