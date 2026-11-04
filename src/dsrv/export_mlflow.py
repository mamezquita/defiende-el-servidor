"""Exporta los runs de MLflow a evidencias/mlflow. Se ejecuta con `python -m dsrv.export_mlflow`."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run

from dsrv.config import EVIDENCE_DIR, resolve_path
from dsrv.mlflowinfo import all_runs, download_artifact, open_tracking
from dsrv.train import EXPERIMENT_NAME, TRACKING_URI

EXPORT_DIR = EVIDENCE_DIR / "mlflow"
LIGHT_ARTIFACTS = {"confusion_matrix.json", "confusion_matrix.png", "threshold_sweep.csv"}

# Traen rutas absolutas o el usuario del sistema, que no deben quedar en el repo.
PRIVATE_COLUMNS = ("artifact_uri",)
PRIVATE_TAGS = ("mlflow.source.name", "mlflow.user")

README = f"""# Evidencias de MLflow

Archivos generados por `python -m dsrv.export_mlflow`.

- `runs.csv`: todos los runs del experimento `{EXPERIMENT_NAME}`, incluidos los fallidos (tag `status=failed`).
- `<run_id>/run.json`: info, params, metrics y tags del run.
- `<run_id>/`: config, `confusion_matrix.json`, `confusion_matrix.png` y `threshold_sweep.csv`.

No se incluyen modelos. Tampoco las rutas absolutas ni el usuario del sistema (`artifact_uri`,
`{PRIVATE_TAGS[0]}`, `{PRIVATE_TAGS[1]}`).

## Cruzar con experimentos.csv

La columna `run_id` de `evidencias/experimentos.csv` es el nombre de la carpeta `<run_id>/` y
coincide con la columna `run_id` de `runs.csv`. Las métricas de `experimentos.csv` son las columnas
`metrics.*` de esa fila. Si una configuración tiene varios runs, `experimentos.csv` usa el último
terminado.

## Abrir la UI

Desde la raíz del repo, en la máquina donde se ejecutaron los experimentos (`mlflow.db` y
`mlartifacts/` no se versionan):

```bash
mlflow ui --backend-store-uri {TRACKING_URI}
```
"""


def _run_document(run: Run) -> dict[str, Any]:
    info = run.info
    return {
        "info": {
            "run_id": info.run_id,
            "run_name": info.run_name,
            "experiment_id": info.experiment_id,
            "status": info.status,
            "lifecycle_stage": info.lifecycle_stage,
            "start_time": info.start_time,
            "end_time": info.end_time,
        },
        "params": dict(sorted(run.data.params.items())),
        "metrics": dict(sorted(run.data.metrics.items())),
        "tags": {
            key: value for key, value in sorted(run.data.tags.items()) if key not in PRIVATE_TAGS
        },
    }


def _is_light_artifact(name: str) -> bool:
    return name in LIGHT_ARTIFACTS or name.endswith(".yaml")


def export_run(client: mlflow.MlflowClient, run: Run, output: Path) -> None:
    run_dir = output / run.info.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_run_document(run), ensure_ascii=False, indent=2, allow_nan=False)
    (run_dir / "run.json").write_text(text + "\n", encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        for artifact in client.list_artifacts(run.info.run_id):
            if not artifact.is_dir and _is_light_artifact(artifact.path):
                downloaded = download_artifact(run.info.run_id, artifact.path, tmp)
                shutil.copy2(downloaded, run_dir / artifact.path)


def write_runs_csv(output: Path) -> None:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME], order_by=["start_time ASC"])
    private = [*PRIVATE_COLUMNS, *(f"tags.{tag}" for tag in PRIVATE_TAGS)]
    runs = runs.drop(columns=private, errors="ignore")
    runs.to_csv(output / "runs.csv", index=False, lineterminator="\n")


def main() -> None:
    client = open_tracking()
    output = resolve_path(EXPORT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    write_runs_csv(output)
    runs = all_runs(client)
    for run in runs:
        export_run(client, run, output)
    (output / "LEEME.md").write_text(README, encoding="utf-8")
    print(f"Exportados {len(runs)} runs a {EXPORT_DIR.as_posix()}")


if __name__ == "__main__":
    main()
