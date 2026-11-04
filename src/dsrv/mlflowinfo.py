"""Consulta de los runs de MLflow que usan las evidencias de la comparación."""

import os
from pathlib import Path

import mlflow
from mlflow.entities import Run

from dsrv.config import CONFIGS_DIR, REPO_ROOT, resolve_path
from dsrv.train import EXPERIMENT_NAME, TRACKING_URI


def open_tracking() -> mlflow.MlflowClient:
    """Abre el backend del repo; su URI es relativa a la raíz."""
    os.chdir(REPO_ROOT)
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient()


def experiment_config_paths() -> list[Path]:
    """Devuelve las configs de experimentos (e1, e2...) en orden, relativas a la raíz del repo."""
    return sorted(CONFIGS_DIR / path.name for path in resolve_path(CONFIGS_DIR).glob("e[0-9]*.yaml"))


def all_runs(client: mlflow.MlflowClient) -> list[Run]:
    """Devuelve todos los runs del experimento, incluidos los fallidos, por fecha de inicio."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"no existe el experimento {EXPERIMENT_NAME}")
    return client.search_runs([experiment.experiment_id], order_by=["attributes.start_time ASC"])


def latest_finished_runs(client: mlflow.MlflowClient) -> dict[str, Run]:
    """Devuelve el último run terminado de cada configuración, en orden de experimento."""
    finished = [
        run
        for run in all_runs(client)
        if run.info.status == "FINISHED" and run.data.tags.get("status") != "failed"
    ]
    runs = {}
    for path in experiment_config_paths():
        matches = [run for run in finished if run.data.tags.get("config_id") == path.stem]
        if not matches:
            raise ValueError(f"{path.stem} no tiene ningún run terminado")
        runs[path.stem] = matches[-1]
    return runs


def download_artifact(run_id: str, artifact: str, destination: str | Path) -> Path:
    """Descarga un artefacto del run en destination y devuelve su ruta."""
    return Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=artifact, dst_path=str(destination)
        )
    )
