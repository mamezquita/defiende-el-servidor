"""Genera evidencias/experimentos.csv. Se ejecuta con `python -m dsrv.build_experiments_table`."""

import json
import tempfile

import pandas as pd

from dsrv.config import EVIDENCE_DIR, load_yaml, resolve_path
from dsrv.metrics import METRIC_NAMES
from dsrv.mlflowinfo import (
    download_artifact,
    experiment_config_paths,
    latest_finished_runs,
    open_tracking,
)

CONCLUSIONS_YAML = EVIDENCE_DIR / "conclusiones.yaml"
EXPERIMENTS_CSV = EVIDENCE_DIR / "experimentos.csv"


def main() -> None:
    client = open_tracking()
    runs = latest_finished_runs(client)
    conclusions = load_yaml(CONCLUSIONS_YAML)

    rows = []
    for path in experiment_config_paths():
        config_id = path.stem
        run = runs[config_id]
        config = load_yaml(path)
        with tempfile.TemporaryDirectory() as tmp:
            matrix_path = download_artifact(run.info.run_id, "confusion_matrix.json", tmp)
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))["matriz"]
        (tn, fp), (fn, tp) = matrix
        rows.append(
            {
                "experiment_id": config_id.split("_")[0].upper(),
                "config_id": config_id,
                "hypothesis": config["hypothesis"],
                "family": config["family"],
                "change": config["change"],
                "params_path": path.as_posix(),
                "seed": config["seed"],
                "git_commit": run.data.tags["git_commit"],
                "data_version": run.data.tags["data_version"],
                "run_id": run.info.run_id,
                "threshold": config["threshold"],
                **{name: run.data.metrics[name] for name in METRIC_NAMES},
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "train_time_s": run.data.metrics["train_seconds"],
                "conclusion": conclusions[config_id],
            }
        )
    output = resolve_path(EXPERIMENTS_CSV)
    pd.DataFrame(rows).to_csv(output, index=False, lineterminator="\n")
    print(f"Escrito {EXPERIMENTS_CSV.as_posix()} con {len(rows)} filas")


if __name__ == "__main__":
    main()
