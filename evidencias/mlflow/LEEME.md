# Evidencias de MLflow

Archivos generados por `python -m dsrv.export_mlflow`.

- `runs.csv`: todos los runs del experimento `defiende-el-servidor`, incluidos los fallidos (tag `status=failed`).
- `<run_id>/run.json`: info, params, metrics y tags del run.
- `<run_id>/`: config, `confusion_matrix.json`, `confusion_matrix.png` y `threshold_sweep.csv`.

No se incluyen modelos. Tampoco las rutas absolutas ni el usuario del sistema (`artifact_uri`,
`mlflow.source.name`, `mlflow.user`).

## Cruzar con experimentos.csv

La columna `run_id` de `evidencias/experimentos.csv` es el nombre de la carpeta `<run_id>/` y
coincide con la columna `run_id` de `runs.csv`. Las métricas de `experimentos.csv` son las columnas
`metrics.*` de esa fila. Si una configuración tiene varios runs, `experimentos.csv` usa el último
terminado.

## Abrir la UI

Desde la raíz del repo, en la máquina donde se ejecutaron los experimentos (`mlflow.db` y
`mlartifacts/` no se versionan):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
