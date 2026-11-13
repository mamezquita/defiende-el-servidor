# Defiende el servidor

Clasificador binario de conexiones de red, normal (0) o ataque (1), entrenado con UNSW-NB15.

Es un proyecto académico de MLOps: lo que se evalúa es la **trazabilidad**, no la métrica. Cada
resultado se puede seguir hasta el código (Git), los datos (DVC), el experimento (MLflow) y el modelo
(Hugging Face), y las dos versiones publicadas, v1 y v2, se pueden recuperar en un entorno nuevo.

## Enlaces

| Qué | Dónde |
|---|---|
| Código | [github.com/mamezquita/defiende-el-servidor](https://github.com/mamezquita/defiende-el-servidor) |
| Datos (remote DVC) | [dagshub.com/mamezquita/defiende-el-servidor](https://dagshub.com/mamezquita/defiende-el-servidor) |
| Modelos | [huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor](https://huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor) |
| Dataset | [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |

## Resultado en una línea

v2 (gradient boosting, umbral 0,64) cumplió en validación la política de FPR <= 5 %, pero **en test
la FPR fue 18,80 %**, por desplazamiento de distribución entre los archivos de training y testing. La
recuperación de v1 y v2 en un entorno nuevo reprodujo las predicciones originales sin diferencias.

## Versiones publicadas

| | v1 | v2 |
|---|---|---|
| Configuración | `e1_lr_base`: regresión logística | `e4_hgb`: HistGradientBoosting |
| Umbral | 0,50, fijado de antemano | 0,64, elegido con la política de validación |
| Tag de código | `v1` → `7750fe4` | `v2` → `c99821b` |
| Commit de entrenamiento | `7750fe4d1bf98560829d1ab4c270e94ef2d61764` | `7750fe4d1bf98560829d1ab4c270e94ef2d61764` |
| Datos | `data-v2` | `data-v2` |
| Run de MLflow | `88648102f7934473aa0b3296937261b9` | `f2f0d4442fcd4aa28f4730c8eb2bc9d5` |
| Revisión de Hugging Face | `6010585cc8729c73d88d5b0d9acb360e0065a351` | `d0375af37f9abf49930b4ec76b7430446098f1f9` |
| SHA-256 del modelo | `2a968766…365d98ab` | `186e9c50…c908014f` |

Ambas versiones salieron del mismo código de entrenamiento y difieren en la configuración. Los SHA-256
completos están en [evidencias/seleccion.json](evidencias/seleccion.json).

La revisión `d0375af3…` de v2 es la que se evaluó en test. Después se publicó la revisión
`43e4338d…`, que solo añade los resultados de test a la model card: su `model.joblib` es idéntico.

## Resultados

En validación, con el umbral de cada versión (20.163 filas):

| Métrica | v1 | v2 |
|---|---:|---:|
| Recall de ataques | 0,9767 | 0,9035 |
| Precision de ataques | 0,8406 | 0,9473 |
| FPR | 0,1793 | 0,0487 |
| Macro F1 | 0,8970 | 0,9277 |
| Accuracy | 0,8974 | 0,9278 |

En test, solo v2 y una sola vez, después del tag `freeze-v2` (82.332 filas, sin deduplicar):

| Métrica | Validación | Test |
|---|---:|---:|
| Recall de ataques | 0,9035 | 0,9699 |
| Precision de ataques | 0,9473 | 0,8634 |
| **FPR** | 0,0487 | **0,1880** |
| Macro F1 | 0,9277 | 0,8960 |
| Accuracy | 0,9278 | 0,8990 |

El modelo y el umbral no se cambiaron después de ver el test. Las causas medidas están en
[evidencias/notas-test.md](evidencias/notas-test.md): el tráfico normal de test difiere del de train,
mientras que los duplicados y el solapamiento con train no explican la subida de la FPR.

## Cómo se construyó

1. **Datos (`data-v1`, `data-v2`).** Se auditaron los CSV originales y se versionaron con DVC. Después
   se limpiaron y se partieron:
   - se excluyen `label`, `attack_cat` e `id` de las entradas;
   - de cada grupo de duplicados exactos se conserva una fila (73.590 eliminadas);
   - se excluyen los grupos con etiquetas contradictorias (940 filas);
   - train (80.648 filas) y validación (20.163) se separan con `GroupShuffleSplit` 80/20, semilla 42,
     agrupando por hash de las entradas, sin ningún grupo en ambos lados.
2. **Experimentos.** Cuatro configuraciones, cada una con su hipótesis, registradas en MLflow.
   Codificadores, imputación y escalado se ajustan solo con train, dentro de un único `Pipeline`.
3. **Selección.** La política de umbral se fijó y se versionó antes de los experimentos
   ([configs/threshold_policy.yaml](configs/threshold_policy.yaml)): maximizar el recall con
   FPR <= 0,05 en validación, desempatando por macro F1.
4. **Publicación.** v1 y v2 se subieron a Hugging Face sin reentrenar, cada una en su revisión y con
   su tag, y se comprobó el SHA-256 del modelo descargado.
5. **Congelación y test.** El tag `freeze-v2` fija la selección. `dsrv.evaluate_test` se niega a
   ejecutarse si la selección cambió o si el test ya se evaluó.
6. **Recuperación.** Se clonó el repo en una carpeta nueva, con venv y caché nuevos, y se recuperaron
   datos y modelos. Resultado en [evidencias/recuperacion/RESULTADO.md](evidencias/recuperacion/RESULTADO.md).

| Experimento | Hipótesis | Recall | FPR | Con la política de umbral |
|---|---|---:|---:|---|
| E1 regresión logística | Una referencia lineal simple separa la mayoría de ataques | 0,9767 | 0,1793 | umbral 0,78: recall 0,7561 |
| E2 + `log1p` y `class_weight` | Menos asimetría mejora el recall sin disparar la FPR | 0,9528 | 0,1404 | umbral 0,79: recall 0,8008 |
| E3 RandomForest | Las interacciones no lineales reducen los falsos negativos | 0,9522 | 0,0883 | umbral 0,65: recall 0,8974 |
| E4 HistGradientBoosting | Iguala al RF con menor costo de inferencia | 0,9471 | 0,0817 | umbral 0,64: recall 0,9035 |

Recall y FPR de las primeras columnas son con umbral 0,5. Las conclusiones de cada experimento están en
[evidencias/experimentos.csv](evidencias/experimentos.csv).

## Instalación

Requiere WSL o Linux, Git y **Python 3.11**. Si `python3.11` no existe, instálalo con
[uv](https://docs.astral.sh/uv/):

```bash
uv python install 3.11
git clone https://github.com/mamezquita/defiende-el-servidor.git
cd defiende-el-servidor
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Todos los comandos se ejecutan desde la raíz del repo, con el venv activado.

### Datos

Los datos están en DagsHub y requieren un token de acceso. Las credenciales se guardan en
`.dvc/config.local`, que Git ignora:

```bash
read -rsp "Token de DagsHub: " DAGSHUB_TOKEN; echo
dvc remote modify origin --local access_key_id "$DAGSHUB_TOKEN"
dvc remote modify origin --local secret_access_key "$DAGSHUB_TOKEN"
unset DAGSHUB_TOKEN
dvc pull
python -m dsrv.verify_data --out /tmp/verificacion
```

`dsrv.verify_data` compara el SHA-256 de cada archivo con `auditoria.json` y `manifest.json`.

## Uso

### Recuperar v1 y v2 desde Hugging Face

No requiere sesión de Hugging Face. Compara las predicciones con las originales
(`y_pred` idéntico y diferencia de score <= 1e-9):

```bash
python -m dsrv.recover --version v1 --sha 6010585cc8729c73d88d5b0d9acb360e0065a351 \
  --inputs evidencias/recuperacion/entradas_fijas.csv \
  --reference evidencias/recuperacion/predicciones_originales_v1.csv --out /tmp/recuperacion
python -m dsrv.recover --version v2 --sha d0375af37f9abf49930b4ec76b7430446098f1f9 \
  --inputs evidencias/recuperacion/entradas_fijas.csv \
  --reference evidencias/recuperacion/predicciones_originales_v2.csv --out /tmp/recuperacion
```

La prueba completa, desde un clon nuevo, está en [scripts/recovery_test.sh](scripts/recovery_test.sh);
sus instrucciones de uso están al inicio del archivo.

### Predecir con un modelo

`dsrv.predict` comprueba el SHA-256 del modelo y la versión de scikit-learn antes de cargarlo. El CSV
debe tener `row_id` y las 42 columnas de entrada:

```bash
python -m dsrv.predict --model-dir <carpeta con model.joblib y config.json> \
  --input entradas.csv --output predicciones.csv
```

El código para usar el modelo directamente desde Hugging Face está en su model card.

## Reproducir

| Paso | Comando | ¿Se puede repetir? |
|---|---|---|
| Pruebas | `python -m pytest` | Sí: 163 pruebas |
| Limpieza y partición | `dvc repro` | Sí: da archivos idénticos byte a byte |
| Auditoría | `python -m dsrv.audit` | Sí |
| Entrenar una configuración | `python -m dsrv.train --config configs/e1_lr_base.yaml` | Sí: crea un run nuevo en `mlflow.db` |
| Exportar MLflow, seleccionar, tabla | `dsrv.export_mlflow`, `dsrv.select`, `dsrv.build_experiments_table` | `select` no, porque la selección ya se publicó |
| Publicar en Hugging Face | `dsrv.package_model`, `dsrv.publish_hf` | No: los tags `v1` y `v2` ya existen |
| Evaluar en test | `python -m dsrv.evaluate_test` | **No**: el test se evalúa una sola vez |
| Recalcular métricas de test | `python -m dsrv.recompute_metrics` | Sí: comprueba `metricas-test.json` |

Notas:
- **Antes de entrenar:** `dsrv.train` exige un árbol de trabajo limpio, y que los datos coincidan con
  `data-v2`.
- **Runs originales:** `mlflow.db` y `mlartifacts/` no se versionan; su exportación está en
  [evidencias/mlflow/](evidencias/mlflow/).
- **Ver la UI de MLflow:** `mlflow ui --backend-store-uri sqlite:///mlflow.db`, en la máquina donde se
  entrenó.

## Estructura

```
configs/      fuente de datos, cuatro experimentos y política de umbral
data/         datos gestionados por DVC (raw/ y prepared/)
evidencias/   resultados generados por código
scripts/      prueba de recuperación en entorno nuevo
src/dsrv/     código del proyecto (python -m dsrv.<modulo>)
tests/        pruebas con pytest
dvc.yaml      etapa prepare: limpieza y partición
```

## Evidencias

| Archivo | Contenido |
|---|---|
| [auditoria.json](evidencias/auditoria.json) | Huellas, duplicados, contradicciones, partición, política de limpieza |
| [experimentos.csv](evidencias/experimentos.csv) | Una fila por experimento, con hipótesis, métricas y conclusión |
| [mlflow/](evidencias/mlflow/) | Exportación de los runs (sin modelos) |
| [umbrales.csv](evidencias/umbrales.csv) | Umbral elegido por la política para cada experimento |
| [seleccion.json](evidencias/seleccion.json) | v1 y v2 congeladas, con referencias de Git, DVC y Hugging Face |
| [hf_revisiones.json](evidencias/hf_revisiones.json) | Revisiones de Hugging Face y verificación de descarga |
| [notas-seleccion.md](evidencias/notas-seleccion.md) | Cautelas de la selección |
| [metricas-test.json](evidencias/metricas-test.json) | Evaluación única en test, con filas vistas y no vistas |
| [predicciones-test.csv](evidencias/predicciones-test.csv) | Predicción de cada fila de test |
| [desplazamiento-test.json](evidencias/desplazamiento-test.json) | Análisis descriptivo de las diferencias entre test y train |
| [notas-test.md](evidencias/notas-test.md) | Por qué el test empeoró frente a validación |
| [recuperacion/](evidencias/recuperacion/) | Prueba de recuperación: entradas, predicciones, log y resultado |

## Limitaciones

- **Dataset histórico:** tráfico generado en un laboratorio en 2015; no representa el tráfico actual.
- **FPR en test:** v2 no cumple en test el límite de FPR con el que se eligió, y validación no
  anticipó ese desplazamiento.
- **Solapamiento con test:** el 8,0 % de las filas de test tiene entradas idénticas a train y el 2,1 %
  a validación.
- **Recuperación probada en la misma máquina:** clon, venv y caché fueron nuevos, pero no se probó en
  otra máquina ni en otro sistema operativo.
- **Pickles:** los modelos son pickles de joblib, y cargarlos puede ejecutar código. Fija la revisión
  por SHA y comprueba el SHA-256 antes de cargarlos.

## Atribución

Datos: Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion
detection systems. *Military Communications and Information Systems Conference (MilCIS)*. Consulta
las condiciones de uso en la [página del dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
