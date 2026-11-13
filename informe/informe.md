# Defiende el servidor: trazabilidad MLOps con UNSW-NB15

**Integrantes:** Miguel Ángel Amézquita Briceño - Grupo 4-6 p. m.

Código: [github.com/mamezquita/defiende-el-servidor](https://github.com/mamezquita/defiende-el-servidor) · Modelos: [huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor](https://huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor) · Fecha: 2026-11-13

## 1. Datos y limpieza

Se auditaron los CSV oficiales de UNSW-NB15: training con 175.341 filas y testing con 82.332, ambos con el SHA-256 esperado, el mismo esquema y sin faltantes. Se excluyen de las entradas `label`, `attack_cat` e `id`, y quedan 42 variables. En training, de cada grupo de duplicados exactos se conservó una fila: se eliminaron 73.590. Se excluyeron 229 grupos contradictorios (mismas entradas, etiquetas 0 y 1), que suman 940 filas. Train y validación salen solo del archivo de training, con `GroupShuffleSplit` 80/20 (semilla 42) agrupado por el hash SHA-256 de las entradas: ningún grupo queda en ambos lados.

| Partición | Filas | Ataques |
|---|---:|---:|
| Training original | 175.341 | 68,1 % |
| Train | 80.648 | 48,6 % |
| Validación | 20.163 | 49,2 % |
| Excluidas | 74.530 | 94,2 % |
| Test (sin modificar) | 82.332 | 55,1 % |

**Cambio de balance:** la deduplicación baja la proporción de ataques de 68,1 % a 48,6 % en train, porque casi todos los duplicados son ataques; test tiene 55,1 %. **Cruces con test:** el 8,0 % de las filas de test tiene entradas idénticas a filas de train y el 2,1 % a validación; por eso test se reporta también separando filas vistas y no vistas.

**Disponibilidad de los datos:** el remote DVC en DagsHub está privado por una caída del servicio y se hará público apenas funcione. Mientras tanto, los datos se recuperan con un ZIP de respaldo del remote (dvc-remote-respaldo.zip, SHA-256 `1f1726747a2c5dc6…`), probado con un clon nuevo desde GitHub: `dvc pull` desde el ZIP dio OK en las huellas de los 5 archivos.

## 2. Cuatro experimentos

Cada configuración se registró en MLflow (4 runs, todos con estado FINISHED) con su hipótesis, el commit, `data_version` y la semilla. Métricas de validación con umbral 0,5.

| Exp. | Hipótesis | Recall | FPR | Macro F1 | Conclusión |
|---|---|---:|---:|---:|---|
| E1 | Una referencia lineal simple separa la mayoría de ataques. | 0,9767 | 0,1793 | 0,8970 | Hipótesis confirmada a medias: con umbral 0,5 detecta el 97,7 % de los ataques, pero con una FPR de 17,9 %. Para cumplir FPR <= 0,05 necesita umbral 0,78 y el recall cae a 75,6 %. |
| E2 | Reducir la asimetría de bytes/duración mejora el recall sin disparar la FPR. | 0,9528 | 0,1404 | 0,9054 | Hipótesis no confirmada con umbral 0,5: frente a E1 el recall baja de 97,7 % a 95,3 % (falsos negativos de 231 a 468), aunque la FPR baja de 17,9 % a 14,0 %. Con FPR <= 0,05 sí mejora el recall (80,1 % frente a 75,6 %), pero log1p y class_weight cambian a la vez, así que no se pueden separar sus efectos. |
| E3 | Las interacciones no lineales entre variables de flujo reducen los falsos negativos. | 0,9522 | 0,0883 | 0,9316 | Hipótesis confirmada solo a FPR comparable: con umbral 0,5 los falsos negativos no bajan frente a E2 (474 frente a 468), aunque la FPR cae a 8,8 %. Con FPR <= 0,05 los falsos negativos sí se reducen frente a E2, de 1.976 a 1.018 (recall 89,7 %). |
| E4 | Boosting iguala al RF con menor costo de inferencia. | 0,9471 | 0,0817 | 0,9325 | Hipótesis confirmada en validación: con FPR <= 0,05 alcanza recall 90,4 % frente a 89,7 % del RF, con inferencia de 0,38 s frente a 0,56 s en 20.163 filas (una medición, un hilo) y un modelo de 1,1 MB frente a 104,9 MB. Es la v2 elegida, con umbral 0,64. |

## 3. Cambios entre v1 y v2

**v1** es `e1_lr_base` (regresión logística, umbral 0,50), la referencia fijada de antemano. **v2** es `e4_hgb` (HistGradientBoosting con OrdinalEncoder, umbral 0,64). Cambian el modelo, el preprocesamiento y el umbral; los datos (`data-v2`) y el código de entrenamiento son los mismos.

| Validación | v1 | v2 | Cambio (puntos) |
|---|---:|---:|---:|
| Recall de ataques | 97,67 % | 90,35 % | −7,32 |
| Precision de ataques | 84,06 % | 94,73 % | +10,66 |
| FPR | 17,93 % | 4,87 % | −13,06 |
| Macro F1 | 89,70 % | 92,77 % | +3,07 |
| Accuracy | 89,74 % | 92,78 % | +3,04 |

**Mejora:** la FPR baja de 17,93 % a 4,87 %, con 1.338 falsas alarmas menos, y suben precision, macro F1 y accuracy. **Empeora:** el recall baja de 97,67 % a 90,35 %: 726 ataques más sin detectar. **Errores aceptados por v2 en validación:** 957 ataques no detectados (9,65 % de los ataques) y 499 falsas alarmas (4,87 % del tráfico normal).

## 4. Elección del umbral

La política se versionó en el commit `826cd3d`, 3 minutos antes del primer run: maximizar `recall_attack` con `fpr <= 0,05` en validación, desempatando por `macro_f1`. Si ningún umbral cumple, el de menor FPR con recall >= 0,90; si tampoco, 0,50. Se probaron 91 umbrales (0,05 a 0,95) en cada experimento.

![Recall y FPR de v2 (`e4_hgb`) en validación según el umbral.](figuras/umbral_v2.png)

*Figura 1. Recall y FPR de v2 (`e4_hgb`) en validación según el umbral.*

| Exp. | Umbral | Recall | FPR | Macro F1 | Regla |
|---|---:|---:|---:|---:|---|
| e1_lr_base | 0,78 | 0,7561 | 0,0454 | 0,8551 | restriccion |
| e2_lr_log_balanced | 0,79 | 0,8008 | 0,0491 | 0,8761 | restriccion |
| e3_rf | 0,65 | 0,8974 | 0,0490 | 0,9245 | restriccion |
| e4_hgb | 0,64 | 0,9035 | 0,0487 | 0,9277 | restriccion |

Los cuatro cumplen la restricción; v2 es el de mayor recall (0,9035). Su ventaja sobre el segundo es pequeña: 0,6 puntos de recall.

## 5. Test final

Evaluación única de v2, después del tag `freeze-v2`, sobre las 82.332 filas de test, sin deduplicar, con la revisión de Hugging Face `d0375af37f9a…`. **La FPR en test (18,80 %) supera el límite de 5 %** con el que se eligió el umbral. No se cambió nada después de ver el resultado.

| Métrica | Validación | Test | Cambio (puntos) |
|---|---:|---:|---:|
| Recall de ataques | 90,35 % | 96,99 % | +6,64 |
| Precision de ataques | 94,73 % | 86,34 % | −8,39 |
| FPR | 4,87 % | 18,80 % | +13,93 |
| Macro F1 | 92,77 % | 89,60 % | −3,17 |
| Accuracy | 92,78 % | 89,90 % | −2,88 |

| Test | Predicho normal | Predicho ataque |
|---|---:|---:|
| Real normal | 30.044 (TN) | 6.956 (FP) |
| Real ataque | 1.363 (FN) | 43.969 (TP) |

| Filas de test (descriptivo) | Filas | Recall | Precision | FPR |
|---|---:|---:|---:|---:|
| Vistas en train o validación | 8.313 | 99,83 % | 96,66 % | 19,52 % |
| No vistas | 74.019 | 96,47 % | 84,62 % | 18,77 % |

La FPR es casi igual en filas vistas y no vistas, así que el solapamiento no explica la subida. Sin duplicados, la FPR sigue en 17,98 %. La explicación con más respaldo es el desplazamiento de distribución del tráfico normal: la distancia KS mediana frente a train es 0,131 en test y 0,007 en validación (evidencias/notas-test.md).

## 6. Limitaciones

- **Dataset histórico:** UNSW-NB15 es tráfico de laboratorio de 2015; no representa el tráfico ni los ataques actuales.
- **Solapamiento con test:** el 8,0 % de test coincide con train y el 2,1 % con validación.
- **Umbral cerca del límite:** FPR de 4,87 % en validación frente a 5 %, elegido entre 4 x 91 combinaciones; en test fue mayor.
- **E2 con dos cambios a la vez:** `log1p` y `class_weight`, así que sus efectos no se pueden separar.
- **MLflow local:** `mlflow.db` y `mlartifacts/` no se versionan; los 4 runs se exportaron a evidencias/mlflow/ sin modelos.
- **Recuperación en la misma máquina:** v1 con 40/40 `y_pred` iguales; v2 con 40/40 `y_pred` iguales, en un clon, venv y caché nuevos, pero sin probar otra máquina u otro sistema operativo.
- **Datos en DagsHub:** privado por una caída del servicio; mientras tanto, ZIP de respaldo.

## Declaración de uso de IA

Usé Claude Code (Anthropic) como asistente de programación durante todo el taller. Bajo mis instrucciones, fase por fase, la herramienta escribió el código, las pruebas, los scripts y la documentación, y ejecutó los comandos de verificación. Yo definí las fases y sus requisitos, tomé las decisiones del proyecto (política de limpieza, política de umbral, selección de v2, publicación y visibilidad de los repositorios), revisé y aprobé cada resultado y cada commit, y ejecuté los pasos que requerían credenciales. Todas las cifras del informe salen del código y de las evidencias del repositorio, no de la herramienta.

## Contribución de cada integrante

Trabajo individual: Miguel Ángel Amézquita Briceño realizó todas las fases del taller.

<div style="page-break-after: always"></div>

## Referencias

- Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*. [research.unsw.edu.au/projects/unsw-nb15-dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. Versión 1.9.1.
- Iterative. DVC: Data Version Control. [dvc.org](https://dvc.org). Versión 3.67.1.
- Zaharia, M. et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. *IEEE Data Engineering Bulletin*, 41(4), 39–45. Versión 3.18.0.
- Hugging Face. huggingface_hub. [huggingface.co/docs/huggingface_hub](https://huggingface.co/docs/huggingface_hub). Versión 2.4.0.
