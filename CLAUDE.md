# Proyecto: Defiende el servidor (MLOps · UNSW-NB15)

## Objetivo
Clasificación binaria normal (0) / ataque (1) con UNSW-NB15. Se califica la trazabilidad:
código (Git/GitHub), datos (DVC + DagsHub), experimentos (MLflow) y modelos (Hugging Face Hub),
y la recuperación de v1 y v2 en un entorno nuevo. La métrica alta no es el objetivo.

## Entorno
- WSL (Ubuntu), bash, VS Code. Python 3.11 en .venv. Todos los comandos en sintaxis bash.
- Rutas relativas a la raíz del repo; nunca rutas personales (/home/...).
- Dependencias con versión exacta en requirements.txt.

## Reglas de datos (no negociables)
1. Excluir de las entradas: label, attack_cat, id. Objetivo = label. Clase positiva = ataque (1).
2. Train y validación salen SOLO de UNSW_NB15_training-set.csv. Split agrupado por hash SHA-256
   de las variables predictoras (GroupShuffleSplit 80/20, semilla 42). Ningún grupo en ambos lados.
3. Grupos con etiquetas contradictorias: excluidos de train/validación y documentados en auditoria.json.
4. Codificadores, imputación, escalado y selección se ajustan solo con train. El umbral, solo con validación.
5. El test (UNSW_NB15_testing-set.csv) no decide nada. Se evalúa una sola vez, completo,
   después del tag freeze-v2.
6. Preprocesamiento + estimador = un único sklearn Pipeline guardado con joblib.

## Reglas de trazabilidad
- Antes de entrenar o evaluar: `git status --porcelain` debe estar vacío. Si no, detenerse y pedir commit.
- Cada run de MLflow registra tags: hypothesis, change, family, config_id, git_commit (SHA completo),
  data_version (tag git + md5 de los .dvc), seed; params; métricas de validación; artefactos:
  pipeline completo, matriz de confusión, config usada.
- Métricas: recall_attack, precision_attack, fpr, macro_f1, accuracy (solo complementaria) y matriz de
  confusión. Denominador cero -> 0.0 con zero_division=0 y flag registrado.
- Toda evidencia en evidencias/ como CSV/JSON válidos generados por código, nunca escritos a mano.
- Hashes SHA-256 en hexadecimal minúscula, calculados por bloques.

## Seguridad
- Nunca leer, imprimir, escribir ni commitear tokens o contraseñas. Las credenciales viven en
  `hf auth login` y .dvc/config.local (ignorado). Si un paso necesita credencial, dame el comando
  para que yo lo ejecute.
- Prohibido: git push --force, borrar tags, reescribir historial.

## Forma de trabajo
- Una fase a la vez. Al cerrar cada fase: correr la verificación, resumir resultados y proponer
  mensaje de commit (feat:, data:, exp:, model:, docs:, fix:). Commit solo con mi aprobación.
- Código en src/dsrv/, ejecutable con `python -m dsrv.<modulo>`; configuración en configs/*.yaml.
- Si cambias un archivo, edita solo la parte necesaria.
- Antes de usar MLflow, DVC o huggingface_hub, revisa la versión instalada: sus APIs cambian.