# Resultado de la prueba de recuperación

**Resultado: SUPERADA.**

Archivo generado por `python -m dsrv.recovery_report` a partir de los archivos de esta carpeta.

- **Repositorio:** https://github.com/mamezquita/defiende-el-servidor.git
- **Commit probado:** `c48ecfb187274bbafd94d708b2e321b8d9d5ee25`
- **Repositorio de modelos:** MiguelAngelAB/unsw-nb15-defiende-servidor
- **Inicio:** 2026-11-04T15:32:15-05:00
- **Fin:** 2026-11-04T15:36:58-05:00
- **Descargas de Hugging Face:** sin sesión (repos públicos), con una caché nueva.
- **Credenciales:** solo `dvc pull` usó un token de DagsHub, que se borró del clon al terminar.
- **Entorno:** clon, venv y caché nuevos, pero el entorno nuevo comparte máquina y sistema operativo
  con el entrenamiento (WSL). No prueba la recuperación en otra máquina u otro sistema operativo.

## Coincidencias con las predicciones originales

| Versión | SHA de Hugging Face | Filas | `row_id` iguales | `y_pred` iguales | Diferencia máxima de score | Tolerancia | Umbral | SHA-256 del modelo | Resultado |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| v1 | `6010585cc8729c73d88d5b0d9acb360e0065a351` | 40 | 40/40 | 40/40 | 0.000e+00 | 1e-09 | 0.5 | `2a968766d5e661ebfe05b174370c5272fc99bdd0befeafc4d17cdc2e365d98ab` | OK |
| v2 | `d0375af37f9abf49930b4ec76b7430446098f1f9` | 40 | 40/40 | 40/40 | 0.000e+00 | 1e-09 | 0.64 | `186e9c50ef5634a2135f299a555de16f0e1f76de7c5c8bbb8f43e2c1c908014f` | OK |

## Datos recuperados con DVC

Resultado: **OK**. `manifest.json`: `6d95f322515e66352c5fd40298904697ffedc2b9e4042ee7ad5d38561786356d`.

| Archivo | SHA-256 calculado | Coincide |
|---|---|---|
| `data/raw/UNSW_NB15_training-set.csv` | `bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa` | sí |
| `data/raw/UNSW_NB15_testing-set.csv` | `734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559` | sí |
| `data/prepared/train.parquet` | `b218b41fd9db3fd8de91dcb0f74b83335cf1e3aab2204f9c690e71067dc87369` | sí |
| `data/prepared/val.parquet` | `19156851bc6b1c05c1a56599e10078172c5b722d834779c26d0fac95bab136da` | sí |
| `data/prepared/splits.csv` | `f8164b870e1fc390d79a23e24d16dc32da1c69dbdea2c09f605ea4d6d328e49e` | sí |

## Tiempos

Total: 282 s.

| Paso | Segundos | Estado |
|---|---:|---|
| clon_y_tags | 15 | ok |
| entorno | 215 | ok |
| datos_dvc | 37 | ok |
| cache_hugging_face | 0 | ok |
| recuperacion_modelos | 15 | ok |
