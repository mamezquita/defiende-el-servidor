# Defiende el servidor

Clasificación binaria normal (0) / ataque (1) con UNSW-NB15. El propósito del proyecto es la
trazabilidad: código (Git/GitHub), datos (DVC + DagsHub), experimentos (MLflow) y modelos
(Hugging Face Hub), y la recuperación de v1 y v2 en un entorno nuevo.

## Entorno

Python 3.11 es obligatorio. Si no existe el lanzador `py`, instala Python con uv y crea el entorno
con `python3.11`:

```bash
uv python install 3.11
python3.11 -m venv .venv
```

> README provisional: se completa en la fase 9.
