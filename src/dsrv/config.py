"""Constantes del proyecto y carga de configuración YAML."""

from pathlib import Path
from typing import Any

import yaml

SEED = 42

TARGET_COLUMN = "label"
EXCLUDED_COLUMNS = ("label", "attack_cat", "id")

# Rutas relativas a la raíz del repo; usar resolve_path() para abrirlas.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = Path("configs")
RAW_DIR = Path("data/raw")
PREPARED_DIR = Path("data/prepared")
EVIDENCE_DIR = Path("evidencias")
TRAINING_CSV = RAW_DIR / "UNSW_NB15_training-set.csv"
TESTING_CSV = RAW_DIR / "UNSW_NB15_testing-set.csv"


def resolve_path(path: str | Path) -> Path:
    """Devuelve la ruta absoluta de una ruta relativa a la raíz del repo."""
    return REPO_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Carga un YAML cuya raíz sea un mapeo. Las rutas relativas parten de la raíz del repo."""
    with resolve_path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: se esperaba un mapeo en la raíz del YAML")
    return data
