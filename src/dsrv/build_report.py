"""Genera el informe y entrega.md desde evidencias/. Se ejecuta con `python -m dsrv.build_report`.

Todas las cifras salen de los archivos de evidencias/; ninguna se escribe a mano. Escribe
informe/informe.md, informe/informe.html (autocontenido, para el PDF), la figura de umbral y
entrega.md. Los datos manuales (integrantes, declaración de IA, contribución) vienen de
informe/datos_manuales.yaml.
"""

import base64
import datetime as dt
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from matplotlib.figure import Figure

from dsrv.config import EVIDENCE_DIR, load_yaml, resolve_path
from dsrv.gitinfo import _git

REPORT_DIR = Path("informe")
FIGURE_PATH = REPORT_DIR / "figuras" / "umbral_v2.png"
MANUAL_DATA = REPORT_DIR / "datos_manuales.yaml"
DELIVERY_FILE = Path("entrega.md")
POLICY_PATH = Path("configs/threshold_policy.yaml")
PREPARED_MANIFEST = Path("data/prepared/manifest.json")
GITHUB_REPO = "https://github.com/mamezquita/defiende-el-servidor"
DAGSHUB_REPO = "https://dagshub.com/mamezquita/defiende-el-servidor"
HF_URL = "https://huggingface.co/MiguelAngelAB/unsw-nb15-defiende-servidor"
DATASET_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

RECALL_COLOR = "#2a78d6"
FPR_COLOR = "#eb6834"
INK = "#0b0b0b"
MUTED_INK = "#5f5e5a"
REFERENCE_INK = "#8a8985"
SURFACE = "#ffffff"

Block = tuple


# Formato numérico en español: coma decimal y punto de miles.
def num(value: float, decimals: int = 4) -> str:
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def pct(value: float, decimals: int = 1) -> str:
    return f"{num(100 * value, decimals)} %"


def count(value: int) -> str:
    return f"{int(value):,}".replace(",", ".")


def points(value: float, decimals: int = 2) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{num(abs(100 * value), decimals)}"


def _read_json(name: str) -> dict[str, Any]:
    return json.loads(resolve_path(EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def load_evidence() -> dict[str, Any]:
    """Lee todas las evidencias que usa el informe."""
    selection = _read_json("seleccion.json")
    v2 = next(item for item in selection["versiones"] if item["version"] == "v2")
    return {
        "audit": _read_json("auditoria.json"),
        "selection": selection,
        "test": _read_json("metricas-test.json"),
        "shift": _read_json("desplazamiento-test.json"),
        "backup": _read_json("respaldo-dvc.json"),
        "revisions": _read_json("hf_revisiones.json"),
        "recovery": [_read_json(f"recuperacion/comparacion_{v}.json") for v in ("v1", "v2")],
        "experiments": pd.read_csv(resolve_path(EVIDENCE_DIR / "experimentos.csv")),
        "thresholds": pd.read_csv(resolve_path(EVIDENCE_DIR / "umbrales.csv")),
        "runs": pd.read_csv(resolve_path(EVIDENCE_DIR / "mlflow" / "runs.csv")),
        "sweep": pd.read_csv(
            resolve_path(EVIDENCE_DIR / "mlflow" / v2["run_id"] / "threshold_sweep.csv")
        ),
        "policy": load_yaml(POLICY_PATH),
        "requirements": dict(
            line.strip().split("==")
            for line in resolve_path("requirements.txt").read_text(encoding="utf-8").splitlines()
            if "==" in line
        ),
    }


def _version(evidence: dict[str, Any], version: str) -> dict[str, Any]:
    return next(item for item in evidence["selection"]["versiones"] if item["version"] == version)


def draw_threshold_figure(evidence: dict[str, Any], path: Path) -> None:
    """Recall y FPR de v2 en validación según el umbral, con el umbral elegido y el límite."""
    sweep = evidence["sweep"]
    v2 = _version(evidence, "v2")
    limit = evidence["policy"]["restriccion"]["maximo"]
    chosen = v2["umbral"]
    row = sweep[sweep["threshold"] == chosen].iloc[0]

    fig = Figure(figsize=(6.4, 2.9), facecolor=SURFACE, layout="constrained")
    ax = fig.add_subplot(facecolor=SURFACE)
    ax.plot(sweep["threshold"], sweep["recall_attack"], color=RECALL_COLOR, lw=2, label="Recall de ataques")
    ax.plot(sweep["threshold"], sweep["fpr"], color=FPR_COLOR, lw=2, label="FPR")
    ax.axhline(limit, color=REFERENCE_INK, lw=1, ls=(0, (4, 3)))
    ax.axvline(chosen, color=REFERENCE_INK, lw=1, ls=(0, (4, 3)))
    ax.text(0.06, limit + 0.02, f"límite FPR {num(limit, 2)}", color=MUTED_INK, fontsize=8)
    ax.text(chosen + 0.01, 0.55, f"umbral elegido {num(chosen, 2)}", color=MUTED_INK, fontsize=8)
    for value, color in ((row["recall_attack"], RECALL_COLOR), (row["fpr"], FPR_COLOR)):
        ax.plot([chosen], [value], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=3)
    ax.text(chosen + 0.01, row["recall_attack"] + 0.03, num(row["recall_attack"], 3), color=INK, fontsize=8)
    ax.text(chosen + 0.01, row["fpr"] + 0.03, num(row["fpr"], 3), color=INK, fontsize=8)

    last = sweep.iloc[-1]
    ax.text(last["threshold"] + 0.01, last["recall_attack"], "Recall", color=INK, fontsize=8, va="center")
    ax.text(last["threshold"] + 0.01, last["fpr"], "FPR", color=INK, fontsize=8, va="center")

    ax.set_xlim(0.04, 1.03)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Umbral de score_attack", color=MUTED_INK, fontsize=9)
    ax.set_ylabel("Proporción", color=MUTED_INK, fontsize=9)
    ax.tick_params(colors=MUTED_INK, labelsize=8, length=0)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d3d2ce")
    ax.legend(loc="center left", frameon=False, fontsize=8, labelcolor=INK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)


def build_blocks(evidence: dict[str, Any], manual: dict[str, str], figure: Path) -> list[Block]:
    """Contenido del informe como bloques, para renderizar a Markdown y a HTML."""
    audit = evidence["audit"]
    raw = audit["archivos"]["training"]
    test_file = audit["archivos"]["testing"]
    splits = audit["splits"]
    crosses = audit["cruces_particiones"]
    contradictions = audit["contradicciones"]["training"]
    v1, v2 = _version(evidence, "v1"), _version(evidence, "v2")
    m1, m2 = v1["metricas_validacion"], v2["metricas_validacion"]
    test = evidence["test"]
    policy = evidence["policy"]
    limit = policy["restriccion"]["maximo"]
    backup = evidence["backup"]
    shift = evidence["shift"]
    runs = evidence["runs"]
    experiments = evidence["experiments"]
    thresholds = evidence["thresholds"]

    val_size = json.loads(resolve_path(PREPARED_MANIFEST).read_text(encoding="utf-8"))["val_size"]
    raw_attack = raw["conteo_label"]["1"] / raw["filas"]
    test_attack = test_file["conteo_label"]["1"] / test_file["filas"]
    seen_train = crosses["train"]["filas_test"] / test_file["filas"]
    seen_val = crosses["val"]["filas_test"] / test_file["filas"]
    policy_commit = _git("log", "--diff-filter=A", "--format=%h", "--", POLICY_PATH.as_posix())
    policy_time = pd.Timestamp(
        _git("log", "--diff-filter=A", "--format=%cI", "--", POLICY_PATH.as_posix())
    )
    first_run = pd.to_datetime(runs["start_time"]).min()
    minutes_before = (first_run - policy_time.tz_convert("UTC")).total_seconds() / 60

    blocks: list[Block] = [
        ("title", "Defiende el servidor: trazabilidad MLOps con UNSW-NB15"),
        ("meta", f"**Integrantes:** {manual['integrantes']}"),
        (
            "meta",
            f"Código: [{GITHUB_REPO.removeprefix('https://')}]({GITHUB_REPO}) · "
            f"Modelos: [{HF_URL.removeprefix('https://')}]({HF_URL}) · "
            f"Fecha: {dt.date.today().isoformat()}",
        ),
        ("h2", "1. Datos y limpieza"),
        (
            "p",
            f"Se auditaron los CSV oficiales de UNSW-NB15: training con {count(raw['filas'])} filas y "
            f"testing con {count(test_file['filas'])}, ambos con el SHA-256 esperado, el mismo esquema "
            f"y sin faltantes. Se excluyen de las entradas `label`, `attack_cat` e `id`, y quedan "
            f"{len(audit['variables']['usadas'])} variables. En training, de cada grupo de duplicados "
            f"exactos se conservó una fila: se eliminaron {count(splits['filas_eliminadas']['duplicado_exacto'])}. "
            f"Se excluyeron {count(contradictions['grupos'])} grupos contradictorios (mismas entradas, "
            f"etiquetas 0 y 1), que suman {count(splits['filas_eliminadas']['grupo_contradictorio'])} filas. "
            f"Train y validación salen solo del archivo de training, con `GroupShuffleSplit` "
            f"{round(100 * (1 - val_size))}/{round(100 * val_size)} (semilla {splits['semilla']}) agrupado por el hash SHA-256 de las entradas: ningún grupo "
            f"queda en ambos lados.",
        ),
        (
            "table",
            ["Partición", "Filas", "Ataques"],
            [
                ["Training original", count(raw["filas"]), pct(raw_attack)],
                ["Train", count(splits["tamanos"]["train"]), pct(splits["proporcion_clases"]["train"]["1"])],
                ["Validación", count(splits["tamanos"]["val"]), pct(splits["proporcion_clases"]["val"]["1"])],
                ["Excluidas", count(splits["tamanos"]["excluded"]), pct(splits["proporcion_clases"]["excluded"]["1"])],
                ["Test (sin modificar)", count(test_file["filas"]), pct(test_attack)],
            ],
            "lrr",
        ),
        (
            "p",
            f"**Cambio de balance:** la deduplicación baja la proporción de ataques de {pct(raw_attack)} "
            f"a {pct(splits['proporcion_clases']['train']['1'])} en train, porque casi todos los "
            f"duplicados son ataques; test tiene {pct(test_attack)}. **Cruces con test:** el "
            f"{pct(seen_train)} de las filas de test tiene entradas idénticas a filas de train y el "
            f"{pct(seen_val)} a validación; por eso test se reporta también separando filas vistas y no "
            f"vistas.",
        ),
        (
            "p",
            "**Disponibilidad de los datos:** el remote DVC en DagsHub está privado por una caída del "
            "servicio y se hará público apenas funcione. Mientras tanto, los datos se recuperan con un "
            f"ZIP de respaldo del remote ({backup['zip']}, SHA-256 `{backup['zip_sha256'][:16]}…`), "
            f"probado con un clon nuevo desde GitHub: `dvc pull` desde el ZIP dio "
            f"{backup['prueba']['resultado']} en las huellas de los "
            f"{len(backup['prueba']['verificacion_datos']['csv_originales_contra_auditoria']) + len(backup['prueba']['verificacion_datos']['datos_preparados_contra_manifest'])} "
            f"archivos.",
        ),
        ("h2", "2. Cuatro experimentos"),
        (
            "p",
            f"Cada configuración se registró en MLflow ({len(runs)} runs, todos con estado "
            f"{'/'.join(sorted(runs['status'].unique()))}) con su hipótesis, el commit, `data_version` y "
            "la semilla. Métricas de validación con umbral 0,5.",
        ),
        (
            "table",
            ["Exp.", "Hipótesis", "Recall", "FPR", "Macro F1", "Conclusión"],
            [
                [
                    row.experiment_id,
                    row.hypothesis,
                    num(row.recall_attack),
                    num(row.fpr),
                    num(row.macro_f1),
                    row.conclusion,
                ]
                for row in experiments.itertuples()
            ],
            "llrrrl",
            "small",
        ),
        ("h2", "3. Cambios entre v1 y v2"),
        (
            "p",
            f"**v1** es `{v1['config_id']}` (regresión logística, umbral {num(v1['umbral'], 2)}), la "
            f"referencia fijada de antemano. **v2** es `{v2['config_id']}` (HistGradientBoosting con "
            f"OrdinalEncoder, umbral {num(v2['umbral'], 2)}). Cambian el modelo, el preprocesamiento y el "
            "umbral; los datos (`data-v2`) y el código de entrenamiento son los mismos.",
        ),
        (
            "table",
            ["Validación", "v1", "v2", "Cambio (puntos)"],
            [
                [label, pct(m1[key], 2), pct(m2[key], 2), points(m2[key] - m1[key])]
                for label, key in (
                    ("Recall de ataques", "recall_attack"),
                    ("Precision de ataques", "precision_attack"),
                    ("FPR", "fpr"),
                    ("Macro F1", "macro_f1"),
                    ("Accuracy", "accuracy"),
                )
            ],
            "lrrr",
        ),
        (
            "p",
            f"**Mejora:** la FPR baja de {pct(m1['fpr'], 2)} a {pct(m2['fpr'], 2)}, con "
            f"{count(m1['confusion_matrix'][0][1] - m2['confusion_matrix'][0][1])} falsas alarmas menos, y "
            f"suben precision, macro F1 y accuracy. **Empeora:** el recall baja de {pct(m1['recall_attack'], 2)} "
            f"a {pct(m2['recall_attack'], 2)}: {count(m2['confusion_matrix'][1][0] - m1['confusion_matrix'][1][0])} "
            f"ataques más sin detectar. **Errores aceptados por v2 en validación:** "
            f"{count(m2['confusion_matrix'][1][0])} ataques no detectados "
            f"({pct(m2['confusion_matrix'][1][0] / sum(m2['confusion_matrix'][1]), 2)} de los ataques) y "
            f"{count(m2['confusion_matrix'][0][1])} falsas alarmas ({pct(m2['fpr'], 2)} del tráfico normal).",
        ),
        ("h2", "4. Elección del umbral"),
        (
            "p",
            f"La política se versionó en el commit `{policy_commit}`, {count(round(minutes_before))} minutos "
            f"antes del primer run: maximizar `{policy['objetivo']['maximizar']}` con "
            f"`{policy['restriccion']['metrica']} <= {num(limit, 2)}` en validación, desempatando por "
            f"`{policy['desempate']}`. Si ningún umbral cumple, el de menor FPR con recall >= "
            f"{num(policy['respaldo']['recall_attack_minimo'], 2)}; si tampoco, "
            f"{num(policy['respaldo']['umbral_por_defecto'], 2)}. Se probaron {len(evidence['sweep'])} "
            f"umbrales ({num(evidence['sweep']['threshold'].min(), 2)} a "
            f"{num(evidence['sweep']['threshold'].max(), 2)}) en cada experimento.",
        ),
        ("fig", figure, f"Recall y FPR de v2 (`{v2['config_id']}`) en validación según el umbral."),
        (
            "table",
            ["Exp.", "Umbral", "Recall", "FPR", "Macro F1", "Regla"],
            [
                [
                    row.config_id,
                    num(row.umbral, 2),
                    num(row.recall_attack),
                    num(row.fpr),
                    num(row.macro_f1),
                    row.regla,
                ]
                for row in thresholds.itertuples()
            ],
            "lrrrrl",
            "small",
        ),
        (
            "p",
            f"Los cuatro cumplen la restricción; v2 es el de mayor recall ({num(m2['recall_attack'])}). "
            f"Su ventaja sobre el segundo es pequeña: "
            f"{num(100 * (thresholds['recall_attack'].max() - thresholds['recall_attack'].nlargest(2).iloc[-1]), 1)} "
            f"puntos de recall.",
        ),
        ("h2", "5. Test final"),
        (
            "p",
            f"Evaluación única de v2, después del tag `{test['freeze_tag']}`, sobre las "
            f"{count(test['filas_evaluadas'])} filas de test, sin deduplicar, con la revisión de Hugging "
            f"Face `{test['hf_sha'][:12]}…`. **La FPR en test ({pct(test['fpr'], 2)}) supera el límite de "
            f"{pct(limit, 0)}** con el que se eligió el umbral. No se cambió nada después de ver el resultado.",
        ),
        (
            "table",
            ["Métrica", "Validación", "Test", "Cambio (puntos)"],
            [
                [label, pct(m2[key], 2), pct(test[key], 2), points(test[key] - m2[key])]
                for label, key in (
                    ("Recall de ataques", "recall_attack"),
                    ("Precision de ataques", "precision_attack"),
                    ("FPR", "fpr"),
                    ("Macro F1", "macro_f1"),
                    ("Accuracy", "accuracy"),
                )
            ],
            "lrrr",
        ),
        (
            "table",
            ["Test", "Predicho normal", "Predicho ataque"],
            [
                ["Real normal", f"{count(test['matriz_confusion'][0][0])} (TN)", f"{count(test['matriz_confusion'][0][1])} (FP)"],
                ["Real ataque", f"{count(test['matriz_confusion'][1][0])} (FN)", f"{count(test['matriz_confusion'][1][1])} (TP)"],
            ],
            "lrr",
        ),
        (
            "table",
            ["Filas de test (descriptivo)", "Filas", "Recall", "Precision", "FPR"],
            [
                [
                    label,
                    count(group["filas"]),
                    pct(group["recall_attack"], 2),
                    pct(group["precision_attack"], 2),
                    pct(group["fpr"], 2),
                ]
                for label, group in (
                    ("Vistas en train o validación", test["descriptivo_filas_vistas"]["vistas_en_train_o_val"]),
                    ("No vistas", test["descriptivo_filas_vistas"]["no_vistas"]),
                )
            ],
            "lrrrr",
        ),
        (
            "p",
            f"La FPR es casi igual en filas vistas y no vistas, así que el solapamiento no explica la "
            f"subida. Sin duplicados, la FPR sigue en {pct(shift['metricas_test_filas_unicas']['fpr'], 2)}. "
            f"La explicación con más respaldo es el desplazamiento de distribución del tráfico normal: "
            f"la distancia KS mediana frente a train es "
            f"{num(shift['desplazamiento_normales_filas_unicas']['mediana_ks_test'], 3)} en test y "
            f"{num(shift['desplazamiento_normales_filas_unicas']['mediana_ks_control'], 3)} en validación "
            f"(evidencias/notas-test.md).",
        ),
        ("h2", "6. Limitaciones"),
        (
            "ul",
            [
                "**Dataset histórico:** UNSW-NB15 es tráfico de laboratorio de 2015; no representa el "
                "tráfico ni los ataques actuales.",
                f"**Solapamiento con test:** el {pct(seen_train)} de test coincide con train y el "
                f"{pct(seen_val)} con validación.",
                f"**Umbral cerca del límite:** FPR de {pct(m2['fpr'], 2)} en validación frente a "
                f"{pct(limit, 0)}, elegido entre {len(thresholds)} x {len(evidence['sweep'])} "
                "combinaciones; en test fue mayor.",
                "**E2 con dos cambios a la vez:** `log1p` y `class_weight`, así que sus efectos no se "
                "pueden separar.",
                f"**MLflow local:** `mlflow.db` y `mlartifacts/` no se versionan; los {len(runs)} runs se "
                "exportaron a evidencias/mlflow/ sin modelos.",
                "**Recuperación en la misma máquina:** "
                + "; ".join(
                    f"{item['version']} con {item['coincidencias_y_pred']}/{item['filas']} `y_pred` iguales"
                    for item in evidence["recovery"]
                )
                + ", en un clon, venv y caché nuevos, pero sin probar otra máquina u otro sistema "
                "operativo.",
                "**Datos en DagsHub:** privado por una caída del servicio; mientras tanto, ZIP de respaldo.",
            ],
        ),
        ("h2", "Declaración de uso de IA"),
        ("p", manual["declaracion_uso_ia"]),
        ("h2", "Contribución de cada integrante"),
        ("p", manual["contribucion_integrantes"]),
        ("break",),
        ("h2", "Referencias"),
        (
            "ul",
            [
                "Moustafa, N. y Slay, J. (2015). UNSW-NB15: a comprehensive data set for network "
                "intrusion detection systems. *Military Communications and Information Systems "
                f"Conference (MilCIS)*. [{DATASET_URL.removeprefix('https://')}]({DATASET_URL})",
                "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of "
                f"Machine Learning Research*, 12, 2825–2830. Versión {evidence['requirements']['scikit-learn']}.",
                "Iterative. DVC: Data Version Control. [dvc.org](https://dvc.org). Versión "
                f"{evidence['requirements']['dvc[s3]']}.",
                "Zaharia, M. et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow. "
                "*IEEE Data Engineering Bulletin*, 41(4), 39–45. Versión "
                f"{evidence['requirements']['mlflow']}.",
                "Hugging Face. huggingface_hub. "
                "[huggingface.co/docs/huggingface_hub](https://huggingface.co/docs/huggingface_hub). "
                f"Versión {evidence['requirements']['huggingface_hub']}.",
            ],
        ),
    ]
    return blocks


def _inline_html(text: str) -> str:
    """Convierte el subconjunto de Markdown usado en línea: enlaces, `código`, **negrita**, *cursiva*."""
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)


def render_markdown(blocks: list[Block], figure_link: str) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "title":
            lines += [f"# {block[1]}", ""]
        elif kind in ("meta", "p"):
            lines += [block[1], ""]
        elif kind == "h2":
            lines += [f"## {block[1]}", ""]
        elif kind == "ul":
            lines += [*(f"- {item}" for item in block[1]), ""]
        elif kind == "table":
            headers, rows, align = block[1], block[2], block[3]
            separator = ["---:" if a == "r" else "---" for a in align]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(separator) + "|")
            lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
            lines.append("")
        elif kind == "fig":
            lines += [f"![{block[2]}]({figure_link})", "", f"*Figura 1. {block[2]}*", ""]
        elif kind == "break":
            lines += ['<div style="page-break-after: always"></div>', ""]
    return "\n".join(lines)


CSS = """
@page { size: A4; margin: 1.6cm 1.7cm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.38;
       color: #0b0b0b; background: #ffffff; margin: 0; }
h1 { font-size: 16pt; margin: 0 0 4pt; }
h2 { font-size: 12pt; margin: 11pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; }
.meta { color: #5f5e5a; margin: 0 0 2pt; text-align: left; }
ul { margin: 0 0 6pt; padding-left: 16pt; } li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin: 3pt 0 8pt; break-inside: avoid; }
th, td { border-bottom: 0.6pt solid #d3d2ce; padding: 2.5pt 6pt; vertical-align: top; }
th { text-align: left; border-bottom: 1pt solid #8a8985; }
td.r, th.r { text-align: right; white-space: nowrap; }
table.small { font-size: 8pt; } table.small td { padding: 2pt 4pt; }
code { font-family: Consolas, monospace; font-size: 0.92em; }
a { color: #1c5cab; text-decoration: none; }
figure { margin: 4pt 0 6pt; break-inside: avoid; } figure img { width: 88%; }
figcaption { font-size: 8.5pt; color: #5f5e5a; }
.break { break-after: page; }
"""


def render_html(blocks: list[Block], figure_bytes: bytes) -> str:
    parts: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "title":
            parts.append(f"<h1>{_inline_html(block[1])}</h1>")
        elif kind == "meta":
            parts.append(f'<p class="meta">{_inline_html(block[1])}</p>')
        elif kind == "p":
            parts.append(f"<p>{_inline_html(block[1])}</p>")
        elif kind == "h2":
            parts.append(f"<h2>{_inline_html(block[1])}</h2>")
        elif kind == "ul":
            items = "".join(f"<li>{_inline_html(item)}</li>" for item in block[1])
            parts.append(f"<ul>{items}</ul>")
        elif kind == "table":
            headers, rows, align = block[1], block[2], block[3]
            css = f' class="{block[4]}"' if len(block) > 4 else ""
            head = "".join(
                f'<th class="{a}">{_inline_html(h)}</th>' for h, a in zip(headers, align)
            )
            body = "".join(
                "<tr>"
                + "".join(f'<td class="{a}">{_inline_html(str(c))}</td>' for c, a in zip(row, align))
                + "</tr>"
                for row in rows
            )
            parts.append(f"<table{css}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif kind == "fig":
            data = base64.b64encode(figure_bytes).decode("ascii")
            parts.append(
                f'<figure><img alt="{html.escape(block[2])}" src="data:image/png;base64,{data}">'
                f"<figcaption>Figura 1. {_inline_html(block[2])}</figcaption></figure>"
            )
        elif kind == "break":
            parts.append('<div class="break"></div>')
    body = "\n".join(parts)
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        "<title>Defiende el servidor: informe</title>"
        f"<style>{CSS}</style></head><body>\n{body}\n</body></html>\n"
    )


def build_delivery(evidence: dict[str, Any]) -> str:
    """Redacta entrega.md: qué se entrega, dónde está y cómo obtener los datos."""
    backup = evidence["backup"]
    revisions = evidence["revisions"]["versiones"]
    return f"""# Entrega: Defiende el servidor

## Qué se entrega

| Qué | Dónde |
|---|---|
| Informe | `informe/informe.pdf` |
| Código y evidencias | [{GITHUB_REPO.removeprefix('https://')}]({GITHUB_REPO}) |
| Modelos v1 y v2 | [{HF_URL.removeprefix('https://')}]({HF_URL}) |
| Datos (remote DVC) | [{DAGSHUB_REPO.removeprefix('https://')}]({DAGSHUB_REPO}), privado por ahora |
| Respaldo de los datos | `{backup['zip']}` |

| Versión | Tag de código | Revisión de Hugging Face | SHA-256 del modelo |
|---|---|---|---|
| v1 | `v1` | `{revisions['v1']['hf_sha']}` | `{revisions['v1']['model_sha256']}` |
| v2 | `v2` | `{revisions['v2']['hf_sha']}` | `{revisions['v2']['model_sha256']}` |

## Datos: DagsHub privado y respaldo en ZIP

DagsHub está privado por una caída del servicio y se hará público apenas funcione. Mientras tanto,
los datos se recuperan con el ZIP de respaldo del remote DVC:

- **Archivo:** `{backup['zip']}` ({count(backup['zip_bytes'])} bytes, {backup['objetos_dvc_en_zip']} objetos DVC)
- **SHA-256:** `{backup['zip_sha256']}`
- **Prueba:** {backup['prueba']['entorno']}; resultado **{backup['prueba']['resultado']}**
  (commit `{backup['prueba']['commit_clonado']}`). Detalle en `evidencias/respaldo-dvc.json`.

Uso, también explicado en el `LEEME.txt` del ZIP:

```bash
python3 -m zipfile -e {backup['zip']} <carpeta>
git clone {GITHUB_REPO}.git
cd defiende-el-servidor
# Instalar según el README, y luego:
dvc remote add -d respaldo <carpeta>/dvc-respaldo
dvc pull
python -m dsrv.verify_data --out verificacion
```

Archivo generado por `python -m dsrv.build_report`.
"""


def main() -> None:
    evidence = load_evidence()
    manual = yaml.safe_load(resolve_path(MANUAL_DATA).read_text(encoding="utf-8"))

    figure = resolve_path(FIGURE_PATH)
    draw_threshold_figure(evidence, figure)
    blocks = build_blocks(evidence, manual, FIGURE_PATH)

    report_dir = resolve_path(REPORT_DIR)
    figure_link = FIGURE_PATH.relative_to(REPORT_DIR).as_posix()
    (report_dir / "informe.md").write_text(render_markdown(blocks, figure_link), encoding="utf-8")
    (report_dir / "informe.html").write_text(
        render_html(blocks, figure.read_bytes()), encoding="utf-8"
    )
    resolve_path(DELIVERY_FILE).write_text(build_delivery(evidence), encoding="utf-8")
    print("Escritos informe/informe.md, informe/informe.html, la figura y entrega.md")


if __name__ == "__main__":
    main()
