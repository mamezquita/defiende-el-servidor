import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from dsrv import recover as rv
from dsrv import recovery_report as report
from dsrv import verify_data as vd
from dsrv.config import REPO_ROOT, load_yaml
from dsrv.features import build_pipeline
from dsrv.hashing import sha256_file
from dsrv.predict import load_packaged_model, predict_frame

FEATURES = ["dur", "proto", "service", "state"]
SHA_V1 = "a" * 40
SHA_V2 = "b" * 40
SCRIPT = REPO_ROOT / "scripts" / "recovery_test.sh"


def make_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, rows)
    return pd.DataFrame(
        {
            "row_id": [f"train:{i}" for i in range(rows)],
            "dur": rng.random(rows) + label,
            "proto": rng.choice(["tcp", "udp"], rows),
            "service": rng.choice(["-", "dns"], rows),
            "state": rng.choice(["FIN", "CON"], rows),
            "y_true": label,
        }
    )


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Revisión simulada en Hugging Face, con su registro y sus archivos de referencia."""
    frame = make_frame()
    pipeline = build_pipeline(load_yaml("configs/e1_lr_base.yaml"), frame[FEATURES])
    pipeline.fit(frame[FEATURES], frame["y_true"])

    remote = tmp_path / "remote"
    remote.mkdir()
    joblib.dump(pipeline, remote / "model.joblib")
    sha = sha256_file(remote / "model.joblib")
    config = {
        "threshold": 0.5,
        "positive_class": {"label": 1, "name": "ataque"},
        "input_columns": FEATURES,
        "sklearn_version": sklearn.__version__,
        "model_sha256": sha,
    }
    (remote / "config.json").write_text(json.dumps(config), encoding="utf-8")

    revisions = {
        "hf_repo": "repo/prueba",
        "versiones": {
            "v1": {"hf_sha": SHA_V1, "model_sha256": sha},
            "v2": {"hf_sha": SHA_V2, "model_sha256": "0" * 64},
        },
    }
    registry = tmp_path / "hf_revisiones.json"
    registry.write_text(json.dumps(revisions), encoding="utf-8")
    monkeypatch.setattr(rv, "REVISIONS_JSON", registry)

    inputs = tmp_path / "entradas.csv"
    frame.to_csv(inputs, index=False)
    model, loaded = load_packaged_model(remote)
    reference = tmp_path / "referencia.csv"
    predict_frame(model, loaded, frame).to_csv(reference, index=False)

    requested = []

    def download(repo, name, revision=None, local_dir=None):
        requested.append((repo, name, revision))
        target = Path(local_dir) / name
        shutil.copy(remote / name, target)
        return str(target)

    class Workspace:
        pass

    workspace = Workspace()
    workspace.root, workspace.remote = tmp_path, remote
    workspace.inputs, workspace.reference = inputs, reference
    workspace.download, workspace.requested = download, requested
    workspace.out = tmp_path / "salida"
    return workspace


def run(hub, **changes):
    arguments = {
        "version": "v1",
        "sha": SHA_V1,
        "inputs": hub.inputs,
        "reference": hub.reference,
        "output_dir": hub.out,
        "download": hub.download,
        **changes,
    }
    return rv.recover(**arguments)


def test_recovers_and_matches_reference_with_declared_tolerance(hub):
    comparison = run(hub)

    assert comparison["resultado"] == "OK"
    assert comparison["coincidencias_y_pred"] == comparison["filas"] == 40
    assert comparison["coincidencias_row_id"] == 40
    assert comparison["diferencia_maxima_score"] == 0
    assert comparison["tolerancia_score"] == 1e-9
    assert all(revision == SHA_V1 for _, _, revision in hub.requested)
    assert {name for _, name, _ in hub.requested} == {"model.joblib", "config.json"}


def test_writes_predictions_and_comparison_with_hashes(hub):
    run(hub)
    written = json.loads((hub.out / "comparacion_v1.json").read_text(encoding="utf-8"))
    predictions = hub.out / "predicciones_recuperadas_v1.csv"

    assert predictions.exists()
    assert written["sha256_salida"] == sha256_file(predictions)
    assert written["sha256_entradas"] == sha256_file(hub.inputs)
    assert written["sha256_modelo"] == sha256_file(hub.remote / "model.joblib")
    assert written["sha256_referencia"] == sha256_file(hub.reference)
    assert "/" not in json.dumps({k: written[k] for k in written if k != "hf_repo"})


@pytest.mark.parametrize("sha", ["abc123", "A" * 40, "g" * 40, "v1", ""])
def test_rejects_anything_but_a_full_sha(hub, sha):
    with pytest.raises(rv.RecoveryError, match="SHA completo"):
        run(hub, sha=sha)
    assert hub.requested == []


def test_rejects_a_revision_other_than_the_registered_one(hub):
    with pytest.raises(rv.RecoveryError, match="no es el registrado"):
        run(hub, sha="c" * 40)
    assert hub.requested == []


def test_rejects_another_hf_repo(hub):
    with pytest.raises(rv.RecoveryError, match="repo"):
        run(hub, hf_repo="otro/repo")


def test_rejects_model_that_differs_from_config_before_loading(hub):
    (hub.remote / "model.joblib").write_bytes(b"no es un pickle")

    with pytest.raises(rv.RecoveryError, match="config.json"):
        run(hub)
    assert not hub.out.exists()


def test_rejects_model_that_differs_from_registry(hub):
    with pytest.raises(rv.RecoveryError, match="hf_revisiones"):
        run(hub, version="v2", sha=SHA_V2)


def reference_frame(rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [f"r{i}" for i in range(rows)],
            "score_attack": [0.25] * rows,
            "y_pred": [0] * rows,
        }
    )


def test_score_difference_at_the_tolerance_passes():
    reference = reference_frame()
    recovered = reference.copy()
    recovered.loc[1, "score_attack"] = 0.25 + 1e-10

    result = rv.compare_predictions(recovered, reference)
    assert result["resultado"] == "OK"
    assert result["diferencia_maxima_score"] == pytest.approx(1e-10)


def test_score_difference_above_the_tolerance_fails():
    reference = reference_frame()
    recovered = reference.copy()
    recovered.loc[1, "score_attack"] = 0.25 + 1e-8

    assert rv.compare_predictions(recovered, reference)["resultado"] == "FALLA"


def test_any_different_decision_fails():
    reference = reference_frame()
    recovered = reference.copy()
    recovered.loc[2, "y_pred"] = 1

    result = rv.compare_predictions(recovered, reference)
    assert result["resultado"] == "FALLA"
    assert result["coincidencias_y_pred"] == 2


def test_rows_in_another_order_fail():
    reference = reference_frame()
    recovered = reference.iloc[::-1].reset_index(drop=True)

    result = rv.compare_predictions(recovered, reference)
    assert result["resultado"] == "FALLA"
    assert result["coincidencias_row_id"] == 1


def test_another_row_count_fails():
    result = rv.compare_predictions(reference_frame(2), reference_frame(3))

    assert result["resultado"] == "FALLA"
    assert result["diferencia_maxima_score"] is None


def test_failed_comparison_still_writes_evidence_and_exits_with_error(hub, monkeypatch):
    frame = pd.read_csv(hub.reference)
    frame["y_pred"] = 1 - frame["y_pred"]
    frame.to_csv(hub.reference, index=False)
    monkeypatch.setattr(rv, "hf_hub_download", hub.download)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recover",
            "--version",
            "v1",
            "--sha",
            SHA_V1,
            "--inputs",
            str(hub.inputs),
            "--reference",
            str(hub.reference),
            "--out",
            str(hub.out),
        ],
    )
    # El valor por defecto de `download` se fija al definir la función.
    monkeypatch.setattr(rv.recover, "__defaults__", (None, hub.download))

    assert rv.main() == 1
    written = json.loads((hub.out / "comparacion_v1.json").read_text(encoding="utf-8"))
    assert written["resultado"] == "FALLA"
    assert (hub.out / "predicciones_recuperadas_v1.csv").exists()


def test_recovery_modules_do_not_need_mlflow():
    code = (
        "import sys, dsrv.recover, dsrv.verify_data, dsrv.recovery_report;"
        "sys.exit(1 if 'mlflow' in sys.modules else 0)"
    )

    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


@pytest.fixture
def data_tree(tmp_path):
    raw = tmp_path / "train.csv"
    raw.write_text("a\n1\n", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "train.parquet").write_bytes(b"datos")
    manifest = {"archivos_sha256": {"train.parquet": sha256_file(prepared / "train.parquet")}}
    (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    audit = {"archivos": {"training": {"ruta": str(raw), "sha256_esperado": sha256_file(raw)}}}
    (tmp_path / "auditoria.json").write_text(json.dumps(audit), encoding="utf-8")
    return tmp_path


def test_verify_data_accepts_matching_files(data_tree):
    result = vd.verify(data_tree / "auditoria.json", data_tree / "prepared")

    assert result["resultado"] == "OK"
    assert result["csv_originales_contra_auditoria"][0]["coincide"] is True
    assert result["datos_preparados_contra_manifest"][0]["coincide"] is True


def test_verify_data_detects_changed_raw_file(data_tree):
    (data_tree / "train.csv").write_text("a\n2\n", encoding="utf-8")

    assert vd.verify(data_tree / "auditoria.json", data_tree / "prepared")["resultado"] == "FALLA"


def test_verify_data_detects_changed_prepared_file(data_tree):
    (data_tree / "prepared" / "train.parquet").write_bytes(b"otros")

    assert vd.verify(data_tree / "auditoria.json", data_tree / "prepared")["resultado"] == "FALLA"


def test_verify_data_detects_missing_file(data_tree):
    (data_tree / "train.csv").unlink()

    result = vd.verify(data_tree / "auditoria.json", data_tree / "prepared")
    assert result["resultado"] == "FALLA"
    assert result["csv_originales_contra_auditoria"][0]["calculado"] is None


def report_inputs(**changes):
    summary = {
        "estado": "OK",
        "repo_url": "https://github.com/x/y.git",
        "commit": "c" * 40,
        "hf_repo": "repo/prueba",
        "inicio": "2026-11-04T10:00:00-05:00",
        "fin": "2026-11-04T10:05:00-05:00",
    }
    comparison = {
        "version": "v1",
        "hf_sha": SHA_V1,
        "filas": 40,
        "coincidencias_row_id": 40,
        "coincidencias_y_pred": 40,
        "diferencia_maxima_score": 0.0,
        "tolerancia_score": 1e-9,
        "umbral": 0.5,
        "sha256_modelo": "s" * 64,
        "resultado": "OK",
        **changes,
    }
    data = {
        "resultado": "OK",
        "manifest_sha256": "m" * 64,
        "csv_originales_contra_auditoria": [
            {"archivo": "data/raw/a.csv", "calculado": "h" * 64, "coincide": True}
        ],
        "datos_preparados_contra_manifest": [],
    }
    times = [{"paso": "clon", "segundos": "3", "estado": "ok"}]
    return summary, [comparison], data, times


def test_report_states_success_and_shows_the_table():
    text = report.build_report(*report_inputs())

    assert "**Resultado: SUPERADA.**" in text
    assert f"`{SHA_V1}`" in text
    assert "| 40/40 | 40/40 |" in text
    assert "Total: 3 s." in text


def test_report_states_failure_if_any_comparison_failed():
    text = report.build_report(*report_inputs(resultado="FALLA", coincidencias_y_pred=39))

    assert "**Resultado: NO SUPERADA.**" in text
    assert "39/40" in text


def test_report_states_failure_if_data_failed():
    summary, comparisons, data, times = report_inputs()
    data["resultado"] = "FALLA"

    assert "NO SUPERADA" in report.build_report(summary, comparisons, data, times)


def test_script_has_valid_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0


def test_script_never_traces_commands_or_prints_the_token():
    text = SCRIPT.read_text(encoding="utf-8")

    assert not re.search(r"^\s*set\s+-\w*x", text, flags=re.MULTILINE)
    assert not re.search(r"echo[^\n]*\$\{?DAGSHUB_TOKEN", text)
    assert not re.search(r"log_run[^\n]*DAGSHUB_TOKEN", text)
    assert "--exclude-editable" in text


def test_script_requires_a_token_before_creating_anything(tmp_path):
    workdir = tmp_path / "trabajo"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--commit",
            "c" * 40,
            "--sha-v1",
            SHA_V1,
            "--sha-v2",
            SHA_V2,
            "--workdir",
            str(workdir),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )

    assert result.returncode == 2
    assert "DAGSHUB_TOKEN" in result.stderr
    assert not workdir.exists()
