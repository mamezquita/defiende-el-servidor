"""Pruebas de la evaluación final con datos sintéticos: nunca leen el archivo de test real."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from dsrv import document_test_hf as doc
from dsrv import evaluate_test as ev
from dsrv import recompute_metrics as rc
from dsrv.config import load_yaml
from dsrv.features import build_pipeline
from dsrv.freeze_selection import freeze, incomplete_fields
from dsrv.gitinfo import DirtyWorktreeError
from dsrv.hashing import sha256_dataframe_rows, sha256_file
from dsrv.package_model import STAGING_FILES
from dsrv.predict import predict_frame
from dsrv.publish_hf import PublishError

FEATURES = ["dur", "proto", "service", "state"]
SHA_EVALUATED = "e" * 40
SHA_DOC = "f" * 40


def make_frame(rows: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, 2, rows)
    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "dur": np.round(rng.random(rows) + label, 6),
            "proto": rng.choice(["tcp", "udp"], rows),
            "service": rng.choice(["-", "dns"], rows),
            "state": rng.choice(["FIN", "CON"], rows),
            "attack_cat": np.where(label == 1, "Generic", "Normal"),
            "label": label,
        }
    )


def make_config(threshold: float = 0.5, sha: str = "") -> dict:
    return {
        "threshold": threshold,
        "decision_rule": "y_pred = 1 si score_attack >= threshold",
        "positive_class": {"label": 1, "name": "ataque"},
        "input_columns": FEATURES,
        "sklearn_version": sklearn.__version__,
        "model_sha256": sha,
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Árbol temporal: el test es sintético y las rutas reales quedan fuera de alcance."""
    test_csv = tmp_path / "test.csv"
    frame = make_frame()
    frame.to_csv(test_csv, index=False)

    train = make_frame(seed=1)
    pipeline = build_pipeline(load_yaml("configs/e4_hgb.yaml"), train[FEATURES])
    pipeline.fit(train[FEATURES], train["label"])
    model_dir = tmp_path / "hub"
    model_dir.mkdir()
    joblib.dump(pipeline, model_dir / "model.joblib")
    model_sha = sha256_file(model_dir / "model.joblib")
    (model_dir / "config.json").write_text(
        json.dumps(make_config(sha=model_sha)), encoding="utf-8"
    )

    # Las 10 primeras filas de test están en train, las 5 siguientes en val.
    hashes = sha256_dataframe_rows(frame, FEATURES)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    pd.DataFrame(
        {
            "row_id": [f"train:{i}" for i in range(16)],
            "group_id": [*hashes[:10], *hashes[10:15], "otro"],
            "split": ["train"] * 10 + ["val"] * 5 + ["excluded"],
            "motivo_exclusion": [""] * 15 + ["grupo_contradictorio"],
        }
    ).to_csv(prepared / "splits.csv", index=False)

    source = tmp_path / "source.yaml"
    source.write_text(
        "archivos:\n  testing:\n"
        f"    ruta: {test_csv}\n    sha256: {sha256_file(test_csv)}\n    filas: {len(frame)}\n",
        encoding="utf-8",
    )
    selection = tmp_path / "seleccion.json"
    document = {
        "version": "v2",
        "hf_repo": "repo/prueba",
        "hf_sha": SHA_EVALUATED,
        "model_sha256": model_sha,
        "umbral": 0.5,
    }
    selection.write_text(json.dumps({"versiones": [document]}), encoding="utf-8")

    def fake_download(repo, name, revision=None, local_dir=None):
        assert revision == SHA_EVALUATED
        target = Path(local_dir) / name
        shutil.copy(model_dir / name, target)
        return str(target)

    monkeypatch.setattr(ev, "hf_hub_download", fake_download)
    monkeypatch.setattr(ev, "SOURCE_CONFIG", source)
    monkeypatch.setattr(ev, "PREPARED_DIR", prepared)
    monkeypatch.setattr(ev, "SELECTION_JSON", selection)
    monkeypatch.setattr(ev, "METRICS_JSON", tmp_path / "metricas-test.json")
    monkeypatch.setattr(ev, "PREDICTIONS_CSV", tmp_path / "predicciones-test.csv")
    monkeypatch.setattr(ev, "assert_clean_worktree", lambda: None)
    monkeypatch.setattr(ev, "get_git_commit", lambda: "c" * 40)
    monkeypatch.setattr(ev, "_git", lambda *args: "")
    return SimpleNamespace(
        root=tmp_path, frame=frame, document=document, model_dir=model_dir, test_csv=test_csv
    )


def run_evaluation(workspace):
    ev.main()
    report = json.loads((workspace.root / "metricas-test.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(workspace.root / "predicciones-test.csv")
    return report, predictions


def test_writes_every_row_without_filtering(workspace):
    # Duplicar filas no debe cambiar el total: el test no se deduplica.
    frame = pd.concat([workspace.frame, workspace.frame.head(7)], ignore_index=True)
    frame.to_csv(workspace.test_csv, index=False)
    source = workspace.root / "source.yaml"
    source.write_text(
        "archivos:\n  testing:\n"
        f"    ruta: {workspace.test_csv}\n    sha256: {sha256_file(workspace.test_csv)}\n"
        f"    filas: {len(frame)}\n",
        encoding="utf-8",
    )
    report, predictions = run_evaluation(workspace)

    assert report["filas_evaluadas"] == len(frame) == len(predictions) == 107
    assert list(predictions.columns) == ["row_id", "id", "y_true", "score_attack", "y_pred"]
    assert predictions["row_id"].tolist() == [f"test:{i}" for i in range(107)]
    assert predictions["id"].tolist() == frame["id"].tolist()
    assert predictions["y_true"].tolist() == frame["label"].tolist()


def test_report_has_the_required_fields(workspace):
    report, predictions = run_evaluation(workspace)

    assert report["version"] == "v2"
    assert report["hf_sha"] == SHA_EVALUATED
    assert report["model_sha256"] == workspace.document["model_sha256"]
    assert report["dataset_sha256"] == sha256_file(workspace.test_csv)
    assert report["umbral"] == 0.5
    assert report["clase_positiva"]["label"] == 1
    assert report["git_commit"] == "c" * 40
    assert "regla_decision" in report and "fecha" in report
    (tn, fp), (fn, tp) = report["matriz_confusion"]
    assert tn + fp + fn + tp == 100
    assert report["recall_attack"] == tp / (tp + fn)
    assert report["fpr"] == fp / (fp + tn)
    assert (predictions["y_pred"] == (predictions["score_attack"] >= 0.5).astype(int)).all()


def test_seen_and_unseen_rows_are_only_descriptive(workspace):
    report, _ = run_evaluation(workspace)
    groups = report["descriptivo_filas_vistas"]

    assert "nota" in groups
    assert groups["vistas_en_train"]["filas"] == 10
    assert groups["vistas_en_val"]["filas"] == 5
    assert groups["vistas_en_train_o_val"]["filas"] == 15
    assert groups["no_vistas"]["filas"] == 85
    # Las métricas globales son las del test completo, no las de un grupo.
    (tn, fp), (fn, tp) = report["matriz_confusion"]
    assert tn + fp + fn + tp == 100


def test_aborts_if_metrics_already_exist(workspace):
    (workspace.root / "metricas-test.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ev.EvaluationAborted, match="ya existe"):
        ev.main()
    assert not (workspace.root / "predicciones-test.csv").exists()


def test_aborts_if_predictions_already_exist(workspace):
    (workspace.root / "predicciones-test.csv").write_text("x", encoding="utf-8")

    with pytest.raises(ev.EvaluationAborted, match="ya existe"):
        ev.main()


def test_second_run_is_rejected_and_changes_nothing(workspace):
    run_evaluation(workspace)
    before = (workspace.root / "metricas-test.json").read_bytes()

    with pytest.raises(ev.EvaluationAborted, match="ya existe"):
        ev.main()
    assert (workspace.root / "metricas-test.json").read_bytes() == before


def test_aborts_on_dirty_worktree(workspace, monkeypatch):
    def dirty():
        raise DirtyWorktreeError("cambios")

    monkeypatch.setattr(ev, "assert_clean_worktree", dirty)

    with pytest.raises(DirtyWorktreeError):
        ev.main()
    assert not (workspace.root / "metricas-test.json").exists()


def test_aborts_if_freeze_is_not_ancestor(workspace, monkeypatch):
    def not_ancestor(*args):
        if args[0] == "merge-base":
            raise subprocess.CalledProcessError(1, "git")
        return ""

    monkeypatch.setattr(ev, "_git", not_ancestor)

    with pytest.raises(ev.EvaluationAborted, match="ancestro"):
        ev.main()


def test_aborts_if_selection_changed_since_freeze(workspace, monkeypatch):
    monkeypatch.setattr(ev, "_git", lambda *args: "diff" if args[0] == "diff" else "")

    with pytest.raises(ev.EvaluationAborted, match="cambió"):
        ev.main()


def test_aborts_if_test_file_hash_differs(workspace):
    with workspace.test_csv.open("a", encoding="utf-8") as file:
        file.write("999,0.1,tcp,-,FIN,Normal,0\n")

    with pytest.raises(ev.EvaluationAborted, match="SHA-256 del test"):
        ev.main()
    assert not (workspace.root / "metricas-test.json").exists()


def test_aborts_if_downloaded_model_hash_differs(workspace):
    (workspace.model_dir / "model.joblib").write_bytes(b"otro")

    with pytest.raises(ev.EvaluationAborted, match="SHA-256 del modelo"):
        ev.main()


def test_aborts_if_config_threshold_differs_from_frozen(workspace):
    selection = workspace.root / "seleccion.json"
    selection.write_text(
        json.dumps({"versiones": [{**workspace.document, "umbral": 0.9}]}), encoding="utf-8"
    )

    with pytest.raises(ev.EvaluationAborted, match="umbral"):
        ev.main()


def test_recompute_accepts_the_written_report(workspace):
    report, predictions = run_evaluation(workspace)
    masks = ev.seen_masks(ev.read_raw_csv(workspace.test_csv), FEATURES)

    assert rc.find_differences(predictions, report, masks) == []


@pytest.mark.parametrize(
    "tamper, expected",
    [
        (lambda report, preds: report.update(recall_attack=0.1), "recall_attack"),
        (lambda report, preds: report.update(filas_evaluadas=5), "filas_evaluadas"),
        (lambda report, preds: report["matriz_confusion"][0].__setitem__(0, -1), "matriz"),
        (lambda report, preds: preds.__setitem__("y_pred", 1 - preds["y_pred"]), "y_pred"),
        (lambda report, preds: preds.__setitem__("row_id", "x"), "row_id"),
        (lambda report, preds: report.update(umbral=1.01), "y_pred"),
    ],
)
def test_recompute_detects_tampering(workspace, tamper, expected):
    report, predictions = run_evaluation(workspace)
    tamper(report, predictions)

    assert any(expected in item for item in rc.find_differences(predictions, report))


def test_recompute_detects_changed_descriptive_section(workspace):
    report, predictions = run_evaluation(workspace)
    masks = ev.seen_masks(ev.read_raw_csv(workspace.test_csv), FEATURES)
    report["descriptivo_filas_vistas"]["no_vistas"]["filas"] = 1

    assert any("descriptivo" in item for item in rc.find_differences(predictions, report, masks))


def selection_document(**changes):
    return {
        "version": "v2",
        "config_id": "e4_hgb",
        "run_id": "r" * 32,
        "git_commit": "c" * 40,
        "data_version": "data-v2|x=1",
        "umbral": 0.64,
        "metricas_validacion": {"fpr": 0.04},
        "razon_seleccion": "razón",
        "hf_repo": "repo/prueba",
        "hf_sha": "e" * 40,
        "model_sha256": "s" * 64,
        **changes,
    }


REVISIONS = {
    "versiones": {
        "v2": {
            "hf_sha": "e" * 40,
            "model_sha256": "s" * 64,
            "git_tag_commit": "d" * 40,
            "hf_tag": "v2",
        }
    }
}


def test_freeze_adds_references_and_state():
    frozen = freeze({"versiones": [selection_document()]}, REVISIONS)

    assert frozen["estado"] == "congelada"
    document = frozen["versiones"][0]
    assert document["git_tag_commit"] == "d" * 40
    assert document["dvc_tag"] == "data-v2"
    assert document["hf_tag"] == "v2"


@pytest.mark.parametrize(
    "changes, field",
    [
        ({"hf_sha": ""}, "hf_sha"),
        ({"model_sha256": ""}, "model_sha256"),
        ({"razon_seleccion": ""}, "razon_seleccion"),
        ({"git_commit": "abc"}, "git_commit"),
    ],
)
def test_incomplete_selection_is_detected(changes, field):
    assert field in incomplete_fields(selection_document(**changes))


def test_freeze_rejects_selection_that_differs_from_revisions():
    with pytest.raises(ValueError, match="hf_sha"):
        freeze({"versiones": [selection_document(hf_sha="a" * 40)]}, REVISIONS)


CARD = f"""# Card

## Métricas

{doc.PENDING_NOTICE}

## Umbral

Texto.
"""


def report_for_card():
    group = {
        "filas": 10,
        "recall_attack": 0.9,
        "precision_attack": 0.8,
        "fpr": 0.1,
        "macro_f1": 0.85,
    }
    return {
        "hf_sha": SHA_EVALUATED,
        "model_sha256": "s" * 64,
        "dataset_sha256": "t" * 64,
        "filas_evaluadas": 82332,
        "umbral": 0.64,
        "freeze_tag": "freeze-v2",
        "git_commit": "c" * 40,
        "recall_attack": 0.9,
        "precision_attack": 0.8,
        "fpr": 0.06,
        "macro_f1": 0.85,
        "accuracy": 0.86,
        "matriz_confusion": [[1, 2], [3, 4]],
        "descriptivo_filas_vistas": {"vistas_en_train_o_val": group, "no_vistas": group},
    }


VALIDATION = {
    "recall_attack": 0.9035,
    "precision_attack": 0.9473,
    "fpr": 0.0487,
    "macro_f1": 0.9277,
    "accuracy": 0.9278,
}


def test_card_section_cites_evaluated_revision_and_model():
    section = doc.build_section(report_for_card(), VALIDATION)

    assert SHA_EVALUATED in section
    assert "s" * 64 in section
    assert "82.332" in section


def test_update_card_replaces_notice_and_keeps_the_rest():
    updated = doc.update_card(CARD, doc.build_section(report_for_card(), VALIDATION))

    assert doc.PENDING_NOTICE not in updated
    assert doc.PUBLISHED_NOTICE in updated
    assert updated.index(doc.SECTION_TITLE) < updated.index("## Umbral")
    assert "Texto." in updated


def test_update_card_rejects_unexpected_structure():
    with pytest.raises(PublishError):
        doc.update_card("# Sin secciones", "sección")


class FakeHub:
    def __init__(self, repo_dir, tag_sha=SHA_EVALUATED):
        self.repo_dir = repo_dir
        self.tag_sha = tag_sha
        self.uploads = []

    def upload_file(self, **kwargs):
        self.uploads.append(kwargs)
        return SimpleNamespace(oid=SHA_DOC, commit_url=f"https://hf.test/{SHA_DOC}")

    def list_repo_files(self, repo, revision=None):
        return [*STAGING_FILES, ".gitattributes"]

    def repo_info(self, repo, revision=None):
        return SimpleNamespace(sha=self.tag_sha)


@pytest.fixture
def doc_workspace(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    hub.mkdir()
    (hub / "model.joblib").write_bytes(b"modelo")
    (hub / "README.md").write_text(CARD, encoding="utf-8")
    (hub / "config.json").write_text(
        json.dumps({"validation_metrics": VALIDATION}), encoding="utf-8"
    )
    report = {**report_for_card(), "model_sha256": sha256_file(hub / "model.joblib")}
    metrics = tmp_path / "metricas-test.json"
    metrics.write_text(json.dumps(report), encoding="utf-8")
    revisions = tmp_path / "hf_revisiones.json"
    entry = {"hf_sha": SHA_EVALUATED, "model_sha256": report["model_sha256"]}
    revisions.write_text(json.dumps({"versiones": {"v2": entry}}), encoding="utf-8")
    monkeypatch.setattr(doc, "METRICS_JSON", metrics)
    monkeypatch.setattr(doc, "REVISIONS_JSON", revisions)

    def download(repo, name, revision=None, local_dir=None):
        target = Path(local_dir) / name
        shutil.copy(hub / name, target)
        return str(target)

    return SimpleNamespace(root=tmp_path, hub=hub, revisions=revisions, download=download)


def test_publish_uploads_only_the_readme_and_records_both_revisions(doc_workspace):
    api = FakeHub(doc_workspace.hub)
    documentation = doc.publish_test_section(api=api, download=doc_workspace.download)

    upload = api.uploads[0]
    assert documentation == SHA_DOC
    assert len(api.uploads) == 1
    assert upload["path_in_repo"] == "README.md"
    assert upload["parent_commit"] == SHA_EVALUATED
    entry = json.loads(doc_workspace.revisions.read_text(encoding="utf-8"))["versiones"]["v2"]
    assert entry["hf_sha"] == SHA_EVALUATED
    assert entry["revision_evaluada"] == SHA_EVALUATED
    assert entry["revision_documentacion"] == SHA_DOC


def test_model_changed_in_new_revision_fails_and_records_nothing(doc_workspace):
    (doc_workspace.hub / "model.joblib").write_bytes(b"otro")
    api = FakeHub(doc_workspace.hub)

    with pytest.raises(PublishError, match="SHA-256"):
        doc.publish_test_section(api=api, download=doc_workspace.download)
    entry = json.loads(doc_workspace.revisions.read_text(encoding="utf-8"))["versiones"]["v2"]
    assert "revision_documentacion" not in entry


def test_moved_tag_fails(doc_workspace):
    api = FakeHub(doc_workspace.hub, tag_sha="0" * 40)

    with pytest.raises(PublishError, match="tag v2"):
        doc.publish_test_section(api=api, download=doc_workspace.download)


def test_second_documentation_is_rejected(doc_workspace):
    api = FakeHub(doc_workspace.hub)
    doc.publish_test_section(api=api, download=doc_workspace.download)

    with pytest.raises(PublishError, match="ya está registrada"):
        doc.publish_test_section(api=api, download=doc_workspace.download)
    assert len(api.uploads) == 1
