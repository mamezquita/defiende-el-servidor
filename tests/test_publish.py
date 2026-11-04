import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dsrv import publish_hf
from dsrv.hashing import sha256_file
from dsrv.package_model import STAGING_FILES, build_config, pinned_requirements
from dsrv.publish_hf import PublishError, check_staging, publish

SHA_V1 = "a" * 40
SHA_V2 = "b" * 40


class FakeApi:
    """Simula Hugging Face: guarda qué se subió y etiquetó, sin red."""

    def __init__(self, existing_tags=(), files=None):
        self.tags = list(existing_tags)
        self.uploads = []
        self.files = files
        self.staged = None

    def list_repo_refs(self, repo):
        return SimpleNamespace(tags=[SimpleNamespace(name=name) for name in self.tags])

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        self.staged = Path(kwargs["folder_path"])
        sha = SHA_V1 if len(self.uploads) == 1 else SHA_V2
        return SimpleNamespace(oid=sha, commit_url=f"https://hf.test/commit/{sha}")

    def create_tag(self, repo, tag, revision, tag_message=None):
        self.tags.append(tag)
        self.tagged = (tag, revision)

    def list_repo_files(self, repo, revision=None):
        return self.files or [*STAGING_FILES, ".gitattributes"]


def fake_download(api, content=None):
    def download(repo, filename, revision=None, local_dir=None):
        target = Path(local_dir) / filename
        if content is None:
            shutil.copy(api.staged / filename, target)
        else:
            target.write_bytes(content)
        return str(target)

    return download


def make_staging(root: Path, version: str) -> Path:
    staging = root / "hf_staging" / version
    staging.mkdir(parents=True)
    (staging / "model.joblib").write_bytes(f"modelo {version}".encode())
    digest = sha256_file(staging / "model.joblib")
    (staging / "model.joblib.sha256").write_text(f"{digest}  model.joblib\n", encoding="utf-8")
    (staging / "requirements.txt").write_text("joblib==1.6.0\n", encoding="utf-8")
    (staging / "README.md").write_text("# card\n", encoding="utf-8")
    config = {
        "config_id": f"config_{version}",
        "threshold": 0.5,
        "model_sha256": digest,
        "git_commit": "c" * 40,
        "git_tag": version,
        "git_tag_commit": "d" * 40,
        "data_version": "data-v2",
        "mlflow_run_id": "r" * 32,
    }
    (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return staging


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(publish_hf, "STAGING_DIR", tmp_path / "hf_staging")
    monkeypatch.setattr(publish_hf, "REVISIONS_JSON", tmp_path / "hf_revisiones.json")
    monkeypatch.setattr(publish_hf, "SELECTION_JSON", tmp_path / "seleccion.json")
    selection = {
        "estado": "borrador",
        "versiones": [
            {"version": version, "hf_repo": "", "hf_sha": "", "model_sha256": ""}
            for version in ("v1", "v2")
        ],
    }
    (tmp_path / "seleccion.json").write_text(json.dumps(selection), encoding="utf-8")
    make_staging(tmp_path, "v1")
    make_staging(tmp_path, "v2")
    return tmp_path


def test_publish_uploads_only_the_staging_and_replaces_previous_files(workspace):
    api = FakeApi()
    sha = publish("v1", api=api, download=fake_download(api))

    upload = api.uploads[0]
    assert sha == SHA_V1
    assert upload["folder_path"] == str(workspace / "hf_staging" / "v1")
    assert upload["delete_patterns"] == "*"
    assert api.tagged == ("v1", SHA_V1)


def test_publish_records_evidence_and_completes_selection(workspace):
    api = FakeApi()
    publish("v1", api=api, download=fake_download(api))

    selection = json.loads((workspace / "seleccion.json").read_text(encoding="utf-8"))
    assert selection["estado"] == "borrador"
    assert selection["versiones"][0]["hf_sha"] == SHA_V1
    assert selection["versiones"][1]["hf_sha"] == ""

    publish("v2", api=api, download=fake_download(api))

    selection = json.loads((workspace / "seleccion.json").read_text(encoding="utf-8"))
    revisions = json.loads((workspace / "hf_revisiones.json").read_text(encoding="utf-8"))
    assert selection["estado"] == "publicada"
    assert selection["versiones"][1]["hf_sha"] == SHA_V2
    assert selection["versiones"][1]["model_sha256"] == revisions["versiones"]["v2"]["model_sha256"]
    assert revisions["versiones"]["v1"]["hf_sha"] == SHA_V1
    assert revisions["versiones"]["v1"]["descarga_verificada"] is True


def test_v2_requires_v1_first(workspace):
    api = FakeApi()

    with pytest.raises(PublishError, match="v1"):
        publish("v2", api=api, download=fake_download(api))
    assert api.uploads == []


def test_existing_tag_blocks_upload(workspace):
    api = FakeApi(existing_tags=["v1"])

    with pytest.raises(PublishError, match="ya existe"):
        publish("v1", api=api, download=fake_download(api))
    assert api.uploads == []


def test_downloaded_hash_mismatch_fails_and_records_nothing(workspace):
    api = FakeApi()

    with pytest.raises(PublishError, match="SHA-256"):
        publish("v1", api=api, download=fake_download(api, content=b"otro modelo"))
    assert not (workspace / "hf_revisiones.json").exists()


def test_revision_with_unexpected_files_fails(workspace):
    api = FakeApi(files=[*STAGING_FILES, ".gitattributes", "LICENSE"])

    with pytest.raises(PublishError, match="contiene"):
        publish("v1", api=api, download=fake_download(api))


def test_model_not_matching_config_blocks_upload(workspace):
    (workspace / "hf_staging" / "v1" / "model.joblib").write_bytes(b"alterado")
    api = FakeApi()

    with pytest.raises(PublishError, match="model.joblib"):
        publish("v1", api=api, download=fake_download(api))
    assert api.uploads == []


def test_staging_with_extra_file_is_rejected(tmp_path):
    staging = make_staging(tmp_path, "v1")
    (staging / ".dvc").mkdir()
    (staging / ".dvc" / "config.local").write_text("x", encoding="utf-8")

    with pytest.raises(PublishError, match="sobran"):
        check_staging(staging)


def test_staging_with_missing_file_is_rejected(tmp_path):
    staging = make_staging(tmp_path, "v1")
    (staging / "README.md").unlink()

    with pytest.raises(PublishError, match="faltan"):
        check_staging(staging)


@pytest.mark.parametrize(
    "content",
    [
        "token hf_" + "a" * 30,
        "ghp_" + "a" * 30,
        "AKIA" + "A" * 16,
        "access_key_id = x",
        "/home/usuario/repo",
        "C:\\Users\\usuario",
    ],
)
def test_staging_with_secret_or_personal_path_is_rejected(tmp_path, content):
    staging = make_staging(tmp_path, "v1")
    (staging / "README.md").write_text(content, encoding="utf-8")

    with pytest.raises(PublishError, match="README.md"):
        check_staging(staging)


def test_pinned_requirements_take_exact_versions():
    text = "pandas==3.1.0\nnumpy==2.4.6\nscikit-learn==1.9.1\ndvc[s3]==3.67.1\njoblib==1.6.0\n"

    assert pinned_requirements(text) == (
        "scikit-learn==1.9.1\npandas==3.1.0\nnumpy==2.4.6\njoblib==1.6.0\n"
    )


def test_config_has_required_traceability_fields():
    document = {
        "version": "v2",
        "config_id": "e4_hgb",
        "umbral": 0.64,
        "git_commit": "c" * 40,
        "data_version": "data-v2",
        "run_id": "r" * 32,
        "metricas_validacion": {"fpr": 0.04},
    }
    config = build_config(document, ["dur"], "s" * 64, "d" * 40)

    assert config["threshold"] == 0.64
    assert config["positive_class"]["label"] == 1
    assert config["github_commit_url"].endswith(f"/commit/{'c' * 40}")
    assert config["git_tag_commit"] == "d" * 40
    assert {"label", "attack_cat", "id"} <= set(config["excluded_columns"])
    assert yaml.safe_dump(config)
