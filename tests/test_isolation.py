import io
import os
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts.check_isolation import check
from scripts.prepare_data import fetch_annotations
import run_experiment

ROOT = Path(__file__).resolve().parents[1]


def test_actual_runtime_is_inside_checkout():
    assert check()["status"] == "isolated"


def test_checker_rejects_external_cache(monkeypatch):
    monkeypatch.setenv("TORCH_HOME", "/outside-checkout-test")
    with pytest.raises(RuntimeError, match="Outside checkout"):
        check()


def test_wrapper_overrides_inherited_paths_without_changing_parent(tmp_path):
    fake_bin = tmp_path / "fake bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PWD" "${VIRTUAL_ENV-unset}"\n')
    fake_uv.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
               UV_CACHE_DIR="/outside-uv-cache", UV_PYTHON_INSTALL_DIR="/outside-python", VIRTUAL_ENV="/other-venv")
    before = dict(os.environ)
    result = subprocess.run([str(ROOT / "uv.sh"), "--version"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(ROOT / ".local/uv-cache"), str(ROOT / ".local/python"), str(ROOT), "unset"]
    assert dict(os.environ) == before


def zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipped:
        zipped.writestr("annotations/example.json", "{}")
    return buffer.getvalue()


@pytest.mark.parametrize("status", [200, 206])
def test_annotation_download_resumes_or_restarts_without_curl(tmp_path, monkeypatch, status):
    payload = zip_bytes()
    partial = tmp_path / "annotations_trainval2017.zip.part"
    partial.write_bytes(payload[:20])
    response = io.BytesIO(payload[20:] if status == 206 else payload)
    response.status = status
    response.headers = {"Content-Range": f"bytes 20-{len(payload)-1}/{len(payload)}"}
    def open_request(request, timeout):
        assert request.get_header("Range") == "bytes=20-"
        return response
    monkeypatch.setattr("scripts.prepare_data.urllib.request.urlopen", open_request)
    archive = fetch_annotations(tmp_path)
    assert archive.read_bytes() == payload
    assert not partial.exists()


def test_teacher_download_explicitly_uses_checkout(monkeypatch):
    calls = []
    monkeypatch.setattr(run_experiment.torch.hub, "load_state_dict_from_url", lambda *a, **kw: calls.append(kw))
    monkeypatch.setattr(run_experiment, "sha256_file", lambda path: "b5f2ef4d" + "0" * 56)
    monkeypatch.setenv("TORCH_HOME", "/outside-checkout-test")
    path = run_experiment.fetch_teacher()
    assert Path(calls[0]["model_dir"]) == ROOT / "checkpoints"
    assert path.parent == ROOT / "checkpoints"
