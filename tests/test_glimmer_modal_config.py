"""Glimmer Modal config, lock download keys and synthetic image: no network, no weights."""

from __future__ import annotations

import json
import re
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.glimmer_modal_config import load_config, load_download_spec, synthetic_png
from tools.glimmer_probe import DEFAULT_PIN, ProofError

CONFIG = ROOT / "config" / "glimmer_modal.json"


def _edit(tmp_path: Path, src: Path, **changes: object) -> Path:
    data = json.loads(src.read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    out = tmp_path / src.name
    out.write_text(json.dumps(data))
    return out


def test_shipped_config_and_lock_load():
    cfg = load_config()
    assert cfg.cpu > 0 and cfg.memory_mb >= 32768 and "gpu" not in CONFIG.read_text()
    spec = load_download_spec()
    assert [name for name, _, _ in spec.files] == [
        "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf",
        "mmproj-Muse-Glimmer-30B-Q4_K_M.gguf",
    ]
    assert len(spec.revision) == 40 and spec.runtime_version == "0.3.36"


@pytest.mark.parametrize("key", ["cpu", "memory_mb", "volume_name", "timeout_s"])
def test_config_missing_key_fails(tmp_path, key):
    with pytest.raises(ProofError, match=key):
        load_config(_edit(tmp_path, CONFIG, **{key: None}))


def test_config_rejects_typo_wrong_type_and_empty(tmp_path):
    with pytest.raises(ProofError, match="unknown"):
        load_config(_edit(tmp_path, CONFIG, cpuu=8))
    with pytest.raises(ProofError, match="cpu"):
        load_config(_edit(tmp_path, CONFIG, cpu="8"))
    with pytest.raises(ProofError, match="volume_name"):
        load_config(_edit(tmp_path, CONFIG, volume_name=" "))
    with pytest.raises(ProofError, match="apt_packages"):
        load_config(_edit(tmp_path, CONFIG, apt_packages=[]))
    with pytest.raises(ProofError, match="cannot read"):
        load_config(tmp_path / "missing.json")


def test_lock_placeholder_or_missing_download_key_fails(tmp_path):
    with pytest.raises(ProofError, match="revision"):
        load_download_spec(_edit(tmp_path, DEFAULT_PIN, revision="UNVERIFIED"))
    with pytest.raises(ProofError, match="weights_file"):
        load_download_spec(_edit(tmp_path, DEFAULT_PIN, weights_file=None))
    with pytest.raises(ProofError, match="runtime_sdist_sha256"):
        load_download_spec(_edit(tmp_path, DEFAULT_PIN, runtime_sdist_sha256="abc"))


def test_synthetic_png_is_a_valid_png():
    png = synthetic_png(32)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    pos, idat = 8, b""
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, body = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        assert struct.unpack(">I", png[pos + 8 + length : pos + 12 + length])[0] == zlib.crc32(kind + body)
        idat += body if kind == b"IDAT" else b""
        pos += 12 + length
    assert len(zlib.decompress(idat)) == 32 * (1 + 32 * 3)
    with pytest.raises(ProofError):
        synthetic_png(4)


def test_runtime_deps_and_max_tokens_are_pinned_config(tmp_path):
    cfg = load_config()
    assert {d.split("==")[0] for d in cfg.runtime_deps} >= {"numpy", "typing-extensions", "diskcache", "jinja2"}
    assert cfg.grade_max_tokens >= 512
    for key in ("runtime_deps", "grade_max_tokens"):
        with pytest.raises(ProofError, match=key):
            load_config(_edit(tmp_path, CONFIG, **{key: None}))
    for bad in (["numpy"], ["numpy>=2"], ["numpy==latest"], []):
        with pytest.raises(ProofError, match="runtime_deps"):
            load_config(_edit(tmp_path, CONFIG, runtime_deps=bad))
    with pytest.raises(ProofError, match="grade_max_tokens"):
        load_config(_edit(tmp_path, CONFIG, grade_max_tokens=0))


def test_timeouts_are_config_and_function_timeout_must_cover_them(tmp_path):
    cfg = load_config()
    assert cfg.download_timeout_s > 0 and cfg.proof_timeout_s > 0
    assert cfg.timeout_s > cfg.download_timeout_s + 2 * cfg.proof_timeout_s
    for key in ("download_timeout_s", "proof_timeout_s"):
        with pytest.raises(ProofError, match=key):
            load_config(_edit(tmp_path, CONFIG, **{key: None}))
    with pytest.raises(ProofError, match="timeout_s must exceed"):
        load_config(_edit(tmp_path, CONFIG, proof_timeout_s=cfg.timeout_s))


def test_timeout_budget_boundary_counts_two_downloads_and_two_proofs(tmp_path):
    base = {"download_timeout_s": 120, "proof_timeout_s": 600}
    floor = 2 * 120 + 2 * 600
    with pytest.raises(ProofError, match="timeout_s must exceed"):
        load_config(_edit(tmp_path, CONFIG, timeout_s=floor, **base))  # equal is rejected
    assert load_config(_edit(tmp_path, CONFIG, timeout_s=floor + 1, **base)).timeout_s == floor + 1


@pytest.mark.parametrize("bad", ["Infinity", "NaN", "-1", "0"])
def test_config_rejects_non_finite_or_non_positive_timeouts(tmp_path, bad):
    src = CONFIG.read_text()
    for key in ("timeout_s", "download_timeout_s", "proof_timeout_s"):
        path = tmp_path / f"{key}-{bad}.json"
        path.write_text(re.sub(rf'("{key}": )\d+', rf"\g<1>{bad}", src))
        with pytest.raises(ProofError, match=key):
            load_config(path)


def _load_deploy(monkeypatch):
    import importlib.util
    import types

    class _Any:
        def __getattr__(self, name):
            return self

        def __call__(self, *a, **k):
            return a[0] if len(a) == 1 and callable(a[0]) and not k else self

    stub = types.ModuleType("modal")
    stub.is_local = lambda: False
    stub.Image = stub.App = stub.Volume = _Any()
    monkeypatch.setitem(sys.modules, "modal", stub)
    monkeypatch.setenv("PARTFIDELITY_ROOT", str(CONFIG.parents[1]))
    spec = importlib.util.spec_from_file_location("glimmer_deploy", CONFIG.parents[1] / "deploy" / "glimmer_modal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_pinned_gives_every_download_the_config_timeout(tmp_path, monkeypatch):
    import io

    mod = _load_deploy(monkeypatch)
    seen = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda url, **k: seen.append(k) or _Resp(b"x"))
    monkeypatch.setattr(mod, "_sha256", lambda path: "wrong")
    monkeypatch.setattr(mod.volume, "commit", lambda: None, raising=False)
    # the digest check fails after the first download, so drive each pinned file in turn
    for name, size, want in mod.spec.files:
        monkeypatch.setattr(mod, "spec", mod.spec.__class__(**{**mod.spec.__dict__, "files": ((name, size, want),)}))
        with pytest.raises(SystemExit):
            mod.fetch_pinned(tmp_path / name)
    assert len(seen) == 2 and all(k == {"timeout": mod.cfg.download_timeout_s} for k in seen)


def test_deploy_passes_config_timeouts_and_uses_a_fresh_dir_per_call(tmp_path, monkeypatch):
    import subprocess
    import tempfile
    import types

    mod = _load_deploy(monkeypatch)
    root = tmp_path / "tmproot"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    calls = []
    monkeypatch.setattr(mod, "fetch_pinned", lambda dest: {n: tmp_path / n for n, _, _ in mod.spec.files})
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: calls.append(cmd) or types.SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )
    mod.glimmer_proof()
    mod.glimmer_proof()
    for cmd in calls:
        assert cmd[cmd.index("--timeout-s") + 1] == str(mod.cfg.proof_timeout_s)
    rows = [c[c.index("--out") + 1] for c in calls if "--out" in c]
    assert len(rows) == 2 and len(set(rows)) == 2  # a fresh dir per call
    assert list(root.iterdir()) == []  # and each is removed afterwards
