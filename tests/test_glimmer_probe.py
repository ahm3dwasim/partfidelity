"""Glimmer proof harness tests. Fake loader only: no weights, no network."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.contracts import GlimmerProof
from tools import glimmer_probe
from tools.glimmer_probe import ProofError

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PROOF_FIELDS = {field.name for field in dataclasses.fields(GlimmerProof)}
WEIGHTS_SHA = "ab" * 32
MMPROJ_SHA = "cd" * 32
TEST_VM_ID = "ref-vm-01"


@pytest.fixture(autouse=True)
def _no_glimmer_env(monkeypatch):
    monkeypatch.delenv("GLIMMER_WEIGHTS", raising=False)
    monkeypatch.delenv("GLIMMER_MMPROJ", raising=False)
    monkeypatch.delenv("GLIMMER_VM_ID", raising=False)


def _pin_dict(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "model_id": "muse-glimmer",
        "revision": "0" * 40,
        "licence": "Apache-2.0",
        "licence_source": "https://dev.meta.ai/models/muse-glimmer/",
        "quantisation": "Q4_K_M",
        "runtime": "llama.cpp",
        "runtime_version": "0.3.16",
        "weights_sha256": WEIGHTS_SHA,
        "mmproj_sha256": MMPROJ_SHA,
    }
    data.update(overrides)
    return data


def _write_pin(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "glimmer_pin.json"
    path.write_text(json.dumps(_pin_dict(**overrides)), encoding="utf-8")
    return path


def _write_image(tmp_path: Path, name: str = "owned.png") -> Path:
    path = tmp_path / name
    path.write_bytes(PNG)
    return path


def _darwin_host(**overrides: object) -> dict[str, object]:
    host: dict[str, object] = {
        "system": "Darwin",
        "release": "24.6.0",
        "machine": "arm64",
        "processor": "arm",
        "cpu_count": 8,
        "gpu": "apple-silicon",
    }
    host.update(overrides)
    return host


def _linux_host(**overrides: object) -> dict[str, object]:
    host: dict[str, object] = {
        "system": "Linux",
        "release": "6.8.0",
        "machine": "x86_64",
        "processor": "x86_64",
        "cpu_count": 4,
        "gpu": "none",
    }
    host.update(overrides)
    return host


def _fake_measurements(
    monkeypatch,
    ram=(1_000_000, 2_000_000, 3_000_000),
    ticks=(10.0, 10.25),
) -> None:
    # The spawned worker measures its own process RAM; the parent reads
    # ram[0] then ram[1] from this iterator.
    rams = iter(ram)
    clock = iter(ticks)
    monkeypatch.setattr(glimmer_probe, "_peak_ram_bytes", lambda: next(rams))
    monkeypatch.setattr(glimmer_probe, "_wall_clock", lambda: next(clock))


def _probe(
    tmp_path,
    monkeypatch,
    host: dict[str, object],
    loader=None,
    vm_id: str | None = None,
    **pin_overrides: object,
):
    pin_path = _write_pin(tmp_path, **pin_overrides)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: dict(host))
    _fake_measurements(monkeypatch)
    envelope = glimmer_probe.probe_glimmer(
        image, pin_path=pin_path, loader=loader or FakeLoader(), vm_id=vm_id
    )
    return envelope, pin_path, image


def _redigest(envelope: dict) -> dict:
    """Recompute the plain content digest after fabrication (anyone can)."""
    body = {
        name: envelope["receipt"][name]
        for name in glimmer_probe.RECEIPT_FIELDS
        if name != "digest"
    }
    envelope["receipt"]["digest"] = glimmer_probe._digest(envelope["proof"], body)
    return envelope


def _decide(monkeypatch, rows, pin, image, loader=None, host=None, **kwargs):
    """Call verdict on a simulated machine with fresh fake measurements."""
    monkeypatch.setattr(
        glimmer_probe, "measure_host", lambda: dict(host or _darwin_host())
    )
    _fake_measurements(monkeypatch)
    return glimmer_probe.verdict(
        rows, pin=pin, image_path=image, loader=loader or FakeLoader(), **kwargs
    )


class FakeModel:
    def __init__(self, answer: str = "faithful") -> None:
        self.answer = answer
        self.calls: list[tuple[bytes, str]] = []

    def grade(self, image: bytes, prompt: str) -> str:
        self.calls.append((image, prompt))
        return self.answer


class FakeLoader:
    def __init__(
        self,
        model: FakeModel | None = None,
        artifact_digests: tuple[str, str] | None = (WEIGHTS_SHA, MMPROJ_SHA),
    ) -> None:
        self.model = model or FakeModel()
        self.artifact_digests = artifact_digests
        self.loaded = 0

    def load(self) -> FakeModel:
        self.loaded += 1
        return self.model


class FailingLoader:
    artifact_digests = (WEIGHTS_SHA, MMPROJ_SHA)

    def load(self) -> FakeModel:
        raise ProofError("weights did not load")


def _patched_loader(monkeypatch, loader):
    # The CLI path requires verified digests; the fake carries fake ones.
    loader.artifact_digests = ("ab" * 32, "cd" * 32)
    monkeypatch.setattr(glimmer_probe, "build_loader", lambda *a, **k: loader)
    return loader


class SleepingLoader:
    artifact_digests = (WEIGHTS_SHA, MMPROJ_SHA)

    def load(self) -> FakeModel:
        time.sleep(30)
        return FakeModel()


def test_proof_schema_requires_all_fields(tmp_path, monkeypatch, capsys):
    assert PROOF_FIELDS == {
        "model_id",
        "licence",
        "quantisation",
        "peak_ram_bytes",
        "latency_ms",
        "platform",
        "graded_image",
    }
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    _patched_loader(monkeypatch, FakeLoader())
    monkeypatch.setattr(
        glimmer_probe, "measure_host", lambda: _darwin_host()
    )
    _fake_measurements(monkeypatch)

    code = glimmer_probe.main(
        ["probe", "--image", str(image), "--pin", str(pin)]
    )

    captured = capsys.readouterr()
    assert code == 0
    envelope = json.loads(captured.out)
    assert set(envelope) == {"proof", "receipt"}
    row = envelope["proof"]
    assert set(row) == PROOF_FIELDS
    assert all(row[name] is not None for name in PROOF_FIELDS)
    assert set(envelope["receipt"]) == set(glimmer_probe.RECEIPT_FIELDS)
    loaded = glimmer_probe.load_pin(pin)
    proof, _receipt = glimmer_probe.validate_row(envelope, pin=loaded)
    assert proof == GlimmerProof(**row)
    for name in PROOF_FIELDS:
        bad = copy.deepcopy(envelope)
        del bad["proof"][name]
        with pytest.raises(ProofError):
            glimmer_probe.validate_row(bad, pin=loaded)
    bad = copy.deepcopy(envelope)
    bad["proof"]["extra"] = 1
    with pytest.raises(ProofError):
        glimmer_probe.validate_row(bad, pin=loaded)
    for name, value in (
        ("peak_ram_bytes", "123"),
        ("latency_ms", 1.5),
        ("graded_image", "true"),
        ("platform", "windows"),
    ):
        bad = copy.deepcopy(envelope)
        bad["proof"][name] = value
        with pytest.raises(ProofError):
            glimmer_probe.validate_row(bad, pin=loaded)
    # A bare seven-field row is hand-writable shape: it carries no receipt.
    with pytest.raises(ProofError, match="no receipt"):
        glimmer_probe.validate_row(row, pin=loaded)


def test_failed_load_emits_nonzero_and_no_claim(tmp_path, monkeypatch, capsys):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    out = tmp_path / "row.json"
    _patched_loader(monkeypatch, FailingLoader())
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)

    code = glimmer_probe.main(
        ["probe", "--image", str(image), "--pin", str(pin), "--out", str(out)]
    )

    captured = capsys.readouterr()
    assert code != 0
    assert not out.exists()
    assert captured.out == ""
    assert "proof failed" in captured.err
    assert "model_id" not in captured.err
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError):
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FailingLoader())


def test_missing_licence_fails_closed(tmp_path, capsys):
    image = _write_image(tmp_path)
    missing = {key: value for key, value in _pin_dict().items() if key != "licence"}
    cases = [
        missing,
        _pin_dict(licence="UNVERIFIED"),
        _pin_dict(licence=""),
        _pin_dict(licence="TODO fill"),
        _pin_dict(weights_sha256="UNVERIFIED"),
        _pin_dict(mmproj_sha256="not-a-hash"),
    ]
    for index, data in enumerate(cases):
        pin = tmp_path / f"pin-{index}.json"
        pin.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ProofError):
            glimmer_probe.load_pin(pin)
        code = glimmer_probe.main(
            ["probe", "--image", str(image), "--pin", str(pin)]
        )
        captured = capsys.readouterr()
        assert code != 0
        assert captured.out == ""


def test_peak_ram_and_latency_are_measured_not_defaults(tmp_path, monkeypatch):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    # Parent reads 111M then 222G; the spawned worker reports a real RSS that
    # never reaches the 222G fake, so the max is the parent's 222G fake. The
    # fake must sit far above RAM: Modal's sandbox does not reset ru_maxrss
    # across exec, so there the worker inherits the parent's peak instead of
    # measuring its own tens of MB as it does on macOS (bare child reported
    # its parent's exact peak, measured 2026-10-27).
    rams = iter([111_111_111, 222_222_222_222, 333_333_333])
    monkeypatch.setattr(glimmer_probe, "_peak_ram_bytes", lambda: next(rams))
    ticks = iter([100.0, 100.25])
    monkeypatch.setattr(glimmer_probe, "_wall_clock", lambda: next(ticks))

    proof, _receipt = glimmer_probe.validate_row(
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader()),
        pin=glimmer_probe.load_pin(pin),
    )

    assert proof.peak_ram_bytes == 222_222_222_222
    assert proof.latency_ms == 250
    assert proof.graded_image is True

    monkeypatch.setattr(glimmer_probe, "_peak_ram_bytes", lambda: 0)
    monkeypatch.setattr(glimmer_probe, "_wall_clock", lambda: 10.0)
    with pytest.raises(ProofError, match="peak RAM"):
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader())

    monkeypatch.setattr(glimmer_probe, "_peak_ram_bytes", lambda: 1)
    monkeypatch.setattr(glimmer_probe, "_wall_clock", lambda: 5.0)
    with pytest.raises(ProofError, match="latency"):
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader())


def test_private_paths_are_redacted(tmp_path, monkeypatch, capsys):
    private_dir = tmp_path / "alice private"
    private_dir.mkdir()
    image = private_dir / "owned.png"
    image.write_bytes(PNG)
    pin = _write_pin(tmp_path)
    real_build_loader = glimmer_probe.build_loader
    _patched_loader(monkeypatch, FakeLoader())
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)

    code = glimmer_probe.main(["probe", "--image", str(image), "--pin", str(pin)])
    captured = capsys.readouterr()
    assert code == 0
    assert "alice" not in captured.out
    assert str(image) not in captured.out

    missing = private_dir / "missing.png"
    code = glimmer_probe.main(["probe", "--image", str(missing), "--pin", str(pin)])
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "alice" not in captured.err
    assert "missing.png" not in captured.err
    assert "path#" in captured.err

    # Relative user paths leak the same way; only a basename hash may show.
    monkeypatch.chdir(tmp_path)
    code = glimmer_probe.main(
        ["probe", "--image", "private/alice-owned.png", "--pin", str(pin)]
    )
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert "private" not in captured.err
    assert "alice-owned.png" not in captured.err
    assert "path#" in captured.err

    loaded = glimmer_probe.load_pin(pin)
    with pytest.raises(ProofError) as excinfo:
        real_build_loader(
            loaded,
            weights="private/alice-weights.gguf",
            mmproj=tmp_path / "mmproj.gguf",
        )
    assert "private" not in str(excinfo.value)
    assert "alice-weights.gguf" not in str(excinfo.value)
    assert "path#" in str(excinfo.value)


def test_grade_must_be_exactly_one_label(tmp_path, monkeypatch):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    for answer in ("", "not faithful", "unaltered", "maybe"):
        _fake_measurements(monkeypatch)
        with pytest.raises(ProofError, match="grade"):
            glimmer_probe.probe_glimmer(
                image, pin_path=pin, loader=FakeLoader(FakeModel(answer))
            )
    _fake_measurements(monkeypatch)
    envelope = glimmer_probe.probe_glimmer(
        image, pin_path=pin, loader=FakeLoader(FakeModel("  Faithful\n"))
    )
    assert envelope["proof"]["graded_image"] is True
    assert envelope["receipt"]["grade"] == "faithful"


def test_forged_row_is_rejected(tmp_path, monkeypatch):
    envelope, pin_path, _image = _probe(tmp_path, monkeypatch, _darwin_host())
    loaded = glimmer_probe.load_pin(pin_path)

    # Hand-written bare JSON in the frozen shape carries no receipt.
    with pytest.raises(ProofError, match="no receipt"):
        glimmer_probe.validate_row(dict(envelope["proof"]), pin=loaded)
    # Any tampering with a real row breaks its content digest.
    for mutate in (
        lambda row: row["proof"].__setitem__("latency_ms", 1),
        lambda row: row["receipt"].__setitem__("grade", "altered"),
        lambda row: row["receipt"].__setitem__("digest", "0" * 64),
        lambda row: row["receipt"]["host"].__setitem__("cpu_count", 999),
    ):
        forged = copy.deepcopy(envelope)
        mutate(forged)
        with pytest.raises(ProofError, match="digest"):
            glimmer_probe.validate_row(forged, pin=loaded)
    # But the digest is plain sha256, not a signature: a forgery that
    # recomputes it is well-formed, only not yet reproduced.
    forged = copy.deepcopy(envelope)
    forged["receipt"]["grade"] = "altered"
    _redigest(forged)
    glimmer_probe.validate_row(forged, pin=loaded)


def test_platform_is_measured_not_a_label(tmp_path, monkeypatch, capsys):
    envelope, _, _ = _probe(tmp_path, monkeypatch, _darwin_host())
    assert envelope["proof"]["platform"] == "mac"
    assert envelope["receipt"]["host"]["system"] == "Darwin"
    assert envelope["receipt"]["vm_id"] is None

    # A bare Linux host is not the reference VM: the operator must declare it.
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _linux_host())
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="not declared as the reference VM"):
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader())

    # With a declared id the Linux row records it as declared, not attested.
    envelope, _, _ = _probe(tmp_path, monkeypatch, _linux_host(), vm_id=TEST_VM_ID)
    assert envelope["proof"]["platform"] == "vm_cpu"
    assert envelope["receipt"]["host"]["system"] == "Linux"
    assert envelope["receipt"]["vm_id"] == TEST_VM_ID

    # ... and a VM id on macOS is refused.
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="not the reference VM"):
        glimmer_probe.probe_glimmer(
            image, pin_path=pin, loader=FakeLoader(), vm_id=TEST_VM_ID
        )

    host = _darwin_host(system="Windows")
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: dict(host))
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="neither macOS nor Linux"):
        glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader())

    # Row level: a vm_cpu row without a declared id, or a mac row with one, fails.
    loaded = glimmer_probe.load_pin(pin)
    bare_vm = copy.deepcopy(envelope)
    bare_vm["receipt"]["vm_id"] = None
    _redigest(bare_vm)
    with pytest.raises(ProofError, match="not declared"):
        glimmer_probe.validate_row(bare_vm, pin=loaded)
    mac_envelope, _, _ = _probe(tmp_path, monkeypatch, _darwin_host())
    mac_claiming_vm = copy.deepcopy(mac_envelope)
    mac_claiming_vm["receipt"]["vm_id"] = TEST_VM_ID
    _redigest(mac_claiming_vm)
    with pytest.raises(ProofError, match="not the reference VM"):
        glimmer_probe.validate_row(mac_claiming_vm, pin=loaded)

    # There is no --platform label to abuse (--vm-id only declares, never proves).
    with pytest.raises(SystemExit) as excinfo:
        glimmer_probe.main(
            ["probe", "--image", str(image), "--platform", "mac", "--pin", str(pin)]
        )
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_verdict_verifies_local_and_declares_remote(tmp_path, monkeypatch):
    mac, pin_path, image = _probe(tmp_path, monkeypatch, _darwin_host())
    vm, _, _ = _probe(tmp_path, monkeypatch, _linux_host(), vm_id=TEST_VM_ID)
    loaded = glimmer_probe.load_pin(pin_path)

    assert _decide(monkeypatch, [], loaded, image)[0] == "UNVERIFIED"
    assert _decide(monkeypatch, [mac], loaded, image)[0] == "UNVERIFIED"
    assert _decide(monkeypatch, [vm], loaded, image)[0] == "UNVERIFIED"
    status, detail = _decide(monkeypatch, [mac, vm], loaded, image)
    assert status == "VERIFIED-LOCAL"
    assert "reproduced live on this machine" in detail
    assert "DECLARED rows: vm_cpu" in detail
    assert "self-reported, not verified" in detail
    assert "VERIFIED:" not in f"{status}: {detail}"
    assert _decide(monkeypatch, [mac, vm], loaded, image, failed=["vm_cpu"])[0] == "FAIL"
    assert _decide(monkeypatch, [mac, mac], loaded, image)[0] == "FAIL"
    forged = copy.deepcopy(vm)
    forged["proof"]["latency_ms"] += 1
    assert _decide(monkeypatch, [mac, forged], loaded, image)[0] == "FAIL"
    other = dataclasses.replace(loaded, licence="MIT")
    assert _decide(monkeypatch, [mac, vm], other, image)[0] == "FAIL"


def test_verdict_cli_exit_codes(tmp_path, monkeypatch, capsys):
    mac, pin_path, image = _probe(tmp_path, monkeypatch, _darwin_host())
    vm, _, _ = _probe(tmp_path, monkeypatch, _linux_host(), vm_id=TEST_VM_ID)
    mac_row = tmp_path / "mac.json"
    vm_row = tmp_path / "vm.json"
    mac_row.write_text(json.dumps(mac), encoding="utf-8")
    vm_row.write_text(json.dumps(vm), encoding="utf-8")
    _patched_loader(monkeypatch, FakeLoader())
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())

    def _run(extra: list[str]) -> int:
        _fake_measurements(monkeypatch)
        return glimmer_probe.main(
            ["verdict", "--pin", str(pin_path), "--image", str(image), *extra]
        )

    assert _run(["--row", str(mac_row), "--row", str(vm_row)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("VERIFIED-LOCAL:")
    assert "DECLARED rows: vm_cpu" in out
    assert "VERIFIED:" not in out
    assert _run(["--row", str(mac_row)]) == 2
    assert "UNVERIFIED" in capsys.readouterr().out
    assert _run(["--row", str(mac_row), "--row", str(vm_row), "--failed", "mac"]) == 1
    assert "FAIL" in capsys.readouterr().out

    # No pin, or an unfilled pin, is a non-zero exit: the gate never skips it.
    assert (
        glimmer_probe.main(
            [
                "verdict",
                "--image",
                str(image),
                "--row",
                str(mac_row),
                "--row",
                str(vm_row),
            ]
        )
        != 0
    )
    capsys.readouterr()
    unfilled = tmp_path / "unfilled.json"
    unfilled.write_text(
        json.dumps(_pin_dict(revision="UNVERIFIED")), encoding="utf-8"
    )
    assert (
        glimmer_probe.main(
            [
                "verdict",
                "--pin",
                str(unfilled),
                "--image",
                str(image),
                "--row",
                str(mac_row),
                "--row",
                str(vm_row),
            ]
        )
        == 1
    )
    capsys.readouterr()


def test_pin_binds_artifacts(tmp_path, monkeypatch):
    weights = tmp_path / "weights.gguf"
    weights.write_bytes(b"weights-bytes")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"mmproj-bytes")
    weights_sha = hashlib.sha256(b"weights-bytes").hexdigest()
    mmproj_sha = hashlib.sha256(b"mmproj-bytes").hexdigest()
    pin_path = _write_pin(
        tmp_path, weights_sha256=weights_sha, mmproj_sha256=mmproj_sha
    )
    loaded = glimmer_probe.load_pin(pin_path)
    monkeypatch.setitem(
        sys.modules, "llama_cpp", types.SimpleNamespace(__version__="0.3.16")
    )

    loader = glimmer_probe.build_loader(loaded, weights=weights, mmproj=mmproj)
    assert loader.artifact_digests == (weights_sha, mmproj_sha)

    # The receipt carries the measured digests, not a copy of the pin claim.
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)
    image = _write_image(tmp_path)
    fake = FakeLoader()
    fake.artifact_digests = loader.artifact_digests
    envelope = glimmer_probe.probe_glimmer(
        image, pin_path=pin_path, loader=fake
    )
    assert envelope["receipt"]["weights_sha256"] == weights_sha
    assert envelope["receipt"]["mmproj_sha256"] == mmproj_sha

    wrong = dataclasses.replace(loaded, weights_sha256="00" * 32)
    with pytest.raises(ProofError, match="does not match the pin"):
        glimmer_probe.build_loader(wrong, weights=weights, mmproj=mmproj)
    monkeypatch.setitem(
        sys.modules, "llama_cpp", types.SimpleNamespace(__version__="0.0.0")
    )
    with pytest.raises(ProofError, match="does not match the pin"):
        glimmer_probe.build_loader(loaded, weights=weights, mmproj=mmproj)


def test_load_and_grade_are_bounded(tmp_path, monkeypatch):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)

    started = time.monotonic()
    with pytest.raises(ProofError, match="timed out"):
        glimmer_probe.probe_glimmer(
            image, pin_path=pin, loader=SleepingLoader(), timeout_s=0.2
        )
    assert time.monotonic() - started < 10


def test_existing_out_is_refused(tmp_path, monkeypatch, capsys):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    out = tmp_path / "row.json"
    out.write_text('{"stale": true}', encoding="utf-8")
    _patched_loader(monkeypatch, FakeLoader())
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)

    code = glimmer_probe.main(
        ["probe", "--image", str(image), "--pin", str(pin), "--out", str(out)]
    )

    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    assert out.read_text(encoding="utf-8") == '{"stale": true}'


def test_peak_ram_units(tmp_path, monkeypatch):
    class Usage:
        ru_maxrss = 1024

    monkeypatch.setattr(resource, "getrusage", lambda who: Usage())
    monkeypatch.setattr(sys, "platform", "linux")
    assert glimmer_probe._peak_ram_bytes() == 1024 * 1024
    monkeypatch.setattr(sys, "platform", "darwin")
    assert glimmer_probe._peak_ram_bytes() == 1024


def test_weights_inside_repo_are_refused(tmp_path):
    pin = glimmer_probe.load_pin(_write_pin(tmp_path))
    with pytest.raises(ProofError, match="outside the repository"):
        glimmer_probe.build_loader(
            pin,
            weights=ROOT / "weights.gguf",
            mmproj=tmp_path / "mmproj.gguf",
        )


def test_harness_stays_out_of_product_runtime():
    # The frozen GlimmerProof contract lives in app/contracts.py; the harness must not.
    for path in (ROOT / "app").glob("*.py"):
        source = path.read_text(encoding="utf-8").lower()
        assert "glimmer_probe" not in source, path
        assert "from tools" not in source, path


def test_fabricated_local_row_fails_live_rerun(tmp_path, monkeypatch):
    mac, pin_path, image = _probe(tmp_path, monkeypatch, _darwin_host())
    vm, _, _ = _probe(tmp_path, monkeypatch, _linux_host(), vm_id=TEST_VM_ID)
    loaded = glimmer_probe.load_pin(pin_path)

    # Correctly formatted, valid digest, pin-matched — but fabricated: the
    # live model grades faithful, not altered.
    forged = copy.deepcopy(mac)
    forged["receipt"]["grade"] = "altered"
    _redigest(forged)
    glimmer_probe.validate_row(forged, pin=loaded)

    status, detail = _decide(monkeypatch, [forged, vm], loaded, image)
    assert status == "FAIL"
    assert "not reproduced live" in detail
    assert "grade differs" in detail


def test_fabricated_remote_row_is_declared_never_verified(tmp_path, monkeypatch):
    mac, pin_path, image = _probe(tmp_path, monkeypatch, _darwin_host())
    loaded = glimmer_probe.load_pin(pin_path)

    # Hand-write a distinct Linux/vm_cpu receipt with arbitrary host facts
    # and a declared vm_id, then recompute the public digest (anyone can).
    forged_vm = copy.deepcopy(mac)
    forged_vm["proof"]["platform"] = "vm_cpu"
    forged_vm["receipt"]["host"] = _linux_host(cpu_count=999, machine="aarch64")
    forged_vm["receipt"]["fingerprint"] = glimmer_probe.host_fingerprint(
        forged_vm["receipt"]["host"]
    )
    forged_vm["receipt"]["vm_id"] = "invented-ref-vm"
    _redigest(forged_vm)
    glimmer_probe.validate_row(forged_vm, pin=loaded)

    status, detail = _decide(monkeypatch, [mac, forged_vm], loaded, image)
    assert status == "VERIFIED-LOCAL"
    assert "DECLARED rows: vm_cpu (vm_id invented-ref-vm" in detail
    assert "self-reported, not verified" in detail
    assert "VERIFIED:" not in f"{status}: {detail}"


def test_vm_identity_is_declared_and_fingerprint_proves_nothing(
    tmp_path, monkeypatch
):
    mac, pin_path, image = _probe(tmp_path, monkeypatch, _darwin_host())
    # Any non-empty string is recorded as the declared VM id: it attests nothing.
    vm, _, _ = _probe(tmp_path, monkeypatch, _linux_host(), vm_id="invented-ref-vm")
    loaded = glimmer_probe.load_pin(pin_path)
    status, detail = _decide(monkeypatch, [mac, vm], loaded, image)
    assert status == "VERIFIED-LOCAL"
    assert "DECLARED rows: vm_cpu (vm_id invented-ref-vm" in detail
    assert "self-reported, not verified" in detail

    # Same hardware under two OS labels shares one fingerprint, and that no
    # longer rejects the rows: fingerprints are never compared with each other.
    hardware = {"machine": "arm64", "processor": "arm", "cpu_count": 8, "gpu": "none"}
    mac_row, pin_path, image = _probe(
        tmp_path, monkeypatch, _darwin_host(**hardware)
    )
    vm_row, _, _ = _probe(
        tmp_path, monkeypatch, _linux_host(**hardware), vm_id=TEST_VM_ID
    )
    assert (
        mac_row["receipt"]["fingerprint"] == vm_row["receipt"]["fingerprint"]
    )
    status, detail = _decide(
        monkeypatch,
        [mac_row, vm_row],
        glimmer_probe.load_pin(pin_path),
        image,
        host=_darwin_host(**hardware),
    )
    assert status == "VERIFIED-LOCAL"
    assert "DECLARED rows: vm_cpu" in detail


def test_receipt_digests_must_match_pin(tmp_path, monkeypatch):
    pin_path = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())

    # A loader that verified nothing produces no row.
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="did not verify"):
        glimmer_probe.probe_glimmer(
            image, pin_path=pin_path, loader=FakeLoader(artifact_digests=None)
        )

    # A loader whose digests disagree with the pin produces no row either.
    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="do not match the pin"):
        glimmer_probe.probe_glimmer(
            image,
            pin_path=pin_path,
            loader=FakeLoader(artifact_digests=("00" * 32, "11" * 32)),
        )

    envelope, _, _ = _probe(tmp_path, monkeypatch, _darwin_host())
    loaded = glimmer_probe.load_pin(pin_path)
    # Digest-less rows fail, even with a recomputed digest.
    for name in ("weights_sha256", "mmproj_sha256"):
        bad = copy.deepcopy(envelope)
        bad["receipt"][name] = None
        _redigest(bad)
        with pytest.raises(ProofError, match="must be a 64-hex sha256"):
            glimmer_probe.validate_row(bad, pin=loaded)
    # Well-formed but stale digests fail too: a pin change invalidates old rows.
    stale = copy.deepcopy(envelope)
    stale["receipt"]["weights_sha256"] = "00" * 32
    _redigest(stale)
    with pytest.raises(ProofError, match="does not match the pin"):
        glimmer_probe.validate_row(stale, pin=loaded)
    moved = dataclasses.replace(
        loaded, weights_sha256="00" * 32, mmproj_sha256="11" * 32
    )
    with pytest.raises(ProofError, match="does not match the pin"):
        glimmer_probe.validate_row(envelope, pin=moved)


class ForkingLoader:
    """Grades at once but leaves a descendant holding the pipe open."""

    artifact_digests = (WEIGHTS_SHA, MMPROJ_SHA)

    def load(self) -> FakeModel:
        child = os.fork()
        if child == 0:
            try:
                time.sleep(30)
            finally:
                os._exit(0)
        return FakeModel()


def test_worker_group_timeout_kills_descendant(tmp_path, monkeypatch):
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())
    _fake_measurements(monkeypatch)

    started = time.monotonic()
    with pytest.raises(ProofError, match="timed out"):
        glimmer_probe.probe_glimmer(
            image, pin_path=pin, loader=ForkingLoader(), timeout_s=0.1
        )
    assert time.monotonic() - started < 2


def _pid_dead(pid: int, timeout: float = 10.0) -> bool:
    """True once signalling the pid fails (reaped, not a lingering zombie)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    return False


def _read_pid(pid_file: Path, timeout: float = 10.0) -> int:
    """Wait for a pid file the child may still be writing, then parse it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return int(pid_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            time.sleep(0.05)
    raise AssertionError(f"{pid_file} never held a pid")


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_cleanup_kills_worker_group(tmp_path, signum):
    worker_pid = tmp_path / "worker.pid"
    grandchild_pid = tmp_path / "grandchild.pid"
    ready = tmp_path / "handlers.ready"
    (tmp_path / "slow_loader_mod.py").write_text(
        "import os\n"
        "import time\n"
        "\n"
        "def _write_pid(path):\n"
        "    tmp = path + '.tmp'\n"
        "    with open(tmp, 'w') as handle:\n"
        "        handle.write(str(os.getpid()))\n"
        "    os.replace(tmp, path)\n"
        "\n"
        "class Slow:\n"
        "    def __init__(self, worker_pid, grandchild_pid):\n"
        "        self._worker_pid = worker_pid\n"
        "        self._grandchild_pid = grandchild_pid\n"
        "    def load(self):\n"
        "        child = os.fork()\n"
        "        if child == 0:\n"
        "            _write_pid(self._grandchild_pid)\n"
        "            time.sleep(60)\n"
        "            os._exit(0)\n"
        "        _write_pid(self._worker_pid)\n"
        "        time.sleep(60)\n"
        "        raise AssertionError('unreachable')\n",
        encoding="utf-8",
    )
    script = tmp_path / "slow_probe.py"
    script.write_text(
        "import signal as _signal\n"
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from slow_loader_mod import Slow\n"
        "from tools.glimmer_probe import _load_and_grade_bounded\n"
        "\n"
        "# Record each handler install so the test signals only once the\n"
        "# harness cleanup is in place (the pid files alone cannot prove it).\n"
        "_real_signal = _signal.signal\n"
        "def _recording_signal(signum, handler):\n"
        "    installed = _real_signal(signum, handler)\n"
        "    if signum in (_signal.SIGTERM, _signal.SIGINT):\n"
        f"        with open({str(ready)!r}, 'a') as handle:\n"
        "            handle.write(str(signum) + '\\n')\n"
        "    return installed\n"
        "_signal.signal = _recording_signal\n"
        "\n"
        f"_load_and_grade_bounded(Slow({str(worker_pid)!r}, {str(grandchild_pid)!r}), b'x', 60.0)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env={},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while True:
            if (
                worker_pid.exists()
                and grandchild_pid.exists()
                and ready.exists()
                and len(ready.read_text(encoding="utf-8").split()) >= 2
            ):
                break
            if time.monotonic() >= deadline:
                pytest.fail("worker pids or handler installs never appeared")
            time.sleep(0.05)
        proc.send_signal(signum)
        assert proc.wait(timeout=15) == -signum
        for pid_file in (worker_pid, grandchild_pid):
            assert _pid_dead(_read_pid(pid_file))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_signal_handlers_are_restored(tmp_path, monkeypatch):
    before_term = signal.getsignal(signal.SIGTERM)
    before_int = signal.getsignal(signal.SIGINT)
    pin = _write_pin(tmp_path)
    image = _write_image(tmp_path)
    monkeypatch.setattr(glimmer_probe, "measure_host", lambda: _darwin_host())

    _fake_measurements(monkeypatch)
    glimmer_probe.probe_glimmer(image, pin_path=pin, loader=FakeLoader())
    assert signal.getsignal(signal.SIGTERM) is before_term
    assert signal.getsignal(signal.SIGINT) is before_int

    _fake_measurements(monkeypatch)
    with pytest.raises(ProofError, match="timed out"):
        glimmer_probe.probe_glimmer(
            image, pin_path=pin, loader=SleepingLoader(), timeout_s=0.1
        )
    assert signal.getsignal(signal.SIGTERM) is before_term
    assert signal.getsignal(signal.SIGINT) is before_int


def test_bounded_load_rejects_non_main_thread():
    errors: list[str] = []

    def _call() -> None:
        try:
            glimmer_probe._load_and_grade_bounded(FakeLoader(), b"x", 60.0)
        except ProofError as exc:
            errors.append(str(exc))

    worker = threading.Thread(target=_call)
    worker.start()
    worker.join(timeout=15)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert "main thread" in errors[0]


def test_bounded_load_rejects_second_thread_before_spawn(monkeypatch):
    started = []
    calls: list[tuple] = []
    real_popen = subprocess.Popen

    def _recording_popen(*args, **kwargs):
        calls.append(args)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", _recording_popen)
    gate = threading.Event()
    release = threading.Event()

    def _wait() -> None:
        started.append(True)
        gate.set()
        release.wait(timeout=15)

    helper = threading.Thread(target=_wait)
    helper.start()
    assert gate.wait(timeout=15)
    try:
        assert threading.active_count() >= 2
        with pytest.raises(ProofError, match="no other threads"):
            glimmer_probe._load_and_grade_bounded(FakeLoader(), b"x", 60.0)
    finally:
        release.set()
        helper.join(timeout=15)
    assert not helper.is_alive()
    assert started == [True]
    assert calls == []


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_before_popen_kills_worker_group(tmp_path, signum):
    window = tmp_path / "window.open"
    worker_pid = tmp_path / "worker.pid"
    (tmp_path / "slow_loader_mod.py").write_text(
        "import time\n"
        "\n"
        "class Slow:\n"
        "    def load(self):\n"
        "        time.sleep(60)\n"
        "        raise AssertionError('unreachable')\n",
        encoding="utf-8",
    )
    script = tmp_path / "window_probe.py"
    script.write_text(
        "import os\n"
        "import subprocess as _sp\n"
        "import sys\n"
        "import time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from slow_loader_mod import Slow\n"
        "from tools import glimmer_probe\n"
        "from tools.glimmer_probe import _load_and_grade_bounded\n"
        "\n"
        "# Hold the handler-installed pre-Popen window open: the test's\n"
        "# signal only sets the flag here, so the parent must survive it.\n"
        "def _hold_window():\n"
        f"    with open({str(window)!r}, 'w') as handle:\n"
        "        handle.write(str(os.getpid()))\n"
        "    time.sleep(10)\n"
        "glimmer_probe._POPEN_SETUP_HOOK = _hold_window\n"
        "\n"
        "# Record the worker pid right after Popen returns (before the flag\n"
        "# check kills it): the worker itself is killed too fast to report.\n"
        "_real_popen = _sp.Popen\n"
        "def _recording_popen(*args, **kwargs):\n"
        "    proc = _real_popen(*args, **kwargs)\n"
        f"    with open({str(worker_pid)!r}, 'w') as handle:\n"
        "        handle.write(str(proc.pid))\n"
        "    return proc\n"
        "_sp.Popen = _recording_popen\n"
        "\n"
        "_load_and_grade_bounded(Slow(), b'x', 60.0)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        env={},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while not window.exists():
            if time.monotonic() >= deadline:
                pytest.fail("pre-Popen window never opened")
            time.sleep(0.05)
        proc.send_signal(signum)
        # Flagged, not handled yet: the parent must survive the signal
        # while the window is held (default action would kill it at once).
        survive_until = time.monotonic() + 2.0
        while time.monotonic() < survive_until:
            assert proc.poll() is None, "signal before Popen was not flagged"
            time.sleep(0.05)
        deadline = time.monotonic() + 15
        while not worker_pid.exists():
            if time.monotonic() >= deadline:
                pytest.fail("worker pid never appeared")
            time.sleep(0.05)
        assert proc.wait(timeout=15) == -signum
        assert _pid_dead(int(worker_pid.read_text(encoding="utf-8")))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


@pytest.fixture(autouse=True)
def _grade_max_tokens(monkeypatch):
    monkeypatch.setenv("GLIMMER_MAX_TOKENS", "64")


def test_runtime_not_installed_message_keeps_prefix_and_cause(monkeypatch):
    monkeypatch.delitem(sys.modules, "llama_cpp", raising=False)
    monkeypatch.setitem(sys.modules, "llama_cpp", None)  # import raises ImportError
    with pytest.raises(ProofError, match=r"^llama\.cpp runtime is not installed: \w*Error: .*llama_cpp"):
        glimmer_probe._installed_runtime_version()
    fake = types.ModuleType("llama_cpp")
    fake.Llama = object
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", None)
    loader = glimmer_probe.LlamaCppLoader(Path("w"), Path("m"), ("a", "b"), 64)
    with pytest.raises(ProofError, match=r"^llama\.cpp runtime is not installed: \w*Error: "):
        loader.load()


def test_max_tokens_is_required_and_positive(monkeypatch):
    monkeypatch.delenv("GLIMMER_MAX_TOKENS")
    with pytest.raises(ProofError, match="max tokens not supplied"):
        glimmer_probe._resolve_max_tokens(None)
    with pytest.raises(ProofError, match="positive"):
        glimmer_probe._resolve_max_tokens(0)
    assert glimmer_probe._resolve_max_tokens("2048") == 2048


class _FakeLlm:
    def __init__(self, content, finish="stop"):
        self.content, self.finish, self.kwargs = content, finish, None

    def create_chat_completion(self, **kwargs):
        self.kwargs = kwargs
        return {"choices": [{"finish_reason": self.finish, "message": {"content": self.content}}]}


@pytest.mark.parametrize(
    "content,want",
    [
        ("altered", "altered"),
        ("thinking about it\nfaithful", "faithful"),
        ("to=self<|message|>it is not altered<|eom|><|start|>assistant to=user<|message|>altered<|eot|>", "altered"),
    ],
)
def test_grade_reads_final_answer_and_uses_pinned_max_tokens(content, want):
    llm = _FakeLlm(content)
    answer = glimmer_probe._LlamaCppModel(llm, 777).grade(b"\x89PNG\r\n\x1a\nx", "p")
    assert llm.kwargs["max_tokens"] == 777
    assert glimmer_probe._require_grade(answer) == want


@pytest.mark.parametrize("content", ["it is not faithful", "faithful or altered", "", "altered\nI am unsure"])
def test_grade_never_relaxes_the_label_rule(content):
    answer = glimmer_probe._LlamaCppModel(_FakeLlm(content), 64).grade(b"x", "p")
    with pytest.raises(ProofError, match="no altered/faithful grade"):
        glimmer_probe._require_grade(answer)


def test_truncated_answer_is_no_claim():
    with pytest.raises(ProofError, match="did not grade"):
        glimmer_probe._LlamaCppModel(_FakeLlm("faithful", finish="length"), 64).grade(b"x", "p")


@pytest.mark.parametrize(
    "answer",
    ["faith-ful", "faith ful", "altered!", "altered.", "not faithful", "FAITHFUL.", "al tered", "altered faithful"],
)
def test_require_grade_rejects_anything_but_the_exact_label(answer):
    with pytest.raises(ProofError, match="no altered/faithful grade"):
        glimmer_probe._require_grade(answer)


def test_require_grade_allows_only_surrounding_whitespace_and_case():
    assert glimmer_probe._require_grade("  Altered\n") == "altered"
    assert glimmer_probe._require_grade("faithful") == "faithful"


@pytest.mark.parametrize("command", ["probe", "verdict"])
def test_cli_passes_timeout_to_the_proof(tmp_path, monkeypatch, command):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        raise ProofError("stop")

    monkeypatch.setattr(glimmer_probe, "probe_glimmer" if command == "probe" else "verdict", fake)
    monkeypatch.setattr(glimmer_probe, "load_pin", lambda p: None)
    glimmer_probe.main([command, "--image", str(tmp_path / "i.png"), "--timeout-s", "42"])
    assert seen["timeout_s"] == 42.0


@pytest.mark.parametrize("command", ["probe", "verdict"])
@pytest.mark.parametrize("bad", ["inf", "-inf", "nan", "-1", "0", "abc"])
def test_cli_rejects_non_finite_or_non_positive_timeout_before_spawning(tmp_path, monkeypatch, command, bad):
    def spawned(*a, **k):
        raise AssertionError("must not spawn")

    monkeypatch.setattr(glimmer_probe, "_load_and_grade_bounded", spawned)
    with pytest.raises(SystemExit) as exc:
        glimmer_probe.main([command, "--image", str(tmp_path / "i.png"), "--timeout-s", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0, 0.0])
def test_bounded_load_rejects_non_finite_timeout(bad):
    with pytest.raises(ProofError, match="finite positive"):
        glimmer_probe._load_and_grade_bounded(None, b"", bad)
