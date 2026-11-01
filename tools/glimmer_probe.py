"""Offline Glimmer proof harness (the Glimmer proof module).

Loads a pinned open-weight model on one local machine, grades one owned or
synthetic image, measures peak process RAM and load-plus-grade wall latency,
and emits an evidence envelope holding the frozen ``GlimmerProof`` plus a
probe-computed receipt. The product runtime never imports this module.

Every model fact comes from ``glimmer_requirements.lock``. A missing or
ambiguous fact fails closed: the harness never substitutes a default, and a
row exists only after a real load and a real grade inside a bounded worker
process. The platform is measured from the host, never chosen by the caller,
and a Linux row additionally carries an operator-declared VM id, recorded
as declared, never attested: a bare Linux host is not the reference VM.

No signature proves origin. The receipt carries only a plain content digest
that detects accidental corruption; anyone can recompute it. A row counts as
``reproduced`` only when ``verdict`` re-runs the probe live on the machine
it runs on and the fresh outputs agree with that machine's row. Any other
row is reported as ``DECLARED`` (self-reported, never verified): the
verdict never reproduces a row from another machine, so a fabricated remote
row with a recomputed digest still validates as DECLARED. The overall
result is VERIFIED-LOCAL for the reproduced local row plus the list of
DECLARED rows (actual Mac and reference VM CPU) before any catch-table or demo
claim.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import hmac
import json
import math
import os
import pickle
import platform
import re
import resource
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.contracts import GlimmerProof  # noqa: E402

DEFAULT_PIN = Path(__file__).with_name("glimmer_requirements.lock")

Platform = Literal["mac", "vm_cpu"]
PLATFORMS: tuple[Platform, ...] = ("mac", "vm_cpu")
EXIT_CODES = {"VERIFIED-LOCAL": 0, "FAIL": 1, "UNVERIFIED": 2}

# The VM id is declared by the operator, never attested: a bare Linux host is
# not the reference VM.
VM_ID_ENV = "GLIMMER_VM_ID"
# Fixed bound for one load plus one grade; the worker is killed past it.
PROOF_TIMEOUT_S = 600.0
_TERM_GRACE_S = 2.0
_WORKER_OUTPUT_LIMIT = 1 << 20
# Test-only hook, called in the parent after the SIGTERM/SIGINT handlers
# are installed but before Popen starts the worker, to hold the pre-spawn
# window open. None in production; the regression test sets it in a subprocess.
_POPEN_SETUP_HOOK: Callable[[], None] | None = None

# A pin value that matches one of these, or contains one of the tokens below,
# is a placeholder rather than a recorded fact.
_AMBIGUOUS = frozenset(
    {
        "",
        "unknown",
        "unverified",
        "tbd",
        "todo",
        "placeholder",
        "latest",
        "main",
        "master",
        "head",
        "trunk",
        "develop",
        "dev",
        "nightly",
        "n/a",
        "na",
        "none",
    }
)
_AMBIGUOUS_TOKENS = (
    "unknown",
    "unverified",
    "placeholder",
    "tbd",
    "todo",
    "not set",
    "not recorded",
    "to fill",
    "??",
    "n/a",
)
_PIN_FIELDS = (
    "model_id",
    "revision",
    "licence",
    "licence_source",
    "quantisation",
    "runtime",
    "runtime_version",
    "weights_sha256",
    "mmproj_sha256",
)
PROOF_FIELDS = tuple(field.name for field in dataclasses.fields(GlimmerProof))
HOST_FIELDS = ("system", "release", "machine", "processor", "cpu_count", "gpu")
# The fingerprint covers hardware only. It binds a row to the machine that
# recorded it (checked for the local row); rows are never compared with each
# other, so it proves nothing about distinct machines.
_HARDWARE_FIELDS = ("machine", "processor", "cpu_count", "gpu")
RECEIPT_FIELDS = (
    "fingerprint",
    "host",
    "weights_sha256",
    "mmproj_sha256",
    "image_sha256",
    "revision",
    "runtime_version",
    "grade",
    "vm_id",
    "digest",
)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

# Only the grade language is fixed; the model and runtime pins are not.
GRADE_PROMPT = (
    "You are grading one repair-step photo. Answer with exactly one word: "
    "'altered' if the part pixels were changed, otherwise 'faithful'."
)
GRADE_LABELS = ("altered", "faithful")

# Paths may contain spaces, so stop at a newline or quote rather than whitespace.
_ABSOLUTE_PATH = re.compile(r"(?<![\w:/])/[^\n\"']*")
_HOME = str(Path.home())


class ProofError(RuntimeError):
    """A proof step failed, so no claim may be emitted."""


@dataclasses.dataclass(frozen=True)
class Pin:
    model_id: str
    revision: str
    licence: str
    licence_source: str
    quantisation: str
    runtime: str
    runtime_version: str
    weights_sha256: str
    mmproj_sha256: str


class GradedModel(Protocol):
    def grade(self, image: bytes, prompt: str) -> str: ...


class ModelLoader(Protocol):
    def load(self) -> GradedModel: ...


def redact(text: object) -> str:
    """Remove absolute paths (and the home directory) from operator-facing text."""
    cleaned = _ABSOLUTE_PATH.sub("<redacted>", str(text))
    return cleaned.replace(_HOME, "<redacted>")


def _path_token(path: object) -> str:
    """Identify a user-supplied path by a basename hash only, never the path."""
    name = Path(str(path)).name or "file"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"path#{digest}"


def _require_fact(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ProofError(f"field {name} must be a string")
    cleaned = value.strip()
    lowered = cleaned.lower()
    if (
        lowered in _AMBIGUOUS
        or any(token in lowered for token in _AMBIGUOUS_TOKENS)
        or lowered.startswith("<")
        or lowered.startswith(("replace", "fill", "set "))
        or not re.search(r"[a-z0-9]", lowered)
    ):
        raise ProofError(f"field {name} is missing or ambiguous")
    return cleaned


def _require_sha256(name: str, value: object) -> str:
    cleaned = _require_fact(name, value)
    if _SHA256.match(cleaned.lower()) is None:
        raise ProofError(f"pin field {name} must be a 64-hex sha256")
    return cleaned.lower()


def load_pin(path: Path | str = DEFAULT_PIN) -> Pin:
    """Read the exact model/runtime pin. Any missing or placeholder fact fails."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProofError(f"cannot read the pin ({type(exc).__name__})") from exc
    if not isinstance(raw, dict):
        raise ProofError("the pin must be a JSON object")
    missing = [name for name in _PIN_FIELDS if name not in raw]
    if missing:
        raise ProofError(f"pin is missing field(s): {', '.join(missing)}")
    facts = {
        name: _require_fact(name, raw[name])
        for name in _PIN_FIELDS
        if name not in ("weights_sha256", "mmproj_sha256")
    }
    for name in ("weights_sha256", "mmproj_sha256"):
        facts[name] = _require_sha256(name, raw[name])
    return Pin(**facts)


def _resolve_vm_id(explicit: str | None = None) -> str | None:
    """The operator-declared VM id, from the flag or the environment, if any."""
    raw = explicit if explicit is not None else os.environ.get(VM_ID_ENV)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(
    proof: Mapping[str, object], receipt: Mapping[str, object]
) -> str:
    """Plain content digest: detects accidental corruption, proves no origin."""
    body = _canonical({"proof": dict(proof), "receipt": dict(receipt)})
    return hashlib.sha256(body).hexdigest()


def _detect_gpu() -> str:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple-silicon"
    if Path("/dev/dri").exists():
        return "drm"
    return "none"


def measure_host() -> dict[str, object]:
    """Measure this host's identity from the OS. No caller label is accepted."""
    host: dict[str, object] = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count() or 0,
        "gpu": _detect_gpu(),
    }
    for name in ("system", "machine"):
        if not isinstance(host[name], str) or not host[name].strip():
            raise ProofError(f"the host {name} was not measured")
    for name in ("release", "processor", "gpu"):
        if not isinstance(host[name], str):
            raise ProofError(f"the host {name} was not measured")
    if type(host["cpu_count"]) is not int or host["cpu_count"] < 0:
        raise ProofError("the host CPU count was not measured")
    return host


def _derive_platform(host: Mapping[str, object], vm_id: str | None) -> Platform:
    """Measured platform plus the operator-declared VM id (declared, not attested)."""
    if host.get("system") == "Darwin":
        if vm_id is not None:
            raise ProofError("a VM id is declared on macOS: this host is not the reference VM")
        return "mac"
    if host.get("system") == "Linux":
        if vm_id is None:
            raise ProofError(
                "this Linux host is not declared as the reference VM "
                f"(pass --vm-id or export {VM_ID_ENV})"
            )
        _require_fact("vm_id", vm_id)
        return "vm_cpu"
    raise ProofError("this host is neither macOS nor Linux: no platform row")


def host_fingerprint(host: Mapping[str, object]) -> str:
    """One-way hardware identity, binding a row to its recording machine."""
    try:
        hardware = {name: host[name] for name in _HARDWARE_FIELDS}
    except KeyError as exc:
        raise ProofError("row host facts are incomplete") from exc
    return hashlib.sha256(_canonical(hardware)).hexdigest()


def _peak_ram_bytes() -> int:
    """Peak process RSS. macOS reports bytes; Linux reports KiB."""
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _wall_clock() -> float:
    return time.perf_counter()


def _measure_peak_ram() -> int:
    value = _peak_ram_bytes()
    if type(value) is not int or value <= 0:
        raise ProofError("peak RAM was not measured")
    return value


def _sha256_file(path: Path) -> str:
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ProofError(f"file not readable: {_path_token(path)}") from exc
    digest = hashlib.sha256()
    try:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    finally:
        handle.close()
    return digest.hexdigest()


def _require_outside_repo(path: Path, label: str) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    root = _REPO_ROOT.resolve()
    inside = False
    try:
        resolved.relative_to(root)
        inside = True
    except ValueError:
        # A case-insensitive filesystem can still resolve a sibling spelling.
        inside = str(resolved).casefold().startswith(str(root).casefold() + os.sep)
    if inside:
        raise ProofError(
            f"{label} must live outside the repository; private weights are never committed"
        )


def _resolve_optional_path(value: Path | str | None, env_name: str) -> Path | None:
    raw = value if value is not None else os.environ.get(env_name)
    if raw is None or str(raw).strip() == "":
        return None
    return Path(raw).expanduser()


def _import_error_text(exc: BaseException) -> str:
    return redact(f"{type(exc).__name__}: {exc}")[:300]


def _resolve_max_tokens(value: int | str | None) -> int:
    raw = value if value is not None else os.environ.get("GLIMMER_MAX_TOKENS")
    try:
        tokens = int(str(raw).strip())
    except ValueError:
        raise ProofError("max tokens not supplied (GLIMMER_MAX_TOKENS)") from None
    if tokens <= 0:
        raise ProofError("max tokens must be positive")
    return tokens


def _installed_runtime_version() -> str:
    try:
        import llama_cpp
    except ImportError as exc:
        raise ProofError(f"llama.cpp runtime is not installed: {_import_error_text(exc)}") from exc
    found = getattr(llama_cpp, "__version__", "")
    if not isinstance(found, str) or not found.strip():
        raise ProofError("installed llama.cpp runtime has no readable version")
    return found.strip()


def build_loader(
    pin: Pin,
    *,
    weights: Path | str | None = None,
    mmproj: Path | str | None = None,
    max_tokens: int | str | None = None,
) -> ModelLoader:
    """Build the pinned local loader after verifying digests and runtime build."""
    if pin.runtime != "llama.cpp":
        raise ProofError(f"unsupported runtime {redact(pin.runtime)!r}")
    weights_path = _resolve_optional_path(weights, "GLIMMER_WEIGHTS")
    mmproj_path = _resolve_optional_path(mmproj, "GLIMMER_MMPROJ")
    if weights_path is None:
        raise ProofError("weights path not supplied (--weights or GLIMMER_WEIGHTS)")
    if mmproj_path is None:
        raise ProofError("mmproj path not supplied (--mmproj or GLIMMER_MMPROJ)")
    tokens = _resolve_max_tokens(max_tokens)
    _require_outside_repo(weights_path, "weights")
    _require_outside_repo(mmproj_path, "mmproj")
    for label, path in (("weights", weights_path), ("mmproj", mmproj_path)):
        if not path.is_file():
            raise ProofError(f"{label} file not found: {_path_token(path)}")
    measured: dict[str, str] = {}
    for label, path, want in (
        ("weights", weights_path, pin.weights_sha256),
        ("mmproj", mmproj_path, pin.mmproj_sha256),
    ):
        got = _sha256_file(path)
        if not hmac.compare_digest(got, want):
            raise ProofError(f"{label} file does not match the pin")
        measured[label] = got
    installed = _installed_runtime_version()
    if installed != pin.runtime_version:
        raise ProofError(
            f"installed llama.cpp {redact(installed)} does not match the pin"
        )
    return LlamaCppLoader(
        weights_path, mmproj_path, (measured["weights"], measured["mmproj"]), tokens
    )


def _image_data_uri(image: bytes) -> str:
    mime = "image/png" if image.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


_MESSAGE_TAG = "<|message|>"
_TURN_END = re.compile(r"<\|(?:eot|eom|end|return)\|>")


def final_answer(text: str) -> str:
    """The model's last turn only: Glimmer thinks first and cannot be told not to.

    Text after the last message tag when special tokens survive, else the last
    non-empty line. The caller still demands exactly one label.
    """
    tail = text.rsplit(_MESSAGE_TAG, 1)[-1]
    tail = _TURN_END.split(tail, 1)[0]
    lines = [line for line in tail.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


class _LlamaCppModel:
    def __init__(self, llm: object, max_tokens: int) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    def grade(self, image: bytes, prompt: str) -> str:
        try:
            response = self._llm.create_chat_completion(  # type: ignore[attr-defined]
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": _image_data_uri(image)},
                            },
                        ],
                    }
                ],
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":  # cut off mid-thought: no claim
                raise ProofError("the model did not finish its answer")
            return final_answer(choice["message"]["content"] or "")
        except Exception as exc:  # noqa: BLE001 - any grade failure means no claim
            raise ProofError("the model did not grade the image") from exc


class LlamaCppLoader:
    """Minimal llama.cpp adapter, imported lazily so the repo carries no ML dependency."""

    def __init__(
        self,
        weights: Path,
        mmproj: Path,
        artifact_digests: tuple[str, str],
        max_tokens: int,
    ) -> None:
        self._max_tokens = max_tokens
        self._weights = weights
        self._mmproj = mmproj
        self.artifact_digests = artifact_digests

    def load(self) -> GradedModel:
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except ImportError as exc:
            raise ProofError(f"llama.cpp runtime is not installed: {_import_error_text(exc)}") from exc
        try:
            # No Glimmer-specific handler ships in 0.3.36; MTMDChatHandler renders the
            # chat template embedded in the GGUF (what the model card says to use).
            handler = MTMDChatHandler(clip_model_path=str(self._mmproj), verbose=False)
            llm = Llama(
                model_path=str(self._weights),
                chat_handler=handler,
                n_ctx=self._max_tokens + 2048,  # room for the image, prompt and the answer
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 - any load failure means no claim
            raise ProofError("the model did not load") from exc
        return _LlamaCppModel(llm, self._max_tokens)


def _kill_group(pid: int) -> int | None:
    """SIGTERM the worker group, escalate to SIGKILL, and reap the worker.

    The worker starts with ``start_new_session=True``, so the group id
    equals the worker pid and the parent's own group is never signalled.
    Never raises for a dead child.
    """
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.monotonic() + _TERM_GRACE_S
    while time.monotonic() < deadline:
        try:
            done, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return None
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        return _reap(pid)
    except ChildProcessError:
        return None


def _reap(pid: int) -> int:
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


def _worker_main(pickle_path: str, fd_arg: str) -> None:
    """Exec'd worker entry: run one load plus one grade, report, exit.

    Started via Popen with ``start_new_session=True``, so this process
    already leads its own group. Never returns: exits via ``os._exit``.
    """
    try:
        write_fd = int(fd_arg)
    except (TypeError, ValueError):
        os._exit(1)
    try:
        with open(pickle_path, "rb") as handle:
            loader, image_bytes = pickle.load(handle)
        model = loader.load()
        answer = model.grade(image_bytes, GRADE_PROMPT)
        peak = _measure_peak_ram()
        payload = json.dumps(
            {"answer": answer, "peak_ram_bytes": peak}
        ).encode("utf-8")
        os.write(write_fd, payload)
        code = 0
    except BaseException as exc:  # noqa: BLE001 - the parent reports it
        try:
            os.write(
                write_fd,
                json.dumps({"error": redact(str(exc))[:500]}).encode("utf-8"),
            )
        except OSError:
            pass
        code = 1
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass
        os._exit(code)


def _load_and_grade_bounded(
    loader: ModelLoader, image_bytes: bytes, timeout_s: float
) -> tuple[str, int]:
    """Run load plus one grade in a spawned worker with a fixed timeout.

    The worker starts via ``subprocess.Popen`` with ``start_new_session=True``
    (never a shell): Popen returns only after exec, so the group already
    exists and its id equals the worker pid. The deadline holds until pipe
    EOF, and past it the parent terminates, then kills and reaps the whole
    group, so a wedged native load or a descendant holding the pipe can
    neither hang the harness nor leak a zombie. SIGTERM and SIGINT handlers
    are installed before Popen: a signal after the worker exists kills its
    group first, while a signal before Popen returns only sets a flag that
    is checked right after, so no window can leak the worker. The previous
    handlers are always restored. Only the main thread with no other thread
    alive may call: handlers need the main thread, and a second thread
    could receive the signal instead.
    """
    if threading.current_thread() is not threading.main_thread():
        raise ProofError(
            "bounded load must run on the main thread: "
            "signal cleanup needs main-thread handlers"
        )
    if threading.active_count() != 1:
        raise ProofError(
            "bounded load must run with no other threads alive: "
            "a second thread could receive the cleanup signal instead"
        )
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError) as exc:
        raise ProofError("the proof timeout is not a number") from exc
    if not (math.isfinite(timeout) and timeout > 0):
        raise ProofError("the proof timeout must be a finite positive number")
    try:
        payload = pickle.dumps((loader, bytes(image_bytes)))
    except Exception as exc:
        raise ProofError("the bounded loader cannot run in the worker") from exc
    read_fd, write_fd = os.pipe()
    fd, pickle_path = tempfile.mkstemp(prefix="glimmer-worker-", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
    except OSError as exc:
        os.close(read_fd)
        os.close(write_fd)
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        raise ProofError("the bounded worker could not start") from exc
    state: dict[str, object] = {"proc": None, "signum": None}
    cleaned = False

    def _cleanup_group() -> None:
        """Terminate, then kill and reap the worker group. Idempotent, total."""
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        proc = state["proc"]
        if proc is not None:
            try:
                _kill_group(proc.pid)  # type: ignore[attr-defined]
            except OSError:
                pass

    def _die_by(signum: int) -> None:
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError):
            pass
        try:
            os.kill(os.getpid(), signum)
        except OSError:
            os._exit(128 + signum)

    def _on_signal(signum: int, _frame: object) -> None:
        if state["proc"] is None:
            if state["signum"] is None:
                state["signum"] = signum
            return
        _cleanup_group()
        assert isinstance(signum, int)
        _die_by(signum)

    old_term: object = None
    old_int: object = None
    try:
        old_term = signal.signal(signal.SIGTERM, _on_signal)
        try:
            old_int = signal.signal(signal.SIGINT, _on_signal)
        except BaseException:
            signal.signal(signal.SIGTERM, old_term)  # type: ignore[arg-type]
            raise
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        raise
    try:
        if _POPEN_SETUP_HOOK is not None:
            _POPEN_SETUP_HOOK()
        env = dict(os.environ)
        path_dirs: list[str] = []
        seen: set[str] = set()
        for entry in sys.path:
            if entry and entry not in seen and os.path.isdir(entry):
                seen.add(entry)
                path_dirs.append(entry)
        if path_dirs:
            env["PYTHONPATH"] = os.pathsep.join(path_dirs)
        script = str(Path(__file__).resolve())
        proc = subprocess.Popen(
            [sys.executable, script, "--worker", pickle_path, str(write_fd)],
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_fd,),
            env=env,
        )
        state["proc"] = proc
        pid = proc.pid
    except BaseException:
        signal.signal(signal.SIGTERM, old_term)  # type: ignore[arg-type]
        if old_int is not None:
            try:
                signal.signal(signal.SIGINT, old_int)  # type: ignore[arg-type]
            except (OSError, ValueError):
                pass
        try:
            os.close(read_fd)
        except OSError:
            pass
        try:
            os.close(write_fd)
        except OSError:
            pass
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
        raise
    try:
        os.close(write_fd)
    except OSError:
        pass
    flagged = state["signum"]
    if flagged is not None:
        assert isinstance(flagged, int)
        _cleanup_group()
        _die_by(flagged)
    chunks: list[bytes] = []
    size = 0
    exit_code: int | None = None
    eof = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _cleanup_group()
                raise ProofError("load or grade timed out")
            ready, _, _ = select.select([read_fd], [], [], min(remaining, 1.0))
            if ready:
                try:
                    chunk = os.read(read_fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    eof = True
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _WORKER_OUTPUT_LIMIT:
                    _cleanup_group()
                    raise ProofError("the load/grade worker returned too much data")
                continue
            if exit_code is None:
                done, status = os.waitpid(pid, os.WNOHANG)
                if done:
                    exit_code = os.waitstatus_to_exitcode(status)
    except BaseException:
        if not eof:
            _cleanup_group()
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)  # type: ignore[arg-type]
        signal.signal(signal.SIGINT, old_int)  # type: ignore[arg-type]
        try:
            os.close(read_fd)
        except OSError:
            pass
        try:
            os.unlink(pickle_path)
        except OSError:
            pass
    if exit_code is None:
        exit_code = _reap(pid)
    proc = state["proc"]
    assert proc is not None
    proc.returncode = exit_code  # type: ignore[attr-defined]
    raw = b"".join(chunks)
    if exit_code != 0 and not raw:
        raise ProofError("the load/grade worker died")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProofError("the load/grade worker returned unreadable data") from exc
    if exit_code != 0 or not isinstance(result, dict) or "error" in result:
        detail = result.get("error") if isinstance(result, dict) else None
        raise ProofError(str(detail) if detail else "the load/grade worker failed")
    answer = result.get("answer")
    peak = result.get("peak_ram_bytes")
    if not isinstance(answer, str) or type(peak) is not int:
        raise ProofError("the load/grade worker returned unreadable data")
    return answer, peak


def _require_grade(answer: str) -> str:
    """Accept exactly one label, so 'not faithful' cannot pass as 'faithful'."""
    cleaned = answer.strip().lower()
    if cleaned in GRADE_LABELS:
        return cleaned
    raise ProofError("the model returned no altered/faithful grade")


def validate_proof(proof: GlimmerProof) -> None:
    """Reject any proof carrying an ambiguous fact or an unmeasured value."""
    if not isinstance(proof, GlimmerProof):
        raise ProofError("proof is not a GlimmerProof")
    _require_fact("model_id", proof.model_id)
    _require_fact("licence", proof.licence)
    _require_fact("quantisation", proof.quantisation)
    if type(proof.peak_ram_bytes) is not int or proof.peak_ram_bytes <= 0:
        raise ProofError("peak_ram_bytes must be a measured positive integer")
    if type(proof.latency_ms) is not int or proof.latency_ms <= 0:
        raise ProofError("latency_ms must be a measured positive integer")
    if proof.platform not in PLATFORMS:
        raise ProofError(f"platform must be one of {', '.join(PLATFORMS)}")
    if proof.graded_image is not True:
        raise ProofError("graded_image must be True for a passing row")


def _proof_from_dict(data: object) -> GlimmerProof:
    if not isinstance(data, Mapping):
        raise ProofError("row proof must be a JSON object")
    keys = set(data)
    missing = set(PROOF_FIELDS) - keys
    extra = keys - set(PROOF_FIELDS)
    if missing:
        raise ProofError(f"row missing field(s): {', '.join(sorted(missing))}")
    if extra:
        raise ProofError(f"row has unknown field(s): {', '.join(sorted(extra))}")
    for name in ("model_id", "licence", "quantisation", "platform"):
        if not isinstance(data[name], str):
            raise ProofError(f"{name} must be a string")
    for name in ("peak_ram_bytes", "latency_ms"):
        if type(data[name]) is not int:
            raise ProofError(f"{name} must be an integer")
    if type(data["graded_image"]) is not bool:
        raise ProofError("graded_image must be a boolean")
    proof = GlimmerProof(**data)  # type: ignore[arg-type]
    validate_proof(proof)
    return proof


def validate_row(
    data: Mapping[str, object], *, pin: Pin
) -> tuple[GlimmerProof, Mapping[str, object]]:
    """Validate an envelope against the frozen shape, the digest, and the pin.

    The digest detects corruption, not origin: anyone can recompute it, so a
    well-formed row is only ``DECLARED`` (self-reported, never verified)
    until ``verdict`` reproduces it live on this machine.
    """
    if not isinstance(data, Mapping):
        raise ProofError("row must be a JSON object")
    if set(data) != {"proof", "receipt"}:
        if set(data) == set(PROOF_FIELDS):
            raise ProofError("row is not probe evidence: it carries no receipt")
        raise ProofError("row must hold exactly proof and receipt")
    proof = _proof_from_dict(data["proof"])
    receipt = data["receipt"]
    if not isinstance(receipt, Mapping) or set(receipt) != set(RECEIPT_FIELDS):
        raise ProofError(
            "row receipt must hold exactly " + ", ".join(sorted(RECEIPT_FIELDS))
        )
    host = receipt["host"]
    if not isinstance(host, Mapping) or set(host) != set(HOST_FIELDS):
        raise ProofError(
            "row host facts must hold exactly " + ", ".join(sorted(HOST_FIELDS))
        )
    for name in ("system", "release", "machine", "processor", "gpu"):
        if not isinstance(host[name], str):
            raise ProofError(f"row host {name} must be a string")
    if type(host["cpu_count"]) is not int:
        raise ProofError("row host cpu_count must be an integer")
    for name in ("weights_sha256", "mmproj_sha256"):
        value = receipt[name]
        if not isinstance(value, str) or _SHA256.match(value.lower()) is None:
            raise ProofError(f"row {name} must be a 64-hex sha256")
        if value.lower() != getattr(pin, name):
            raise ProofError(f"the row does not match the pin field {name}")
    image_sha = receipt["image_sha256"]
    if not isinstance(image_sha, str) or _SHA256.match(image_sha.lower()) is None:
        raise ProofError("row image_sha256 must be a 64-hex sha256")
    for name in ("digest", "grade", "revision", "runtime_version"):
        if not isinstance(receipt[name], str):
            raise ProofError(f"row {name} must be a string")
    vm_id = receipt["vm_id"]
    if vm_id is not None and not isinstance(vm_id, str):
        raise ProofError("row vm_id must be null or a string")
    body = {name: receipt[name] for name in RECEIPT_FIELDS if name != "digest"}
    expected = _digest(dict(data["proof"]), body)
    if not hmac.compare_digest(receipt["digest"], expected):
        raise ProofError("row digest does not match: the row was edited or corrupted")
    fingerprint = receipt["fingerprint"]
    if not isinstance(fingerprint, str) or fingerprint != host_fingerprint(host):
        raise ProofError("row fingerprint does not match its host facts")
    if _derive_platform(host, vm_id) != proof.platform:
        raise ProofError("row platform is not the measured host")
    if receipt["grade"] not in GRADE_LABELS:
        raise ProofError("row grade is not an altered/faithful grade")
    for field in ("model_id", "licence", "quantisation"):
        if getattr(proof, field) != getattr(pin, field):
            raise ProofError(f"the row does not match the pin field {field}")
    for field in ("revision", "runtime_version"):
        _require_fact(f"row {field}", receipt[field])
        if receipt[field] != getattr(pin, field):
            raise ProofError(f"the row does not match the pin field {field}")
    return proof, receipt


def _probe_with_pin(
    image_path: Path | str,
    *,
    pin: Pin,
    loader: ModelLoader | None,
    weights: Path | str | None,
    mmproj: Path | str | None,
    timeout_s: float,
    vm_id: str | None,
) -> dict[str, object]:
    """Shared probe body: ``verdict`` re-runs this live to reproduce the row."""
    host = measure_host()
    platform_name = _derive_platform(host, vm_id)
    fingerprint = host_fingerprint(host)
    image = Path(image_path)
    try:
        image_bytes = image.read_bytes()
    except OSError as exc:
        raise ProofError(f"image not readable: {_path_token(image)}") from exc
    if not image_bytes:
        raise ProofError(f"image is empty: {_path_token(image)}")

    active_loader = (
        loader if loader is not None else build_loader(pin, weights=weights, mmproj=mmproj)
    )
    digests = getattr(active_loader, "artifact_digests", None)
    if digests is None:
        raise ProofError("the loader did not verify the pinned artifacts")
    try:
        weights_sha256, mmproj_sha256 = digests
    except (TypeError, ValueError) as exc:
        raise ProofError("the loader did not verify the pinned artifacts") from exc
    if weights_sha256 != pin.weights_sha256 or mmproj_sha256 != pin.mmproj_sha256:
        raise ProofError("the loaded artifacts do not match the pin")
    # Latency is the whole proof's wall time: load + one grade.
    started = _wall_clock()
    peak = _measure_peak_ram()
    answer, worker_peak = _load_and_grade_bounded(
        active_loader, image_bytes, timeout_s
    )
    try:
        elapsed_s = float(_wall_clock()) - float(started)
    except (TypeError, ValueError) as exc:
        raise ProofError("latency was not measured") from exc
    peak = max(peak, worker_peak, _measure_peak_ram())

    grade = _require_grade(answer)
    latency_ms = int(round(elapsed_s * 1000))
    if latency_ms <= 0:
        raise ProofError("latency was not measured")

    proof = GlimmerProof(
        model_id=pin.model_id,
        licence=pin.licence,
        quantisation=pin.quantisation,
        peak_ram_bytes=peak,
        latency_ms=latency_ms,
        platform=platform_name,
        graded_image=True,
    )
    validate_proof(proof)
    proof_dict = dataclasses.asdict(proof)
    receipt_body = {
        "fingerprint": fingerprint,
        "host": host,
        "weights_sha256": weights_sha256,
        "mmproj_sha256": mmproj_sha256,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "revision": pin.revision,
        "runtime_version": pin.runtime_version,
        "grade": grade,
        "vm_id": vm_id,
    }
    digest = _digest(proof_dict, receipt_body)
    return {"proof": proof_dict, "receipt": {**receipt_body, "digest": digest}}


def probe_glimmer(
    image_path: Path | str,
    *,
    pin_path: Path | str = DEFAULT_PIN,
    loader: ModelLoader | None = None,
    weights: Path | str | None = None,
    mmproj: Path | str | None = None,
    timeout_s: float = PROOF_TIMEOUT_S,
    vm_id: str | None = None,
) -> dict[str, object]:
    """Load, grade one image in a bounded worker, measure, and emit a row.

    The platform is measured from this host, never chosen by the caller; on
    Linux the operator must also declare the VM id, which the row records
    as declared, never attested.
    """
    pin = load_pin(pin_path)
    return _probe_with_pin(
        image_path,
        pin=pin,
        loader=loader,
        weights=weights,
        mmproj=mmproj,
        timeout_s=timeout_s,
        vm_id=_resolve_vm_id(vm_id),
    )


# Receipt fields a live re-run must reproduce. Latency and peak RAM vary run
# to run, so they are measured but never compared.
_REPRODUCED_FIELDS = (
    "fingerprint",
    "weights_sha256",
    "mmproj_sha256",
    "image_sha256",
    "revision",
    "runtime_version",
    "grade",
    "vm_id",
)


def verdict(
    rows: Iterable[Mapping[str, object]],
    *,
    failed: Iterable[str] = (),
    pin: Pin,
    image_path: Path | str | None,
    loader: ModelLoader | None = None,
    weights: Path | str | None = None,
    mmproj: Path | str | None = None,
    vm_id: str | None = None,
    timeout_s: float = PROOF_TIMEOUT_S,
) -> tuple[str, str]:
    """VERIFIED-LOCAL for the live-reproduced local row, plus DECLARED rows.

    The probe is re-run live on this machine and its fresh outputs are
    compared with this machine's row: only an agreeing row counts as
    ``reproduced``. Every other row is reported as ``DECLARED``
    (self-reported, never verified): the verdict never reproduces a row
    from another machine, so a fabricated remote row with a recomputed
    digest still validates as DECLARED. Rows are never compared with each
    other: shared fingerprints prove nothing and are not checked.
    """
    failed_platforms = list(failed)
    unknown = [name for name in failed_platforms if name not in PLATFORMS]
    if unknown:
        return "FAIL", f"unknown failed platform: {', '.join(unknown)}"
    if failed_platforms:
        return (
            "FAIL",
            "a measured load or grade failed on: "
            + ", ".join(sorted(set(failed_platforms))),
        )

    proofs: dict[str, GlimmerProof] = {}
    receipts: dict[str, Mapping[str, object]] = {}
    for data in rows:
        try:
            proof, receipt = validate_row(data, pin=pin)
        except ProofError as exc:
            return "FAIL", str(exc)
        if proof.platform in proofs:
            return "FAIL", f"duplicate row for platform {proof.platform}"
        proofs[proof.platform] = proof
        receipts[proof.platform] = receipt

    if image_path is None:
        raise ProofError("verdict needs the graded image to re-run the probe (--image)")
    declared_vm = _resolve_vm_id(vm_id)
    try:
        local_host = measure_host()
        local_platform = _derive_platform(local_host, declared_vm)
    except ProofError as exc:
        return "UNVERIFIED", f"this machine has no platform row: {exc}"
    local_fingerprint = host_fingerprint(local_host)

    missing = [name for name in PLATFORMS if name not in proofs]
    if local_platform in missing:
        return "UNVERIFIED", "missing platform row(s): " + ", ".join(missing)
    local_receipt = receipts[local_platform]
    if str(local_receipt["fingerprint"]) != local_fingerprint:
        return (
            "UNVERIFIED",
            f"the {local_platform} row is from another machine: "
            "reproduce it where it was recorded",
        )
    try:
        live = _probe_with_pin(
            image_path,
            pin=pin,
            loader=loader,
            weights=weights,
            mmproj=mmproj,
            timeout_s=timeout_s,
            vm_id=declared_vm,
        )
    except ProofError as exc:
        return "FAIL", f"the {local_platform} row was not reproduced live: {exc}"
    live_receipt = live["receipt"]
    assert isinstance(live_receipt, dict)
    for name in _REPRODUCED_FIELDS:
        if live_receipt[name] != local_receipt[name]:
            return (
                "FAIL",
                f"the {local_platform} row was not reproduced live: {name} differs",
            )
    if missing:
        return "UNVERIFIED", "missing platform row(s): " + ", ".join(missing)
    declared_rows = [
        f"{name} (vm_id {receipts[name]['vm_id']}, self-reported, not verified)"
        if name == "vm_cpu"
        else f"{name} (self-reported, not verified)"
        for name in PLATFORMS
        if name != local_platform
    ]
    detail = (
        f"the {local_platform} row was reproduced live on this machine; "
        f"DECLARED rows: {', '.join(declared_rows)}"
    )
    return "VERIFIED-LOCAL", detail


def _write_row(path: Path, payload: str) -> None:
    """Write a row atomically, refusing to overwrite or invalidate prior evidence."""
    token = _path_token(path)
    if path.exists() or path.is_symlink():
        raise ProofError(f"refusing to overwrite an existing row: {token}")
    try:
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=path.parent
        )
    except OSError as exc:
        raise ProofError(f"cannot write a row beside: {token}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_name, path)
        except FileExistsError as exc:
            raise ProofError(
                f"refusing to overwrite an existing row: {token}"
            ) from exc
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass


def _finite_positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glimmer_probe",
        description="Offline Glimmer proof harness (never part of the product runtime).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe", help="load, grade one image, and emit a row")
    probe.add_argument("--image", required=True, type=Path)
    probe.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    probe.add_argument("--weights", type=Path, default=None)
    probe.add_argument("--mmproj", type=Path, default=None)
    probe.add_argument("--vm-id", type=str, default=None)
    probe.add_argument("--out", type=Path, default=None)
    probe.add_argument("--timeout-s", type=_finite_positive, default=PROOF_TIMEOUT_S)

    decide = commands.add_parser(
        "verdict",
        help="VERIFIED-LOCAL for the live-reproduced local row; every "
        "other row is DECLARED (self-reported, never verified)",
    )
    decide.add_argument("--row", action="append", default=[], type=Path)
    decide.add_argument("--failed", action="append", default=[], choices=PLATFORMS)
    decide.add_argument("--pin", type=Path, default=DEFAULT_PIN)
    decide.add_argument("--image", required=True, type=Path)
    decide.add_argument("--weights", type=Path, default=None)
    decide.add_argument("--mmproj", type=Path, default=None)
    decide.add_argument("--vm-id", type=str, default=None)
    decide.add_argument("--timeout-s", type=_finite_positive, default=PROOF_TIMEOUT_S)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "probe":
            envelope = probe_glimmer(
                args.image,
                pin_path=args.pin,
                weights=args.weights,
                mmproj=args.mmproj,
                vm_id=args.vm_id,
                timeout_s=args.timeout_s,
            )
            payload = json.dumps(envelope, indent=2, sort_keys=True) + "\n"
            if args.out is not None:
                _write_row(args.out, payload)
            sys.stdout.write(payload)
            return 0

        rows = [
            json.loads(Path(row_path).read_text(encoding="utf-8"))
            for row_path in args.row
        ]
        pin = load_pin(args.pin)
        status, detail = verdict(
            rows,
            failed=args.failed,
            pin=pin,
            image_path=args.image,
            weights=args.weights,
            mmproj=args.mmproj,
            vm_id=args.vm_id,
            timeout_s=args.timeout_s,
        )
        print(f"{status}: {redact(detail)}")
        return EXIT_CODES[status]
    except ProofError as exc:
        print(f"proof failed: {redact(exc)}", file=sys.stderr)
        return EXIT_CODES["FAIL"]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"proof failed: {type(exc).__name__}", file=sys.stderr)
        return EXIT_CODES["FAIL"]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        if len(sys.argv) != 4:
            os._exit(2)
        _worker_main(sys.argv[2], sys.argv[3])
    raise SystemExit(main())
