"""Disagreement probe: strict config and the two-call logic with a fake client; no network (W-2816)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.providers._transport import HttpResponse
from tools import disagreement_probe as dp

CFG = dp.load_config()


class Fake:
    """Answers "no" when the question names the wrong label, "yes" otherwise."""

    def __init__(self, wrong: str, fail_on: str | None = None):
        self.wrong, self.fail_on, self.questions = wrong, fail_on, []

    def post(self, url, *, headers, body, timeout_s):
        question = json.loads(body)["messages"][0]["content"][0]["text"]
        self.questions.append(question)
        if self.fail_on and self.fail_on in question:
            raise TimeoutError("scripted")
        word = "no" if f"'{self.wrong}'" in question else "yes"
        payload = {"choices": [{"message": {"content": word}}], "usage": {"prompt_tokens": 470, "completion_tokens": 60}}
        return HttpResponse(200, json.dumps(payload).encode(), {})


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "step.png"
    cv2.imwrite(str(path), np.full((480, 640, 3), 90, np.uint8))
    return path


def _edit(tmp_path: Path, old: str, new: str | None) -> Path:
    text = dp.CONFIG.read_text()
    assert old in text
    out = tmp_path / "probe.toml"
    out.write_text(text.replace(old, new if new is not None else ""))
    return out


def test_shipped_config_loads_with_a_distinct_wrong_label():
    assert CFG.wrong_label != CFG.correct_label and CFG.receipt_path.startswith("docs/evidence/") and CFG.earlier_receipt_path.startswith("docs/evidence/")


@pytest.mark.parametrize("old,new", [
    ('wrong_label = "keyboard"', 'wrong_label = "FAN"'),            # same as the correct label
    ('wrong_label = "keyboard"', None),                              # missing key
    ('receipt_path = "docs/', 'receipt_path = "/docs/'),
    ('earlier_receipt_path = "docs/', 'earlier_receipt_path = "../docs/'),             # absolute path
    ('image = "app/', 'image = "../app/'),                          # escapes the repo
    ('timeout_s = 300', 'timeout_s = "300"'),                       # wrong type
    ('[modal]', 'extra = 1\n[modal]'),                              # misspelled / extra key
])
def test_loader_fails_loudly(tmp_path, old, new):
    with pytest.raises(dp.ProbeError):
        dp.load_config(_edit(tmp_path, old, new))


def test_wrong_label_disagrees_and_correct_label_agrees(image):
    fake = Fake(CFG.wrong_label)
    receipt = dp.run_probe(CFG, image, "k-test", fake)
    wrong, control = receipt["cases"]
    assert (wrong["case"], wrong["request_label"], wrong["model_answer"], wrong["agreement"]) == (CFG.wrong_case, CFG.wrong_label, "no", "disagree")
    assert (control["case"], control["request_label"], control["model_answer"], control["agreement"]) == (CFG.control_case, CFG.correct_label, "yes", "agree")
    assert len(fake.questions) == 2 and receipt["model_id"] == dp.ob.load_config().model_id
    assert wrong["started_at"] and wrong["finished_at"] and wrong["cost_usd"] is not None
    assert "k-test" not in json.dumps(receipt)


def test_a_failed_call_is_recorded_as_unavailable_not_dropped(image):
    receipt = dp.run_probe(CFG, image, "k", Fake(CFG.wrong_label, fail_on=CFG.correct_label))
    assert [c["agreement"] for c in receipt["cases"]] == ["disagree", "unavailable"]
    assert receipt["cases"][1]["error"] == "TimeoutError"


def test_missing_key_sends_nothing(image):
    fake = Fake(CFG.wrong_label)
    with pytest.raises(dp.ProbeError):
        dp.run_probe(CFG, image, None, fake)
    assert fake.questions == []


def test_receipt_round_trips_and_summary_names_both_cases(tmp_path, image):
    receipt = dp.run_probe(CFG, image, "k", Fake(CFG.wrong_label))
    path = dp.write_receipt(receipt, tmp_path / "var" / "r.json")
    assert json.loads(path.read_text()) == receipt and not list(path.parent.glob("*.part"))
    text = dp.summary(receipt)
    assert CFG.wrong_case in text and CFG.control_case in text and "disagree" in text


def test_concurrent_writes_never_collide_and_leave_no_part_files(tmp_path, image):
    import threading
    receipt = dp.run_probe(CFG, image, "k", Fake(CFG.wrong_label))
    path, errors, gate = tmp_path / "out" / "r.json", [], threading.Barrier(8)
    path.parent.mkdir()

    def go(n):
        try:
            gate.wait()
            dp.write_receipt({**receipt, "note": f"writer {n}"}, path)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and json.loads(path.read_text())["note"].startswith("writer ")
    assert list(path.parent.iterdir()) == [path]
