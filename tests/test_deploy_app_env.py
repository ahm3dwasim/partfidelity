"""The [modal.app_env] table: allowlisted plain settings, validated at load."""

from pathlib import Path

import pytest

from app.deploy_config import load_deploy_config

SHIPPED = Path(__file__).resolve().parents[1] / "deploy" / "modal_deploy.toml"


def load(tmp_path, table):
    text = SHIPPED.read_text()
    head, tail = text.split("[modal.app_env]", 1)
    rest = tail[tail.index("[host]"):]
    path = tmp_path / "d.toml"
    path.write_text(head + table + "\n" + rest)
    return load_deploy_config(path)


def test_shipped_table_holds_step_grouping_off():
    assert dict(load_deploy_config(SHIPPED).modal.app_env) == {"PARTFIDELITY_STEP_GROUPING": "off"}


def test_good_table(tmp_path):
    cfg = load(tmp_path, '[modal.app_env]\\nPARTFIDELITY_STEP_GROUPING = "on"'.replace("\\n", "\n"))
    assert cfg.modal.app_env == (("PARTFIDELITY_STEP_GROUPING", "on"),)


def test_missing_table_means_no_settings(tmp_path):
    assert load(tmp_path, "").modal.app_env == ()


def test_off_accepted(tmp_path):
    cfg = load(tmp_path, '[modal.app_env]\nPARTFIDELITY_STEP_GROUPING = "off"')
    assert cfg.modal.app_env == (("PARTFIDELITY_STEP_GROUPING", "off"),)


@pytest.mark.parametrize("line, message", [
    ('PARTFIDELITY_PASSWORD = "x"', "not an allowed setting"),
    ('OTHER_SETTING = "x"', "not an allowed setting"),
    ('PARTFIDELITY_PRODUCTION = "0"', "not an allowed setting"),
    ('PARTFIDELITY_STEP_GROUPING = "maybe"', "must be one of: on, off"),
    ('PARTFIDELITY_STEP_GROUPING = true', "must be one of: on, off"),
    ('PARTFIDELITY_STEP_GROUPING = 1', "must be one of: on, off"),
])
def test_bad_entries_fail_naming_the_key(tmp_path, line, message):
    with pytest.raises(RuntimeError, match=message) as err:
        load(tmp_path, "[modal.app_env]\n" + line)
    assert line.split(" =")[0].strip('"') in str(err.value)


def test_modal_not_a_table(tmp_path):
    path = tmp_path / "d.toml"
    path.write_text("modal = 1\n[host]\n")
    with pytest.raises(RuntimeError, match=r"missing the \[modal\] section"):
        load_deploy_config(path)
