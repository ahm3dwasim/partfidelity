"""Exercise Modal's relocated entry point without contacting Modal."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]



@pytest.fixture(autouse=True)
def stub_private_media(monkeypatch, synthetic_sample_bundle):
    from tools import catch_media, sample_media
    monkeypatch.setattr(sample_media, "unpack", lambda root: None)
    monkeypatch.setattr(catch_media, "unpack", lambda root: None)  # the real refusal is test_deploy_fails_when_the_catch_media_is_absent
    global SAMPLE_TEST_BUNDLE
    SAMPLE_TEST_BUNDLE = synthetic_sample_bundle


def shipped(deploy_script=None, sources=None):
    """(folders, files) the image would receive, read from the mocked build_image; `sources` collects each add_local_dir source path."""
    import runpy
    import types
    from unittest.mock import MagicMock, patch

    modal = types.ModuleType("modal")
    modal.is_local = lambda: True
    modal.Image = MagicMock()
    image = modal.Image.debian_slim.return_value
    for step in ("apt_install", "pip_install", "env", "add_local_dir", "add_local_file"):
        getattr(image, step).return_value = image
    modal.App = MagicMock()
    modal.App.return_value.function = lambda **kwargs: lambda f: f
    modal.Volume = MagicMock()
    modal.Secret = MagicMock()
    modal.concurrent = lambda **kwargs: lambda f: f
    modal.asgi_app = lambda: lambda f: f
    saved = sys.modules.get("modal")
    sys.modules["modal"] = modal
    sys.path.insert(0, str(REPO))
    try:  # the private frames are checked at build time (test_how_it_works); this tree reads only what is shipped
        with patch("app.how_media.problems", return_value=[]):
            ns = runpy.run_path(str(deploy_script or REPO / "deploy" / "modal_app.py"))
    finally:
        sys.path.remove(str(REPO))
        if saved is None:
            del sys.modules["modal"]
        else:
            sys.modules["modal"] = saved
    remote = ns["modal_cfg"].remote_root
    if sources is not None:
        sources.extend(Path(c.args[0]) for c in image.add_local_dir.call_args_list)
    rel = lambda call: Path(call.args[1]).relative_to(remote).as_posix()
    return ([rel(c) for c in image.add_local_dir.call_args_list],
            [rel(c) for c in image.add_local_file.call_args_list])


def copy_shipped_tree(root):
    folders, files = shipped()
    for folder in folders:
        shutil.copytree(REPO / folder, root / folder)
    for name in files:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        source = REPO / name
        if not source.is_file() and name.startswith("app/sample_real/"):
            source = SAMPLE_TEST_BUNDLE / Path(name).relative_to("app/sample_real")
        shutil.copy(source, root / name)


def runtime_repo_paths():
    """Repo files the app names by string at runtime, outside app/ (config/film.toml, static css, ...)."""
    import ast
    import re
    # Built from parents[...] / "config" / "<name>": film_config.py, wait_config.py, upload_config.py, open_readback.py and guide_text.py.
    found = {"config/film.toml", "config/wait.toml", "config/upload.toml", "config/open_readback.toml", "config/guide_text.toml"}
    tops = {p.name for p in REPO.iterdir() if p.is_dir() and p.name not in {"app", ".git"}}
    for source in (REPO / "app").rglob("*.py"):
        text = source.read_text(encoding="utf-8")
        found.update(f"config/{name}" for name in re.findall(
            r'''/\s*["']config["']\s*/\s*["']([^"']+\.toml)["']''', text))
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and "/" in node.value:
                if node.value.split("/")[0] in tops and (REPO / node.value).is_file():
                    found.add(node.value)
    return found


def test_every_runtime_file_is_shipped():
    folders, files = shipped()
    missing = [p for p in sorted(runtime_repo_paths())
               if p not in files and not any(p.startswith(f + "/") for f in folders)]
    assert not missing, f"read at runtime but not in the image: {missing}"
    assert "config/film.toml" in files
    assert "config/retention.toml" in files
    assert "config/open_readback.toml" in files


def test_retention_config_ships_in_docker_and_wheel():
    import tomllib
    dockerfile = (REPO / "Dockerfile").read_text()
    assert "COPY config/film.toml config/retention.toml config/exports.toml config/display_images.toml config/" in dockerfile
    assert "pillow" in dockerfile.lower()  # display copies need Pillow at runtime
    assert "COPY config/ config/" not in dockerfile and "teardown.toml" not in dockerfile
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["config/retention.toml"] == "config/retention.toml"
    assert included["config/exports.toml"] == "config/exports.toml"
    assert included["config/display_images.toml"] == "config/display_images.toml"
    assert "config/exports.toml" in shipped()[1]


def test_app_and_home_work_from_shipped_tree_only(tmp_path):
    root = tmp_path / "remote"
    copy_shipped_tree(root)
    script = (
        "import sys; sys.path.insert(0, sys.argv[1])\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import create_app\n"
        "r = TestClient(create_app()).get('/')\n"
        "assert r.status_code == 200, r.status_code\n"
        "assert TestClient(create_app()).get('/static/ui/film.js').status_code == 200\n"
        "from app import film_config\n"
        "pub = film_config.public(film_config.load())\n"
        "urls = [pub['poster_url']] + [f\"{pub['base_url']}/{s['dir']}/{b}?v={pub['version']}\" for s in pub['sets'] for b in pub['bundles']]\n"
        "bad = [u for u in urls if TestClient(create_app()).get(u).status_code != 200]\n"
        "assert len(urls) > 2 and not bad, bad\n"
        "print('HOME=ok')\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)], cwd=root, capture_output=True, text=True,
                            env={"PATH": os.defpath, "PARTFIDELITY_ROOT": str(root),
                                 "PARTFIDELITY_DATA_ROOT": str(tmp_path / "data"), "PARTFIDELITY_DB_PATH": str(tmp_path / "db.sqlite3")})
    assert result.returncode == 0, result.stderr[-2000:]
    assert "HOME=ok" in result.stdout


def run_entry(tmp_path, *, local, root_env):
    root = tmp_path / "remote"
    copy_shipped_tree(root)
    entry = tmp_path / "container" / "modal_app.py"
    entry.parent.mkdir()
    shutil.copy(REPO / "deploy" / "modal_app.py", entry)
    script = '''
import runpy, sys, types
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock
modal = types.ModuleType("modal")
modal.is_local = lambda: LOCAL
modal.Image = MagicMock()
modal.App = MagicMock()
modal.Volume = MagicMock()
modal.Secret = MagicMock()
modal.concurrent = lambda **kwargs: lambda f: f
modal.asgi_app = lambda: lambda f: f
modal.App.return_value.function = lambda **kwargs: lambda f: f
sys.modules["modal"] = modal
namespace = runpy.run_path(ENTRY)
print("ROOT=" + str(namespace["ROOT"]))
print("APP=" + namespace["cfg"].modal.app_name)
if not LOCAL:
    web_globals = namespace["web"].__globals__
    cfg = web_globals["cfg"]
    state = Path(ENTRY).parent / "state"
    host = replace(cfg.host, data_root=state / "data", snapshot_dir=state / "snapshots",
                   local_db=state / "db.sqlite3")
    web_globals["cfg"] = replace(cfg, host=host)
    application = namespace["web"]()
    assert Path(sys.modules["app.hosting"].__file__).is_relative_to(namespace["ROOT"])
    assert application is not None
    print("WEB=ok")
'''.replace("LOCAL", repr(local)).replace("ENTRY", repr(str(entry)))
    child_env = {"PATH": os.defpath}
    if root_env:
        child_env["PARTFIDELITY_ROOT"] = str(root)
    result = subprocess.run([sys.executable, "-c", script], env=child_env, cwd=entry.parent, capture_output=True, text=True)
    return result, root


def test_container_uses_configured_root(tmp_path):
    result, root = run_entry(tmp_path, local=False, root_env=True)
    assert result.returncode == 0, result.stderr
    assert f"ROOT={root}\n" in result.stdout
    assert "APP=partfidelity\n" in result.stdout
    assert "WEB=ok\n" in result.stdout


def test_container_requires_root(tmp_path):
    result, _ = run_entry(tmp_path, local=False, root_env=False)
    assert result.returncode != 0
    assert "PARTFIDELITY_ROOT" in result.stderr


def test_local_image_receives_configured_root(monkeypatch):
    import runpy
    import types
    from unittest.mock import MagicMock

    modal = types.ModuleType("modal")
    modal.is_local = lambda: True
    modal.Image = MagicMock()
    image = modal.Image.debian_slim.return_value
    image.apt_install.return_value = image
    image.pip_install.return_value = image
    image.env.return_value = image
    image.add_local_dir.return_value = image
    modal.App = MagicMock()
    modal.App.return_value.function = lambda **kwargs: lambda f: f
    modal.Volume = MagicMock()
    modal.Secret = MagicMock()
    modal.concurrent = lambda **kwargs: lambda f: f
    modal.asgi_app = lambda: lambda f: f
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.syspath_prepend(str(REPO))

    monkeypatch.setattr("app.how_media.problems", lambda *a, **k: [])  # frames are checked in test_how_it_works
    namespace = runpy.run_path(str(REPO / "deploy" / "modal_app.py"))
    assert namespace["ROOT"] == REPO
    image.env.assert_called_once_with({
        "PARTFIDELITY_PRODUCTION": "1",
        "PARTFIDELITY_ROOT": str(namespace["modal_cfg"].remote_root),
    })


def test_deploy_fails_when_the_private_media_bundle_is_absent(tmp_path, monkeypatch):
    """Release guard: building the deploy image refuses to go on without the approved media."""
    import runpy
    import types
    from unittest.mock import MagicMock
    from tools import sample_media

    monkeypatch.undo()  # drop the autouse unpack stub; the real unpack must run
    modal = types.ModuleType("modal")
    modal.is_local = lambda: True
    modal.Image = modal.App = modal.Volume = modal.Secret = MagicMock()
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.syspath_prepend(str(REPO))
    (tmp_path / "deploy").mkdir()
    for name in ("modal_app.py", "modal_deploy.toml"):
        shutil.copy(REPO / "deploy" / name, tmp_path / "deploy" / name)
    monkeypatch.setattr(sample_media, "BUNDLE", tmp_path / "var" / "none.zip")
    monkeypatch.setattr("tools.catch_media.unpack", lambda root: None)  # this test is about the sample media refusal
    with pytest.raises(RuntimeError, match="Sample media missing.*python tools/sample_media.py unpack"):
        runpy.run_path(str(tmp_path / "deploy" / "modal_app.py"))


def test_deploy_fails_when_the_catch_media_is_absent(tmp_path, monkeypatch):
    """Release guard: the deploy checks the real approved manifest and refuses without the catch images."""
    import runpy
    import types
    from unittest.mock import MagicMock

    monkeypatch.undo()  # drop the autouse unpack stubs; the real unpack must run
    modal = types.ModuleType("modal")
    modal.is_local = lambda: True
    modal.Image = modal.App = modal.Volume = modal.Secret = MagicMock()
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.syspath_prepend(str(REPO))
    (tmp_path / "deploy").mkdir()
    (tmp_path / "config").mkdir()
    for name in ("modal_app.py", "modal_deploy.toml"):
        shutil.copy(REPO / "deploy" / name, tmp_path / "deploy" / name)
    for name in ("catch_media.json", "catch_media.sha256"):  # the real approved manifest, with no images beside it
        shutil.copy(REPO / "config" / name, tmp_path / "config" / name)
    monkeypatch.setattr("tools.sample_media.unpack", lambda root: None)
    with pytest.raises(RuntimeError, match="Catch media missing.*python tools/catch_media.py unpack"):
        runpy.run_path(str(tmp_path / "deploy" / "modal_app.py"))


def test_deploy_unpacks_a_manifest_matched_catch_bundle_and_ships_the_images(tmp_path, monkeypatch):
    """Success path: build_image() runs against one temp tree holding the synthetic bundle and manifest at their default paths, and the installed images are in the dir it ships."""
    from tools import catch_media
    from catch_bundle import make

    monkeypatch.undo()  # drop the autouse unpack stubs; the real catch unpack must run
    monkeypatch.setattr("tools.sample_media.unpack", lambda root: None)
    tree = tmp_path / "tree"
    monkeypatch.setattr("tools.film_frames.unpack", lambda root: None)  # the film frames are not what this test is about
    for sub in ("deploy", "config", "var"):
        (tree / sub).mkdir(parents=True)
    shutil.copy(REPO / "config" / "catch_media.json", tree / "config" / "catch_media.json")
    for name in ("modal_app.py", "modal_deploy.toml"):
        shutil.copy(REPO / "deploy" / name, tree / "deploy" / name)
    shutil.copy(REPO / "pyproject.toml", tree / "pyproject.toml")
    bundle, manifest = make(tmp_path / "private")
    shutil.copy(bundle, tree / "var" / "catch-media.zip")  # the default bundle path
    shutil.copy(manifest, tree / "config" / "catch_media.sha256")  # the default manifest path
    sources = []
    shipped(tree / "deploy" / "modal_app.py", sources)  # runs build_image() with ROOT = tree
    names = catch_media.filenames(catch_media.load())
    static = tree / "static"
    assert static in sources
    assert names and all((static / "ui/media/catch" / n).is_file() for n in names)
    catch_media.check(tree, tree / "config" / "catch_media.sha256")


def test_installed_wheel_resolves_display_images_config(tmp_path):
    import zipfile
    out = tmp_path / "dist"
    build = subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
                           cwd=REPO, capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"hatchling unavailable or build failed offline: {build.stderr.strip()[-200:]}")
    wheel = next(out.glob("*.whl"))
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as z:
        z.extractall(site)  # an installed wheel is exactly its extracted layout
    assert (site / "config" / "display_images.toml").is_file()
    probe = ("import sys; sys.path.insert(0, sys.argv[1]);"
             "from app import display_images as d; import pathlib;"
             "assert pathlib.Path(d.__file__).is_relative_to(sys.argv[1]);"
             "assert d.CONFIG.is_file() and d.CONFIG.is_relative_to(sys.argv[1]);"
             "assert d.cached()['poster_quality']")
    run = subprocess.run([sys.executable, "-I", "-c", probe, str(site)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
