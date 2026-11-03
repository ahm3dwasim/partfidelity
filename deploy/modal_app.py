"""Modal entry for PartFidelity: one container, a Volume, a named Secret.

Deploy (from the repo root, by the maintainer after release approval):

    modal deploy deploy/modal_app.py

The named Secret holds META_API_KEY, META_FILES_BASE_URL and
BASETEN_API_KEY (names only here; values stay in Modal). Verify after
deploy: a finished job's /calls page shows the Spark Open second check as
agrees or disagrees, not "not available" for every part (the latter means
BASETEN_API_KEY is missing from the Secret).

min_containers=max_containers=1 keeps exactly one SQLite writer. The live
database sits on container-local disk; the Volume holds job files and the
periodic ``db.snapshot`` (see app/persistence.py). Importing this module
makes no account call; only ``modal deploy`` does.
"""

from __future__ import annotations

import logging
import os
import sys
import tomllib
from pathlib import Path

import modal

LOCAL = modal.is_local()
ROOT_ENV = "PARTFIDELITY_ROOT"
if LOCAL:
    ROOT = Path(__file__).resolve().parents[1]
else:
    root = os.environ.get(ROOT_ENV)
    if not root:
        raise RuntimeError(f"{ROOT_ENV} must be set in the Modal container")
    ROOT = Path(root)
sys.path.insert(0, str(ROOT))

from app.deploy_config import load_deploy_config  # noqa: E402

cfg = load_deploy_config(ROOT / "deploy" / "modal_deploy.toml")
modal_cfg = cfg.modal


def build_image() -> modal.Image:
    if LOCAL:
        from tools.sample_media import unpack as unpack_sample_media

        unpack_sample_media(ROOT)
        from tools.catch_media import unpack as unpack_catch_media

        unpack_catch_media(ROOT)
    image = modal.Image.debian_slim(python_version=modal_cfg.python_version)
    if not LOCAL:  # the container already holds the built image
        return image
    with open(ROOT / "pyproject.toml", "rb") as handle:
        deps = tomllib.load(handle)["project"]["dependencies"]
    # ffmpeg ships ffprobe; the upload path shells out to both.
    image = image.apt_install("ffmpeg").pip_install(*deps)
    image = image.env({**dict(modal_cfg.app_env), "PARTFIDELITY_PRODUCTION": "1", ROOT_ENV: str(modal_cfg.remote_root)})
    from tools.film_frames import unpack  # the deploy machine unpacks the frame archives into static/ first

    unpack(ROOT)
    from app.how_media import MediaError, check as check_how_media  # private frames are provisioned, never in git

    try:
        check_how_media(ROOT)
    except MediaError as error:
        raise SystemExit(f"deploy refused: {error}") from None
    for folder in modal_cfg.image_folders:
        image = image.add_local_dir(ROOT / folder, str(modal_cfg.remote_root / folder))
    for name in modal_cfg.image_files:
        image = image.add_local_file(ROOT / name, str(modal_cfg.remote_root / name))
    return image


app = modal.App(modal_cfg.app_name)
volume = modal.Volume.from_name(modal_cfg.volume_name, create_if_missing=True)


@app.function(
    image=build_image(),
    volumes={str(modal_cfg.volume_mount): volume},
    secrets=[modal.Secret.from_name(modal_cfg.secret_name)],
    min_containers=1,
    max_containers=1,
    scaledown_window=modal_cfg.scaledown_window_s,
    timeout=modal_cfg.timeout_s,
)
@modal.concurrent(max_inputs=modal_cfg.max_inputs)
@modal.asgi_app()
def web():
    logging.basicConfig(level=logging.INFO)
    from app.hosting import build_hosted_app

    return build_hosted_app(cfg.host, commit=volume.commit)
