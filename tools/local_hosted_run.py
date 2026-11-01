"""Local dry run of the Modal boot path: same app object, temp dirs, uvicorn.

Runs ``build_hosted_app`` (restore, reconcile, snapshot loop, sweep loop)
against a scratch directory laid out like the Volume and serves it with
uvicorn. Docker is not needed. No Modal or Meta call is made.

    python3 tools/local_hosted_run.py --root /tmp/pf-dry --port 8765
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.deploy_config import HostConfig, load_deploy_config  # noqa: E402
from app.hosting import build_hosted_app  # noqa: E402


def local_host(root: Path) -> HostConfig:
    """The deploy file's intervals, with every path moved under ``root``."""
    host = load_deploy_config(ROOT / "deploy" / "modal_deploy.toml").host
    return HostConfig(
        data_root=root / "vol" / "data",
        snapshot_dir=root / "vol" / "snapshots",
        local_db=root / "local" / "partfidelity.sqlite3",
        snapshot_interval_s=host.snapshot_interval_s,
        snapshot_settle_s=host.snapshot_settle_s,
        sweep_interval_s=host.sweep_interval_s,
        snapshot_max_age_s=host.snapshot_max_age_s,
    )


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # A dry run never reaches a provider: replay clients, no key needed.
    os.environ.setdefault("PARTFIDELITY_PIPELINE_MODE", "replay")
    app = build_hosted_app(local_host(args.root.resolve()), commit=None)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
