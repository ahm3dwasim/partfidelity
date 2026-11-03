"""Dry run of the sample's Spark Open second check: prints what would be sent; calls nothing.

    python tools/sample_readback.py --dry-run

The real run happens inside the deployed web container (app/sample_readback.py) when
config/sample_readback.toml has enabled = true; this tool never calls Baseten.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import sample_readback  # noqa: E402
from app.pipeline import guide_from_json  # noqa: E402
from app.providers import open_readback  # noqa: E402
from app.settings import Settings  # noqa: E402


def dry_run() -> str:
    bundle = ROOT / "app" / Settings.__dataclass_fields__["sample_bundle_dirname"].default
    guide = guide_from_json(json.loads((bundle / "guide.json").read_text(encoding="utf-8")))
    cfg, ocfg = sample_readback.load_config(), open_readback.load_config()
    p = sample_readback.plan(guide.labels, cfg, ocfg)
    lines = [f"  {name}: {phrase}" for name, phrase in p["labels"]]
    return "\n".join([
        f"enabled in config: {cfg.enabled}",
        f"model: {ocfg.model_id}",
        f"would send {p['calls']} call(s) (cap {ocfg.max_calls_per_job}), labels:", *lines,
        f"worst-case cost: ${p['worst_case_usd']} (job cap ${ocfg.max_cost_usd_per_job})",
        "labels that already have a Spark Open row on the live sample are skipped in the app; nothing was sent.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.parse_args()
    print(dry_run())


if __name__ == "__main__":
    main()
