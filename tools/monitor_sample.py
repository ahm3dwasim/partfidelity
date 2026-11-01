"""Cached-sample monitor: fetch the deployed sample page, alert on two misses.

Run it every 15 minutes from any scheduler (cron, a GitHub Actions
schedule, Modal's own cron). It reads only the cached replay sample and
/healthz, so it makes no Meta call and needs no secret.

Exit codes: 0 healthy, 1 first miss (state recorded, no page), 2 ALERT
(consecutive misses reached ``--alert-after``, page now). The miss count
lives in ``--state-file`` and resets on the next healthy run.

    python3 tools/monitor_sample.py --base-url https://<app>.modal.run \
        --state-file ~/.cache/partfidelity-monitor.json
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

OK, MISS, ALERT = 0, 1, 2


def fetch(url: str, timeout_s: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return response.status, response.read(2_000_000).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, ""
    except (urllib.error.URLError, OSError, TimeoutError):
        return 0, ""


def probe(base_url: str, sample_path: str, expect: str, timeout_s: float) -> str | None:
    """Return None when healthy, else a short reason."""
    base = base_url.rstrip("/")
    status, _ = fetch(f"{base}/healthz", timeout_s)
    if status != 200:
        return f"healthz answered {status}"
    status, body = fetch(f"{base}{sample_path}", timeout_s)
    if status != 200:
        return f"{sample_path} answered {status}"
    if expect not in body:
        return f"{sample_path} is missing {expect!r}"
    return None


def next_state(previous: int, reason: str | None, alert_after: int) -> tuple[int, int]:
    """Return (consecutive misses, exit code)."""
    if reason is None:
        return 0, OK
    misses = previous + 1
    return misses, ALERT if misses >= alert_after else MISS


def _read_misses(path: Path) -> int:
    try:
        return int(json.loads(path.read_text())["misses"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--sample-path", default="/reader/sample")
    parser.add_argument("--expect", default="Sample guide")
    parser.add_argument("--alert-after", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    reason = probe(args.base_url, args.sample_path, args.expect, args.timeout)
    misses, code = next_state(_read_misses(args.state_file), reason, args.alert_after)
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.state_file.write_text(json.dumps({"misses": misses}))
    label = {OK: "OK", MISS: "MISS", ALERT: "ALERT"}[code]
    print(f"{label} misses={misses}" + (f" reason={reason}" if reason else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
