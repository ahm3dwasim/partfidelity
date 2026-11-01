#!/bin/bash
# reader-module: provision the Playwright Chromium binary for tests/test_reader_browser.py.
# Idempotent and cached: `playwright install` skips browsers already present.
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
if [ -z "${PY:-}" ]; then
  PY="${ROOT}/.venv/bin/python"
  if [ ! -x "${PY}" ]; then
    PY="python3"
  fi
fi
if ! "${PY}" -c "import playwright" 2>/dev/null; then
  echo "ensure_test_browser: playwright is not installed; run \`pip install -e '.[test]'\` first" >&2
  exit 1
fi
if ! "${PY}" -m playwright install chromium; then
  echo "ensure_test_browser: chromium install failed; re-run \`make browser\` (needs network)" >&2
  exit 1
fi
