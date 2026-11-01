# bash for the [[ ]] cleanup guard in clean-clone-check.
SHELL := /bin/bash

# What `make test` runs; narrow it for a quick check, e.g. make test TESTS=tests/test_built.py
TESTS ?= tests

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: film-frames verify verify-public test browser clean-clone-check

browser:
	PY="$(PY)" ./tools/ensure_test_browser.sh

film-frames:
	$(PY) tools/film_frames.py unpack

test: film-frames
	$(PY) -m pytest -q $(TESTS)

verify: browser test

verify-public: verify

clean-clone-check:
	HEAD_SHA=$$(git rev-parse HEAD) || exit 1; \
	TMP=$$(mktemp -d) || exit 1; \
	printf '%s\n' "$$TMP" > "$$TMP/.own-tmp" || exit 1; \
	trap '[[ -n $$TMP && -d $$TMP && ! -L $$TMP && -f $$TMP/.own-tmp && $$(cat -- "$$TMP/.own-tmp") == "$$TMP" ]] && rm -rf -- "$$TMP"' EXIT; \
	git clone --quiet . "$$TMP/clone" || exit 1; \
	git -C "$$TMP/clone" checkout --quiet "$$HEAD_SHA" || exit 1; \
	cd "$$TMP/clone" || exit 1; \
	uv venv --python 3.12 .venv && \
	uv pip install --python .venv/bin/python -e '.[test]' && \
	make verify
