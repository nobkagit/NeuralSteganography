.PHONY: init doctor test lint type smoke clean _ensure_venv

VENV ?= .venv
PYTHON := $(VENV)/bin/python
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

init:
	bash scripts/setup_env.sh

doctor:
	bash scripts/doctor.sh

_ensure_venv:
	@if [ ! -x "$(PYTHON)" ]; then \
		echo "لطفاً ابتدا 'make init' را اجرا کنید تا محیط مجازی ساخته شود." >&2; \
		exit 1; \
	fi

test: _ensure_venv
	$(PYTHON) -m pytest -q

lint: _ensure_venv
	$(RUFF) check .

type: _ensure_venv
	$(MYPY) src

smoke: _ensure_venv
	bash scripts/smoke_test_cli.sh

clean:
	rm -rf $(VENV) __pycache__ */__pycache__ .mypy_cache .pytest_cache .ruff_cache
	rm -f tmp_stego.txt
