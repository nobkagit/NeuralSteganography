.PHONY: install dev test lint format smoke

install:
python -m pip install -e .

dev:
python -m pip install -e .[dev]

test:
pytest -q

smoke:
./scripts/smoke_test_cli.sh
