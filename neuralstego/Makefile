.PHONY: install dev test smoke

PROJECT_ROOT := $(abspath $(CURDIR)/..)

install:
cd $(PROJECT_ROOT) && python -m pip install -e .

dev:
cd $(PROJECT_ROOT) && python -m pip install -e .[dev]

test:
cd $(PROJECT_ROOT) && pytest -q

smoke:
cd $(PROJECT_ROOT) && ./neuralstego/scripts/smoke_test_cli.sh
