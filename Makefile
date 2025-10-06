.PHONY: install setup doctor smoke clean

VENV?=.venv
PYTHON=$(VENV)/bin/python
PIP=$(PYTHON) -m pip

install: $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

$(PYTHON):
	python3 -m venv $(VENV)

setup:
	bash scripts/setup_env.sh

doctor: $(PYTHON)
	$(PYTHON) -m neuralstego doctor

smoke: doctor
	bash scripts/smoke_test_cli.sh

clean:
	rm -rf $(VENV)
	rm -f tmp_stego.txt
	rm -rf models
	rm -rf data
	rm -f .env
