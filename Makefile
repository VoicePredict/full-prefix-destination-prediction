.PHONY: setup verify results tables figures environment data-check plan test full reproduce

PYTHON ?= python3
VENV_PYTHON ?= .venv/bin/python
DATA_ROOT ?= data/Geolife
OUTPUT_ROOT ?= work/runs

verify:
	$(PYTHON) scripts/reproduce.py verify

setup:
	$(PYTHON) scripts/reproduce.py setup

results:
	$(VENV_PYTHON) scripts/reproduce.py results

tables:
	$(VENV_PYTHON) scripts/export_paper_tables.py

figures:
	$(VENV_PYTHON) scripts/generate_figures.py

environment:
	python3.10 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

data-check:
	$(PYTHON) scripts/verify_geolife.py $(DATA_ROOT)

plan:
	$(PYTHON) scripts/reproduce.py full --data-root $(DATA_ROOT) --output-root $(OUTPUT_ROOT) --dry-run

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v

full:
	$(VENV_PYTHON) scripts/reproduce.py full --data-root $(DATA_ROOT) --output-root $(OUTPUT_ROOT)

reproduce:
	$(PYTHON) scripts/reproduce.py setup
	$(MAKE) environment
	$(MAKE) test
	$(MAKE) full
	$(MAKE) results
	$(MAKE) verify
