.PHONY: setup verify results tables figures environment data-check plan test full reproduce

PYTHON ?= python3
VENV_PYTHON ?= .venv/bin/python
DATA_ROOT ?= data/Geolife
OUTPUT_ROOT ?= work/runs
DATA_CHECK ?= full

verify:
	$(PYTHON) scripts/reproduce.py verify

setup:
	$(PYTHON) scripts/reproduce.py setup --data-root "$(DATA_ROOT)"

results:
	$(VENV_PYTHON) scripts/reproduce.py results

tables:
	$(VENV_PYTHON) scripts/export_paper_tables.py

figures:
	$(VENV_PYTHON) scripts/generate_figures.py

environment:
	$(PYTHON) -c 'import sys; version = sys.version_info[:2]; sys.exit("Python 3.10 is required; set PYTHON to its executable (found %d.%d)." % version) if version != (3, 10) else None'
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -c constraints.txt -r requirements.txt
	.venv/bin/python -m pip install --no-deps -e .

data-check:
	$(PYTHON) scripts/verify_geolife.py "$(DATA_ROOT)"

plan:
	$(PYTHON) scripts/reproduce.py full --data-root "$(DATA_ROOT)" --output-root "$(OUTPUT_ROOT)" --dry-run

test:
	$(VENV_PYTHON) -m unittest discover -s tests -v

full:
	$(VENV_PYTHON) scripts/reproduce.py full --data-root "$(DATA_ROOT)" --output-root "$(OUTPUT_ROOT)" --data-check "$(DATA_CHECK)"

reproduce:
	$(MAKE) verify
	$(MAKE) environment
	$(PYTHON) scripts/reproduce.py setup --data-root "$(DATA_ROOT)"
	$(MAKE) test
	$(MAKE) full DATA_CHECK=paths
	$(MAKE) verify
