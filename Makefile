# Thin targets over the rerunnable synthetic-data DAG (stamps decide what actually reruns).
PYTHON ?= $(shell command -v python3 || command -v python)
DAG_CONFIG ?= configs/dag.yaml
CLI = countonce dag

.PHONY: dvc-yaml dvc-check paper site dag dag-status dag-force recipe validate episodes train evaluate lint test

dag:
	$(CLI) run --config $(DAG_CONFIG)

dag-status:
	$(CLI) status --config $(DAG_CONFIG)

dag-force:
	$(CLI) run --config $(DAG_CONFIG) --force

recipe validate episodes train evaluate:
	$(CLI) run --config $(DAG_CONFIG) --from $@

lint:
	ruff check .

test:
	pytest

# Render paper tables from results/bench.csv; builds experimental/paper/main.pdf only if latexmk is installed.
paper:
	$(PYTHON) scripts/make_paper_tables.py --out experimental/paper/tables
	@if command -v latexmk >/dev/null; then cd experimental/paper && latexmk -pdf -interaction=nonstopmode main.tex; \
	else echo "latexmk not found: tables rendered, PDF skipped"; fi

# Static project page (self-contained, placeholders where results are missing).
site:
	$(PYTHON) scripts/make_site.py --out site/index.html

# Regenerate dvc.yaml/params.yaml from synth/dag.py STAGE_GRAPH; dvc-check fails if they drifted.
dvc-yaml:
	$(PYTHON) scripts/make_dvc_yaml.py

dvc-check:
	$(PYTHON) scripts/make_dvc_yaml.py --check
