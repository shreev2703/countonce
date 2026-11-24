# Thin targets over the rerunnable synthetic-data DAG (stamps decide what actually reruns).
DAG_CONFIG ?= configs/dag.yaml
CLI = countonce dag

.PHONY: dag dag-status dag-force recipe validate episodes train evaluate lint test

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
