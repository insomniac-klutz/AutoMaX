UV ?= uv
RUN = $(UV) run --extra dev

.PHONY: help sync lint fmt typecheck test check stat e2e-small schemas contract redteam plugin-validate portability

help:
	@echo "make check      ruff + mypy + unit + property tests"
	@echo "make stat       statistical tests incl. T1-synth (slow, seeded)"
	@echo "make e2e-small  offline end-to-end run on a generated toy dataset"
	@echo "make schemas    regenerate docs/schemas/*.json"

sync:
	$(UV) sync --extra dev

lint:
	$(RUN) ruff check src tests datasets
	$(RUN) ruff format --check src tests datasets

fmt:
	$(RUN) ruff format src tests datasets
	$(RUN) ruff check --fix src tests datasets

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest tests/unit tests/property -q

check: lint typecheck test

stat:
	$(RUN) pytest tests/statistical -q -m "slow or not slow"

e2e-small:
	$(RUN) pytest tests/e2e -q -m "not network"

schemas:
	$(RUN) python -m amx.spec.schema --write

contract redteam plugin-validate portability:
	@echo "'make $@' arrives in a later milestone (see docs/ROLLER.md)"; exit 1
