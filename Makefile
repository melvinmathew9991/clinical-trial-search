# Medical Embeddings Search — task runner
#
# Tuned for the target machine: 4 logical cores, 7.89 GB RAM.
# Every target that touches the pipeline pins BLAS threads to 1 so only
# gensim's `workers` are parallel (Rules.md section 2, ADR-008).
#
# Windows: run these under Git Bash. If `make` is unavailable, each recipe
# below is a plain command you can paste directly.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PY      ?= python
VENV    ?= .venv
BIN     := $(VENV)/bin
ifeq ($(OS),Windows_NT)
BIN     := $(VENV)/Scripts
endif

# Thread pinning — prevents numpy/BLAS from oversubscribing 4 cores
export OMP_NUM_THREADS      := 1
export MKL_NUM_THREADS      := 1
export OPENBLAS_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS  := 1

# Dev profile: small sample so iteration stays fast on the laptop
LIMIT ?=

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
.PHONY: setup
setup:  ## Create venv and install the package with dev + app extras
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev,app]"
	$(BIN)/python -m medsearch.runtime --download-nltk
	@echo "Done. Activate with: source $(BIN)/activate"

.PHONY: hooks
hooks:  ## Install pre-commit hooks
	$(BIN)/pre-commit install

# ---------------------------------------------------------------- preflight
.PHONY: doctor
doctor:  ## Preflight: cores, free RAM, disk, NLTK, artefact sizes
	$(BIN)/medsearch doctor

.PHONY: doctor-full
doctor-full:  ## Preflight plus artefact integrity checks
	$(BIN)/medsearch doctor --full

# ---------------------------------------------------------------- data
.PHONY: data
data:  ## Migrate the legacy corpus into data/raw/ (dry-run first)
	$(BIN)/python scripts/migrate_legacy.py --dry-run
	@echo ""
	@echo "Review the plan above, then run: make data-apply"

.PHONY: data-apply
data-apply:  ## Actually perform the legacy migration
	$(BIN)/python scripts/migrate_legacy.py --apply

.PHONY: preprocess
preprocess:  ## Preprocess the corpus and cache tokens
	$(BIN)/medsearch preprocess $(if $(LIMIT),--limit $(LIMIT),)

# ---------------------------------------------------------------- train
.PHONY: train
train: doctor  ## Train both models and build both indexes (runs doctor first)
	$(BIN)/medsearch train --model all $(if $(LIMIT),--limit $(LIMIT),)
	$(BIN)/medsearch index build --model all $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: train-dev
train-dev:  ## Fast 2000-row training profile for iteration (~90s, low RAM)
	$(MAKE) train LIMIT=2000

# ---------------------------------------------------------------- run
.PHONY: app
app:  ## Launch the Streamlit UI on :8501
	$(BIN)/streamlit run src/medsearch/app/streamlit_app.py

.PHONY: search
search:  ## Ad-hoc search: make search Q="lung failure"
	$(BIN)/medsearch search "$(Q)"

.PHONY: evaluate
evaluate:  ## Run the evaluation suite and write reports/evaluation.json
	$(BIN)/medsearch evaluate

# ---------------------------------------------------------------- quality
.PHONY: lint
lint:  ## ruff check + format check
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

.PHONY: fmt
fmt:  ## Auto-format
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

.PHONY: type
type:  ## mypy --strict
	$(BIN)/mypy

.PHONY: layers
layers:  ## Enforce the Architecture.md layer contract
	$(BIN)/lint-imports

.PHONY: test
test:  ## Run the fast test suite with coverage
	$(BIN)/pytest

.PHONY: test-all
test-all:  ## Full suite including integration, with the 80% coverage gate
	$(BIN)/pytest -m "" --cov-fail-under=80

.PHONY: check
check: lint type layers test-all  ## Everything CI runs, locally

# ---------------------------------------------------------------- housekeeping
.PHONY: clean
clean:  ## Remove caches and build artefacts (keeps data/ and models/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean-artefacts
clean-artefacts:  ## Delete trained models and indexes (frees ~1 GB)
	rm -rf models/* data/interim/* data/processed/* reports/*
	@echo "Artefacts removed. Rebuild with: make train"
