.PHONY: help install lint test smoke clean prepare b0 train eval eval-b0 all

help:
	@echo "Targets:"
	@echo "  install    editable install + dev extras"
	@echo "  lint       ruff + mypy"
	@echo "  test       unit tests only"
	@echo "  smoke      smoke tests (pulls tiny models)"
	@echo "  prepare    run 00_download_models + 01_prepare_efcamdat + 02_prepare_external"
	@echo "  b0         materialise B0 (stock GPT-2) checkpoint — no training"
	@echo "  eval-b0    run PPL + cloze against B0 only"
	@echo "  train      train B1 + B2 + G1 (requires GPU)"
	@echo "  eval       run PPL + cloze on all runs/*/last/"
	@echo "  all        prepare -> b0 -> train -> eval -> aggregate"
	@echo "  clean      remove run outputs and caches"

install:
	pip install -e ".[dev,viz]"

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts
	mypy src

test:
	pytest -m "not smoke and not slow"

smoke:
	pytest -m smoke

prepare:
	python scripts/00_download_models.py
	python scripts/01_prepare_efcamdat.py
	python scripts/02_prepare_external.py

b0:
	python scripts/03_prepare_b0_checkpoint.py --config configs/b0_native.yaml

eval-b0: b0
	python scripts/20_eval_ppl.py --checkpoint runs/b0_native/last --split in_domain
	python scripts/20_eval_ppl.py --checkpoint runs/b0_native/last --split transfer
	python scripts/21_eval_cloze.py --checkpoint runs/b0_native/last
	python scripts/30_aggregate_results.py

train:
	python scripts/10_train_b1_learner.py --config configs/b1_learner.yaml
	python scripts/11_train_b2_metadata_concat.py --config configs/b2_metadata_concat.yaml
	python scripts/12_train_g1_gated.py --config configs/g1_gated.yaml

eval:
	@for run in runs/*/last; do \
		python scripts/20_eval_ppl.py --checkpoint $$run --split in_domain ; \
		python scripts/20_eval_ppl.py --checkpoint $$run --split transfer ; \
		python scripts/21_eval_cloze.py --checkpoint $$run ; \
	done
	python scripts/30_aggregate_results.py

all: prepare b0 train eval

clean:
	rm -rf runs/ data-cache/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
