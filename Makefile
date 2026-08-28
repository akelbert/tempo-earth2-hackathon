# Earth-2 Lab image and workshop notebooks.
#
#   make check        fast, no Docker, no GPU - run this before every commit
#   make build        build the attendee image (needs an amd64 GPU host)
#   make run          run it locally with a GPU
#   make stage        stage a regional TEMPO subset from NASA Earthdata
#
# On an Apple Silicon Mac, use the CPU development variant instead:
#
#   make demo-data    synthetic TEMPO scans, no Earthdata login needed
#   make build-cpu    CPU-only image, builds natively on arm64
#   make lab-cpu      open JupyterLab against the demo data
#
# The attendee image cannot be usefully built or run on a Mac: its base is
# ~20 GB of CUDA with no device to execute it, and cross-building it for amd64
# under emulation takes hours. Build that one on a Linux GPU host or with Cloud
# Build (make cloud-build).

SHELL := /bin/bash

REGION       ?= us-central1
PROJECT      ?= $(shell gcloud config get-value project 2>/dev/null)
REPOSITORY   ?= earth2
IMAGE        ?= earth2-lab
RELEASE      ?= dev
REGISTRY     ?= $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPOSITORY)
TAG          ?= $(REGISTRY)/$(IMAGE):$(RELEASE)
LOCAL_TAG    ?= $(IMAGE):$(RELEASE)
PLATFORM     ?= linux/amd64

PYTHON       ?= .venv/bin/python
E2S_VERSION  ?= 0.17.0

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: notebooks
notebooks: ## Render notebooks/src/*.py to notebooks/*.ipynb
	$(PYTHON) scripts/build_notebooks.py

.PHONY: check
check: ## Notebooks current, lint clean, analysis correct (no Docker, no GPU)
	$(PYTHON) scripts/build_notebooks.py --check
	$(PYTHON) -m ruff check src scripts
	$(PYTHON) scripts/smoke_test.py
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/00_environment_check.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/01_tempo_earth2_intro.ipynb

.PHONY: venv
venv: ## Local development environment
	uv venv --python 3.13
	uv pip install --python $(PYTHON) -e . --group staging --group maps --group dev
	uv pip install --python $(PYTHON) nbclient nbformat ipykernel ipywidgets pandas

# ---------------------------------------------------------------------------

.PHONY: build
build: notebooks ## Build the attendee image
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg EARTH2STUDIO_VERSION=$(E2S_VERSION) \
	  -f docker/Dockerfile \
	  -t $(LOCAL_TAG) \
	  .

.PHONY: build-baked
build-baked: notebooks ## Build with the model checkpoint baked into the image
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg EARTH2STUDIO_VERSION=$(E2S_VERSION) \
	  --build-arg PREFETCH_MODEL=true \
	  --build-arg PREFETCH_MODEL_NAME=FCN \
	  -f docker/Dockerfile \
	  -t $(LOCAL_TAG) \
	  .

CPU_TAG ?= $(IMAGE):cpu

.PHONY: build-cpu
build-cpu: notebooks ## Build the CPU development image (works on Apple Silicon)
	docker build \
	  -f docker/Dockerfile.cpu \
	  -t $(CPU_TAG) \
	  .

.PHONY: build-cpu-fast
build-cpu-fast: notebooks ## Same, without Earth2Studio - quick notebook-layout loop
	docker build \
	  -f docker/Dockerfile.cpu \
	  --build-arg WITH_EARTH2STUDIO=false \
	  -t $(CPU_TAG) \
	  .

.PHONY: cloud-build
cloud-build: notebooks ## Build on Cloud Build and push to Artifact Registry
	gcloud builds submit \
	  --config cloudbuild.yaml \
	  --substitutions=_REGION=$(REGION),_REPOSITORY=$(REPOSITORY),_IMAGE=$(IMAGE),_RELEASE=$(RELEASE),_E2S_VERSION=$(E2S_VERSION) \
	  .

.PHONY: push
push: ## Tag and push to Artifact Registry, then print the resolved digest
	docker tag $(LOCAL_TAG) $(TAG)
	docker push $(TAG)
	@echo
	@echo "Record this digest in the release notes; deploy by digest, not by tag:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(TAG)

# ---------------------------------------------------------------------------

TEMPO_URI ?= $(PWD)/data/tempo/northeast.zarr

.PHONY: run
run: ## Run the image locally with a GPU
	docker run --rm -it --gpus all \
	  -p 8888:8888 \
	  -v $(PWD)/data:/home/jovyan/work/data \
	  -e TEMPO_DATA_URI=/home/jovyan/work/data/tempo/northeast.zarr \
	  -e WORKSHOP_RELEASE=$(RELEASE) \
	  $(LOCAL_TAG) \
	  jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=''

.PHONY: run-cpu
run-cpu: ## Run the image locally without a GPU (fallback rehearsal)
	docker run --rm -it \
	  -p 8888:8888 \
	  -v $(PWD)/data:/home/jovyan/work/data \
	  -e TEMPO_DATA_URI=/home/jovyan/work/data/tempo/northeast.zarr \
	  -e WORKSHOP_RELEASE=$(RELEASE) \
	  $(LOCAL_TAG) \
	  jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=''

.PHONY: verify-image
verify-image: ## Run the environment-check notebook inside the image
	docker run --rm --gpus all $(LOCAL_TAG) \
	  bash -lc 'jupyter nbconvert --to notebook --execute \
	    /opt/earth2/notebooks/00_environment_check.ipynb \
	    --output /tmp/checked.ipynb --ExecutePreprocessor.timeout=600 \
	    && echo "environment check passed"'

# --- CPU development variant -------------------------------------------------

DEMO_DATA ?= $(PWD)/data

# A named volume rather than a bind mount: Docker seeds it from the image, so
# the jovyan ownership survives and there is no macOS UID mapping to argue with.
# It also makes "what does a returning attendee see?" testable.
WORK_VOLUME ?= earth2-lab-work

.PHONY: demo-data
demo-data: ## Synthetic TEMPO scans with a known answer (no Earthdata login)
	$(PYTHON) scripts/make_demo_data.py --output $(DEMO_DATA)

# Both of these depend on build-cpu. The image bakes a copy of scripts/ and
# notebooks/, so running them against a stale image tests last week's code -
# which is a confusing way to lose an afternoon. Docker's layer cache makes the
# rebuild a no-op when nothing changed.
.PHONY: lab-cpu
lab-cpu: build-cpu ## JupyterLab in the CPU image, against the demo data
	@test -d "$(DEMO_DATA)/tempo/northeast.zarr" \
	  || { echo "No demo data. Run: make demo-data"; exit 1; }
	@echo
	@echo "  Open http://localhost:8888/lab   (no token)"
	@echo "  Notebooks are in work/notebooks/  - set USE_PRECOMPUTED = True"
	@echo "  Edits persist in the '$(WORK_VOLUME)' volume; Ctrl-C to stop"
	@echo
	docker run --rm -it \
	  -p 8888:8888 \
	  -v "$(DEMO_DATA)":/data:ro \
	  -v $(WORK_VOLUME):/home/jovyan/work \
	  -e TEMPO_DATA_URI=/data/tempo/northeast.zarr \
	  -e WORKSHOP_PRECOMPUTED=/data/precomputed \
	  -e WORKSHOP_RELEASE=$(RELEASE) \
	  $(CPU_TAG)

# Regression guard: the recovery commands must be reachable from a *login*
# shell, because that is what a JupyterLab terminal starts.
.PHONY: verify-shell
verify-shell: build-cpu ## Check the recovery commands resolve in a login shell
	docker run --rm $(CPU_TAG) bash -lc '\
	  set -e; \
	  for cmd in seed-workshop-notebooks restore-workshop-notebooks reset-user-environment; do \
	    path=$$(command -v $$cmd) || { echo "MISSING in login shell: $$cmd"; exit 1; }; \
	    echo "ok  $$cmd -> $$path"; \
	  done; \
	  reset-user-environment >/dev/null; \
	  echo "ok  reset-user-environment dry run"'

.PHONY: lab-cpu-reset
lab-cpu-reset: ## Discard the local work volume (stands in for a fresh attendee)
	docker volume rm $(WORK_VOLUME) 2>/dev/null || true

# verify-cpu needs no mounted data: test_notebook.py builds its own fixtures in
# a temporary workspace, which is the point - it is a self-contained check that
# the image can run the notebooks it ships.
.PHONY: verify-cpu
verify-cpu: build-cpu ## Execute both notebooks inside the CPU image, headless
	docker run --rm $(CPU_TAG) \
	  bash -lc 'set -e; \
	    for nb in 00_environment_check 01_tempo_earth2_intro; do \
	      echo "--- $$nb ---"; \
	      python /opt/earth2/scripts/test_notebook.py \
	        --notebook /opt/earth2/notebooks/$$nb.ipynb --timeout 600; \
	    done'

# ---------------------------------------------------------------------------

STAGE_DATE   ?= 2026-06-15
STAGE_REGION ?= northeast
STAGE_SCANS  ?= 6

.PHONY: stage
stage: ## Stage a regional TEMPO subset (needs an Earthdata login)
	$(PYTHON) scripts/stage_tempo.py \
	  --date $(STAGE_DATE) \
	  --region $(STAGE_REGION) \
	  --scans $(STAGE_SCANS) \
	  --output data/tempo/$(STAGE_REGION).zarr

.PHONY: stage-dry-run
stage-dry-run: ## Show which granules would be staged, download nothing
	$(PYTHON) scripts/stage_tempo.py \
	  --date $(STAGE_DATE) --region $(STAGE_REGION) --scans $(STAGE_SCANS) \
	  --output data/tempo/$(STAGE_REGION).zarr --dry-run

.PHONY: publish-data
publish-data: ## Copy the staged TEMPO subset to the workshop bucket
	@test -n "$(BUCKET)" || { echo "Set BUCKET=gs://..."; exit 1; }
	gcloud storage rsync -r data/tempo $(BUCKET)/tempo

.PHONY: clean
clean: ## Remove build artefacts (never touches data/)
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
