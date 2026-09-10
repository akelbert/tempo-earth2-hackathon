# Earth-2 Lab image and workshop notebooks.
#
#   make check        fast, no Docker, no GPU - run this before every commit
#   make build        build the attendee image (needs an amd64 GPU host)
#   make run          run it locally with a GPU
#   make stage        stage a regional TEMPO subset from NASA Earthdata
#
# The attendee image cannot be usefully built or run on a Mac: its base is
# ~20 GB of CUDA with no device to execute it, and cross-building it for amd64
# under emulation takes hours. Build that one on a Linux GPU host or with Cloud
# Build (make cloud-build).

SHELL := /bin/bash

REGION       ?= us-central1
PROJECT      ?= $(shell gcloud config get-value project 2>/dev/null)
# Keep Terraform lookup lazy: local-only targets must not wait on remote state.
TF_REPOSITORY = $(shell terraform -chdir=terraform output -raw artifact_registry_repository_id 2>/dev/null)
REPOSITORY   ?= $(if $(TF_REPOSITORY),$(TF_REPOSITORY),tempo-earth2-dev)
IMAGE        ?= earth2-lab
RELEASE      ?= dev
CANDIDATE_RELEASE ?= model-candidate-$(RELEASE)
CANDIDATE_IMAGE_DIGEST ?=
REGISTRY     ?= $(REGION)-docker.pkg.dev/$(PROJECT)/$(REPOSITORY)
TAG          ?= $(REGISTRY)/$(IMAGE):$(RELEASE)
LOCAL_TAG    ?= $(IMAGE):$(RELEASE)
PLATFORM     ?= linux/amd64

HUB_IMAGE     ?= earth2-hub
HUB_TAG       ?= $(REGISTRY)/$(HUB_IMAGE):$(RELEASE)
HUB_LOCAL_TAG ?= $(HUB_IMAGE):$(RELEASE)

CHART_VERSION ?= 4.4.1
HELM_VALUES   ?= jupyterhub/.generated/values-event.yaml
BUILD_SERVICE_ACCOUNT ?= $(shell terraform -chdir=terraform output -raw cloud_build_service_account 2>/dev/null)
BUILD_SOURCE_BUCKET    ?= $(shell terraform -chdir=terraform output -raw cloud_build_source_bucket 2>/dev/null)

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
	$(PYTHON) -m ruff check src scripts jupyterhub/render_values.py jupyterhub/render_benchmark_job.py
	$(PYTHON) scripts/smoke_test.py
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/00_environment_check.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/01_data_and_model_catalog.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/01_tempo_earth2_intro.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/02_tempo_column_vs_surface.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/03_smoke_event.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/04_wetland_response.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/05_scale_matters.ipynb
	$(PYTHON) scripts/test_notebook.py --notebook notebooks/06_bring_your_own_site.ipynb

.PHONY: venv
venv: ## Local development environment
	uv venv --python 3.12
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

.PHONY: build-candidate
build-candidate: notebooks ## Build a non-deployable image for FCN/DLWP/precipitation validation
	docker build \
	  --platform $(PLATFORM) \
	  --build-arg EARTH2STUDIO_VERSION=$(E2S_VERSION) \
	  --build-arg EARTH2STUDIO_EXTRAS=fcn,dlwp,precip-afno,data,utils \
	  --build-arg 'EARTH2STUDIO_IMPORT_CHECK=from earth2studio.models.px import FCN, DLWP; from earth2studio.models.dx import PrecipitationAFNO' \
	  -f docker/Dockerfile \
	  -t $(LOCAL_TAG)-candidate \
	  .

.PHONY: benchmark-candidates
benchmark-candidates: ## Benchmark candidate prognostics on the current GPU
	mkdir -p $(PWD)/benchmark-results/model-cache
	docker run --rm --gpus all \
	  -v $(PWD)/benchmark-results:/results \
	  -v $(PWD)/benchmark-results/model-cache:/opt/earth2/cache/models \
	  -e EARTH2STUDIO_MODEL_CACHE=/opt/earth2/cache/models \
	  $(LOCAL_TAG)-candidate \
	  python /opt/earth2/scripts/benchmark_models.py \
	    --model FCN DLWP --init-time 2026-05-31T12:00 \
	    --output /results/model-benchmark.json
	docker run --rm --gpus all \
	  -v $(PWD)/benchmark-results:/results \
	  -v $(PWD)/benchmark-results/model-cache:/opt/earth2/cache/models \
	  -e EARTH2STUDIO_MODEL_CACHE=/opt/earth2/cache/models \
	  $(LOCAL_TAG)-candidate \
	  python /opt/earth2/scripts/benchmark_precipitation.py \
	    --init-time 2026-05-31T12:00 \
	    --output /results/precipitation-benchmark.json

.PHONY: render-l4-benchmark
render-l4-benchmark: ## Render (but do not submit) a digest-pinned GKE benchmark Job
	@test -n "$(CANDIDATE_IMAGE_DIGEST)" || { echo "Set CANDIDATE_IMAGE_DIGEST to image@sha256:..."; exit 1; }
	$(PYTHON) jupyterhub/render_benchmark_job.py \
	  --image "$(CANDIDATE_IMAGE_DIGEST)"

.PHONY: cloud-build
cloud-build: notebooks ## Build on Cloud Build and push to Artifact Registry
	@test -n "$(PROJECT)" || { echo "Set PROJECT to the target Google Cloud project"; exit 1; }
	@test -n "$(BUILD_SERVICE_ACCOUNT)" || { echo "Set BUILD_SERVICE_ACCOUNT (or apply Terraform first)"; exit 1; }
	@test -n "$(BUILD_SOURCE_BUCKET)" || { echo "Set BUILD_SOURCE_BUCKET (or apply Terraform first)"; exit 1; }
	gcloud builds submit \
	  --project=$(PROJECT) \
	  --config cloudbuild.yaml \
	  --service-account=$(BUILD_SERVICE_ACCOUNT) \
	  --gcs-source-staging-dir=gs://$(BUILD_SOURCE_BUCKET)/source \
	  --substitutions=_REGION=$(REGION),_REPOSITORY=$(REPOSITORY),_IMAGE=$(IMAGE),_HUB_IMAGE=$(HUB_IMAGE),_RELEASE=$(RELEASE),_E2S_VERSION=$(E2S_VERSION),_PREFETCH_MODEL=true \
	  .

.PHONY: cloud-build-candidate
cloud-build-candidate: notebooks ## Build/push candidate extras without changing JupyterHub
	@test -n "$(PROJECT)" || { echo "Set PROJECT to the target Google Cloud project"; exit 1; }
	@test -n "$(BUILD_SERVICE_ACCOUNT)" || { echo "Set BUILD_SERVICE_ACCOUNT (or apply Terraform first)"; exit 1; }
	@test -n "$(BUILD_SOURCE_BUCKET)" || { echo "Set BUILD_SOURCE_BUCKET (or apply Terraform first)"; exit 1; }
	gcloud builds submit \
	  --project=$(PROJECT) \
	  --config cloudbuild-model-candidate.yaml \
	  --service-account=$(BUILD_SERVICE_ACCOUNT) \
	  --gcs-source-staging-dir=gs://$(BUILD_SOURCE_BUCKET)/source \
	  --substitutions=_REGION=$(REGION),_REPOSITORY=$(REPOSITORY),_IMAGE=$(IMAGE),_CANDIDATE_RELEASE=$(CANDIDATE_RELEASE),_E2S_VERSION=$(E2S_VERSION) \
	  .

.PHONY: push
push: ## Tag and push to Artifact Registry, then print the resolved digest
	docker tag $(LOCAL_TAG) $(TAG)
	docker push $(TAG)
	@echo
	@echo "Record this digest in the release notes; deploy by digest, not by tag:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(TAG)

.PHONY: build-hub
build-hub: ## Build the custom Hub image (adds jupyterhub-nativeauthenticator)
	docker build \
	  --platform $(PLATFORM) \
	  -f docker/Dockerfile.hub \
	  -t $(HUB_LOCAL_TAG) \
	  .

.PHONY: push-hub
push-hub: ## Tag and push the Hub image to Artifact Registry
	docker tag $(HUB_LOCAL_TAG) $(HUB_TAG)
	docker push $(HUB_TAG)
	@echo
	@echo "Set the resolved reference as HUB_IMAGE_REFERENCE before rendering Helm values:"
	@docker inspect --format='{{index .RepoDigests 0}}' $(HUB_TAG)

.PHONY: render-helm
render-helm: ## Render validated, digest-pinned JupyterHub values from environment variables
	$(PYTHON) jupyterhub/render_values.py --output $(HELM_VALUES)

.PHONY: helm-template
helm-template: render-helm ## Render the pinned JupyterHub chart locally
	helm template jupyterhub jupyterhub/jupyterhub \
	  --version $(CHART_VERSION) --namespace jupyterhub -f $(HELM_VALUES) >/dev/null

.PHONY: deploy-hub
deploy-hub: helm-template ## Install/upgrade the pinned JupyterHub chart in the current kubectl context
	helm upgrade --install jupyterhub jupyterhub/jupyterhub \
	  --version $(CHART_VERSION) --namespace jupyterhub \
	  -f $(HELM_VALUES) --rollback-on-failure --timeout 20m

# ---------------------------------------------------------------------------

TEMPO_URI ?= $(PWD)/data/tempo/northeast.zarr

.PHONY: run
run: ## Run the image locally with a GPU
	docker run --rm -it --gpus all \
	  -p 127.0.0.1:8888:8888 \
	  -v $(PWD)/data:/home/jovyan/work/data \
	  -e WORKSHOP_DATA_URI=/home/jovyan/work/data \
	  -e TEMPO_DATA_URI=/home/jovyan/work/data/tempo/northeast.zarr \
	  -e WORKSHOP_CONTEXT_DATA_URI=/home/jovyan/work/data/context \
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

# --- Local fixtures ----------------------------------------------------------

DEMO_DATA ?= $(PWD)/data

.PHONY: demo-data
demo-data: ## Synthetic TEMPO scans with a known answer (no Earthdata login)
	$(PYTHON) scripts/make_demo_data.py --output $(DEMO_DATA)
	$(MAKE) teaching-data TEMPO_URI=$(DEMO_DATA)/tempo/northeast.zarr CONTEXT_DATA=$(DEMO_DATA)/context

CONTEXT_DATA ?= $(PWD)/data/context

.PHONY: teaching-data
teaching-data: ## Synthetic context fixtures aligned to the staged TEMPO case
	$(PYTHON) scripts/make_teaching_data.py \
	  --tempo $(TEMPO_URI) \
	  --output $(CONTEXT_DATA)

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

.PHONY: stage-aqs
stage-aqs: ## Stage EPA surface NO2/O3/PM2.5 for the TEMPO case date
	$(PYTHON) scripts/stage_aqs.py --date $(STAGE_DATE) \
	  --region $(STAGE_REGION) --output data/context/aqs/$(STAGE_DATE).csv

AQS_START ?= 2025-09-01
AQS_END   ?= 2026-06-01

.PHONY: stage-aqs-explore
stage-aqs-explore: ## Stage a broader, compressed EPA AQS regional collection
	$(PYTHON) scripts/stage_aqs.py \
	  --start-date $(AQS_START) --end-date $(AQS_END) \
	  --region $(STAGE_REGION) \
	  --partition-by-month --output data/context/aqs/by-month

.PHONY: publish-data
publish-data: ## Copy staged TEMPO and context data to the workshop bucket
	@test -n "$(BUCKET)" || { echo "Set BUCKET=gs://..."; exit 1; }
	gcloud storage rsync -r data/tempo $(BUCKET)/tempo
	gcloud storage rsync -r data/context $(BUCKET)/context

.PHONY: clean
clean: ## Remove build artefacts (never touches data/)
	rm -rf .pytest_cache .ruff_cache src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
