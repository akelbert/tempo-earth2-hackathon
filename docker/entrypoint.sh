#!/usr/bin/env bash
# Earth-2 Lab container entrypoint.
#
# Runs before jupyterhub-singleuser on every start. It must be fast, idempotent,
# and must never destroy attendee work: a spawn that hangs here is indis-
# tinguishable, to an attendee, from the whole event being broken.
set -euo pipefail

log() { printf '[earth2-lab] %s\n' "$*" >&2; }

: "${WORKSHOP_WORK_DIR:=${HOME}/work}"
: "${WORKSHOP_REFERENCE_DIR:=/opt/earth2/notebooks}"
: "${EARTH2STUDIO_DATA_CACHE:=${HOME}/.cache/earth2studio/data}"

mkdir -p "${WORKSHOP_WORK_DIR}" "${EARTH2STUDIO_DATA_CACHE}" "${HOME}/.local/bin"

# Seed the working copy of the notebooks on first start only. Later starts must
# not touch files the attendee has edited.
if [ -d "${WORKSHOP_REFERENCE_DIR}" ]; then
    seed-workshop-notebooks || log "notebook seeding failed; continuing to start the server"
fi

# A user-level pip install (the documented experimentation path) puts scripts in
# ~/.local/bin. Without this they are installed but not runnable, which reads as
# a broken environment.
case ":${PATH}:" in
    *":${HOME}/.local/bin:"*) ;;
    *) export PATH="${HOME}/.local/bin:${PATH}" ;;
esac

# Report the environment into the pod log. On event day this is the first thing
# support looks at, and it costs a fraction of a second.
python - <<'PY' || log "environment probe failed (non-fatal)"
import json
try:
    from tempo_earth2.config import describe_environment
    info = describe_environment()
    keys = ("python", "torch", "cuda_available", "gpu_name", "gpu_memory_gb",
            "compute_capability", "earth2studio", "glue_jupyter",
            "workshop_data_uri", "tempo_data_uri", "context_data_uri", "model_cache",
            "model_cache_free_gb")
    print("[earth2-lab] " + json.dumps({k: info.get(k) for k in keys}))
except Exception as exc:
    print(f"[earth2-lab] environment probe error: {exc}")
PY

exec "$@"
