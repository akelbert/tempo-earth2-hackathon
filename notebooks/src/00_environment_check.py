# %% [markdown]
# # Earth-2 Lab: environment check
#
# Run this first, before anything else, on the first morning.
#
# It answers five questions, and it is designed so that you can paste its
# output straight into a support request if any of them come back wrong:
#
# 1. Which software versions am I actually running?
# 2. Do I have a GPU, and how much memory does it have?
# 3. Can I reach the shared TEMPO data?
# 4. Where do model checkpoints get cached, and is that cache already warm?
# 5. Is my home directory writable, and how much space is left?
#
# Nothing here needs a GPU. If the GPU check fails, the remaining diagnostics
# still run so the output can be attached to the platform incident.

# %%
from tempo_earth2.config import WorkshopConfig, describe_environment

config = WorkshopConfig.from_env()
info = describe_environment()

width = max(len(k) for k in info)
for key, value in info.items():
    print(f"{key:<{width}}  {value}")

# %% [markdown]
# ## 1. Accelerator
#
# The workshop image is built for a single dedicated GPU per attendee. If
# `cuda_available` is `False`, report it to an instructor: the supported
# environment is unhealthy and Earth-2 inference should not be attempted.

# %%
# Every check in this notebook reports rather than raises. A diagnostic that
# dies with a traceback on the thing it was written to diagnose is worse than
# no diagnostic at all.
try:
    import torch
except ImportError as exc:
    torch = None
    print(f"PyTorch is not importable: {exc}")
    print("The image is broken. This is not something you can fix from a notebook.")

if torch is not None and torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU:                 {props.name}")
    print(f"Compute capability:  {props.major}.{props.minor}")
    print(f"Total memory:        {total / 1024**3:.1f} GB")
    print(f"Free memory:         {free / 1024**3:.1f} GB")
    print(f"Visible devices:     {torch.cuda.device_count()}")

    # A GPU that reports much less free memory than total, on a freshly started
    # server, means something else is sharing it. That is worth knowing early.
    if free / total < 0.8:
        print("\n  Note: less than 80% of GPU memory is free on a fresh server.")
        print("  Report this - you may be sharing the device.")
elif torch is not None:
    print("No CUDA device visible. The supported workshop environment is unhealthy.")
    print("Report this output to an instructor and work on non-GPU project tasks.")

# %% [markdown]
# ## 2. Earth2Studio and the model cache
#
# Model checkpoints are multi-gigabyte. The event image contains the reviewed
# checkpoint at `EARTH2STUDIO_MODEL_CACHE`, so fifty servers never download it
# independently. This cell reports what the image contains.

# %%
from pathlib import Path

try:
    import earth2studio

    print(f"earth2studio      {earth2studio.__version__}")
except ImportError as exc:
    print(f"earth2studio      NOT IMPORTABLE - {exc}")

print(f"model cache       {config.model_cache}")
print(f"data cache        {config.data_cache}")

cache = Path(config.model_cache)
if cache.exists():
    files = [f for f in cache.rglob("*") if f.is_file()]
    size = sum(f.stat().st_size for f in files)
    print(f"cache contents    {len(files)} files, {size / 1024**3:.2f} GB")
    for f in sorted(files, key=lambda f: -f.stat().st_size)[:8]:
        print(f"                  {f.stat().st_size / 1024**2:>8.0f} MB  {f.relative_to(cache)}")
else:
    print("cache contents    EMPTY - report this; the event image is incomplete")

# %% [markdown]
# ## 3. TEMPO data access
#
# The staged TEMPO subset is read-only and shared. `manifest.json` next to it
# lists exactly which scans were staged, so you never have to guess a date.

# %%
import json

print(f"TEMPO_DATA_URI    {config.tempo_uri}\n")

if config.manifest:
    print(json.dumps(config.manifest, indent=2))
else:
    print("No manifest found. Either TEMPO_DATA_URI is not set correctly, or the")
    print("staged copy was written without one. The next cell will try to open it")
    print("anyway.")

# %%
from tempo_earth2.tempo import open_tempo_source

try:
    tempo = open_tempo_source(config.tempo_uri)
    print(tempo)
except Exception as exc:
    print(f"Could not open the staged TEMPO data:\n\n  {type(exc).__name__}: {exc}\n")
    print("This is a environment problem, not something you can fix from a notebook.")
    print("Copy this cell's output into a support request.")

# %% [markdown]
# ## 4. Home directory and scratch space
#
# Your home directory is a persistent volume: work saved there survives a
# server restart. Everything outside it does not.

# %%
import shutil

home = Path.home()
usage = shutil.disk_usage(home)
print(f"home              {home}")
print(f"work directory    {config.work_dir}")
print(f"total             {usage.total / 1024**3:.1f} GB")
print(f"used              {usage.used / 1024**3:.1f} GB")
print(f"free              {usage.free / 1024**3:.1f} GB")

probe = config.work_dir / ".write-probe"
try:
    config.work_dir.mkdir(parents=True, exist_ok=True)
    probe.write_text("ok")
    probe.unlink()
    print("writable          yes")
except Exception as exc:
    print(f"writable          NO - {exc}")

# %% [markdown]
# ## 5. Plotting and widgets
#
# If the figure below does not appear, or appears without the interactive
# slider, the problem is browser or network rather than Python - which matters,
# because the fix is different. Try a different browser before asking for help.

# %%
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(5, 3))
x = np.linspace(0, 4 * np.pi, 400)
ax.plot(x, np.sin(x), label="static matplotlib")
ax.legend()
ax.set_title("Static plotting works")
plt.show()

# %% tags=["optional"]
import ipywidgets as widgets
from IPython.display import display

slider = widgets.FloatSlider(value=1.0, min=0.5, max=4.0, description="frequency")
output = widgets.Output()


def _redraw(change):
    with output:
        output.clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(x, np.sin(change["new"] * x))
        ax.set_title(f"Interactive widgets work (f = {change['new']:.1f})")
        plt.show()


slider.observe(_redraw, names="value")
display(slider, output)
_redraw({"new": slider.value})

# %% [markdown]
# ## 6. Recovery commands
#
# Two things can be repaired without an administrator. Both are safe: neither
# deletes anything, they move things aside.
#
# | Symptom | Command (run in a terminal) |
# | --- | --- |
# | A notebook is edited beyond repair | `restore-workshop-notebooks 01_tempo_earth2_intro.ipynb` |
# | Imports or widgets broke after a `pip install` | `reset-user-environment --apply` |
#
# Run either with no arguments first to see what it would do.

# %%
import subprocess

for command in ("restore-workshop-notebooks", "reset-user-environment"):
    result = subprocess.run(
        ["bash", "-lc", f"command -v {command}"], capture_output=True, text=True
    )
    status = result.stdout.strip() or "NOT FOUND"
    print(f"{command:<32}  {status}")

# %% [markdown]
# ---
#
# If everything above looks right, open **`01_tempo_earth2_intro.ipynb`**.
