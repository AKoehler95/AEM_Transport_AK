# AEM grid export → Visual Analysis Workbench — integration guide

The AEM model can now emit its solved concentration grid to disk, in two
formats, ready for the website's Data Analysis Workbench. This guide is for
wiring that into the site so a user can run the AEM model and then either
**download** the grid or **open it straight in the workbench** — no manual
re-entry.

**The model side is done and tested.** Everything under "Model side" below is
shipped. The one piece left is the workbench's `.npz` support (section 4),
which is yours to implement — it's small and fully specified here.

---

## 0. Two formats, two jobs

The grid is fundamentally a 2D array plus two axis vectors. Long-form CSV is a
poor container for it — it repeats every coordinate and stores each number as
~18 text characters, so a default site run is ~4.5 MB. A native array format is
several times smaller. So there are two outputs, for two different audiences:

| Format | What it's for | Notes |
|---|---|---|
| **`.npz`** | The **workbench** — run→import, download, re-import | Compact, lossless, pure-numpy. ~4× (float64) to ~9× (float32) smaller than CSV. Python-only. |
| **`.csv`** | **Humans / other tools** — Excel, R, MATLAB, collaborators | The only format openable without Python. |

Rule of thumb: **`.npz` is the workbench's native format; CSV is the escape
hatch to the non-Python world.** Once the workbench reads `.npz` (section 4),
CSV is no longer needed for any workbench path — keep it only as a
human-facing download.

### The column contract (both formats)

`GRID_COLUMNS = ("x_m", "y_m", "concentration_mgL")` in `src/at_grid_export.py`.
These names are a **contract with the workbench**: its `data_analysis/notation.py`
maps `concentration_mgL` to a typeset `C [mg/L]` axis label by recognising the
trailing `mgl` unit token. Rename a column and the label silently degrades.

- **CSV** — long-form, one row per grid point, y-major (all x at the first y,
  then the next y). Full float precision.
- **NPZ** — the **native grid**, *not* long-form: `x_m` and `y_m` are the 1D
  axes, `concentration_mgL` is the 2D array of shape `(len(y_m), len(x_m))`.
  A consumer rebuilds long-form with one `np.meshgrid`.

Verified: a real run's CSV loads through `datasets.load_csv` + `pivot_gridded`
with no error and renders `C [mg/L]`; the `.npz` round-trips to the identical
grid.

---

## 1. Model side — the API you call

### Config (drives what a run writes to disk)

A single enum in the config dict `ATConfiguration.from_dict()` already accepts:

```python
config_dict["concentration_output"] = "npz"   # "none" | "csv" | "npz" | "both"
config_dict["export_step"]          = 1        # keep every n-th point, both axes
config_dict["export_path"]          = ""       # path STEM (no extension); "" = run dir
config_dict["export_npz_float32"]   = False    # store .npz as float32 (~half size)
```

- `concentration_output` defaults to `"none"`, so an existing run is unchanged.
  An invalid value raises `ValueError` at parse time (a misconfigured web run
  fails loud instead of silently exporting nothing).
- `export_path` is a **stem** — the format appends `.csv` / `.npz`, so `"both"`
  writes two files from one setting. `""` → `grid.csv` / `grid.npz` inside the
  run's own directory.

### Getting the bytes without touching disk

On a completed `ATSimulation`:

```python
sim.grid_npz_bytes(step=1, float32=False)  # -> bytes   (workbench / download)
sim.grid_csv_bytes(step=1)                 # -> bytes   (human download)
sim.export_paths                           # -> list of files run() wrote
```

Or the free functions in `at_grid_export.py`, for when you only have the arrays
(see the pickle gotcha in section 5):

```python
from at_grid_export import grid_npz_bytes, grid_csv_bytes, write_grid_npz, write_grid_csv
```

### Which knob for which path

- **Download format choice (CSV vs NPZ)** is a **workbench UI** decision, not a
  config flag — the workbench holds the grid in memory and can emit either from
  a button (section 4). The config enum only governs what a *run writes to disk*.
- Sampling (`export_step`) is for CLI/batch users wanting a coarse file; the
  website should leave it at `1` and thin in the workbench if ever needed.

---

## 2. Recommended handoff wiring — the workbench pulls, zero coupling to the AEM panel

No change to the AEM result panel. Each run overwrites one `.npz` keyed to the
logged-in user; the workbench offers "load my last AEM run".

**In the AEM run submission**, where it builds the config dict:

```python
import hashlib, os

def _user_export_stem(email: str) -> str:
    # Key by user, NOT one global path — a shared file means two concurrent
    # users clobber each other's exports.
    token = hashlib.sha256(email.encode()).hexdigest()[:16]
    return os.path.join(AEM_EXPORT_DIR, token)          # stem, no extension

config_dict["concentration_output"] = "npz"
config_dict["export_path"]          = _user_export_stem(current_user.email)
```

`AEM_EXPORT_DIR` is a directory you own (add to `settings.py` beside the
`NUMERICAL_*` entries, and to `.gitignore`). The model writes **atomically**
(temp file + `os.replace`), so a new run replaces the user's previous file the
instant it completes — no TTL, no cleanup task, disk bounded at one file per
user, and a crashed run leaves the prior file intact.

**In the workbench**, add a "Load my last AEM run" button that resolves the same
per-user path (read the email from the `?email=` its route already forwards —
`source_inversion_routes.py` sets `params["email"] = current_user.email`) and
feeds the file through the `.npz` loader from section 4. Show the file's mtime
and row count so the user knows which run it is.

---

## 3. Optional shortcut — a download button on the AEM result page

Additive; the pull path in section 2 works without it. Give the user both
formats:

```python
import io
import panel as pn

def _npz_cb():
    sim = state.get("sim")
    return io.BytesIO(sim.grid_npz_bytes()) if sim else io.BytesIO(b"")

def _csv_cb():
    sim = state.get("sim")
    return io.BytesIO(sim.grid_csv_bytes()) if sim else io.BytesIO(b"")

npz_btn = pn.widgets.FileDownload(callback=_npz_cb, filename="aem_grid.npz",
                                  label="Download grid (.npz)", button_type="primary", visible=False)
csv_btn = pn.widgets.FileDownload(callback=_csv_cb, filename="aem_grid.csv",
                                  label="Download grid (.csv)", button_type="default", visible=False)
# unhide both where the run's result is rendered
```

Same pattern as `export_btn` in `panel_numerical_vertical_single.py:55`.

---

## 4. ★ Workbench `.npz` support — the change you implement

Three small additions to the workbench (all in the website repo):

**(a) A loader beside `load_csv`** in `data_analysis/datasets.py` (next to
`load_csv`, line 18). It expands the native grid into the **same long-form
DataFrame** the upload path produces, so every existing tab works unchanged and
the Scientific tab's `pivot_gridded` re-grids exactly as today:

```python
import io
import numpy as np
import pandas as pd

def load_npz(raw: bytes) -> pd.DataFrame:
    with np.load(io.BytesIO(raw)) as d:
        x, y, c = d["x_m"], d["y_m"], d["concentration_mgL"]
    if c.shape != (y.size, x.size):
        raise ValueError("NPZ concentration shape does not match its axes.")
    X, Y = np.meshgrid(x, y)
    return pd.DataFrame({
        "x_m": X.ravel(), "y_m": Y.ravel(), "concentration_mgL": c.ravel(),
    })
```

**(b) Accept `.npz` in the uploader.** In `panel_data_analysis.py`, widen the
FileInput (line 90) and branch in `_on_upload` (line 144):

```python
file_input = pn.widgets.FileInput(accept=".csv,.npz", multiple=False)
...
name = (file_input.filename or "").lower()
is_npz = name.endswith(".npz") or file_input.value[:4] == b"PK\x03\x04"  # npz is a zip
df = datasets.load_npz(file_input.value) if is_npz else datasets.load_csv(file_input.value)
```

The same `load_npz` is what the section-2 "load my last AEM run" button calls.

**(c) An `.npz` download button** beside the existing `data_download`
(`panel_data_analysis.py:411`), enabled once a grid exists, so the user gets the
CSV-or-NPZ download choice. Build the bytes from the current grid with
`at_grid_export.grid_npz_bytes(grid.x, grid.y, grid.values)` (the workbench can
import `at_grid_export` once `src/` is on its path — see section 5), or inline a
two-line `np.savez_compressed` using the same `GRID_COLUMNS` keys.

That's the whole workbench change. Nothing else moves.

---

## 5. CLI, for local testing

```bash
cd src

python main.py --format npz                          # grid.npz in the run directory
python main.py --format both --path /tmp/latest      # /tmp/latest.csv + /tmp/latest.npz
python main.py --format npz --npz-float32 --path /tmp/x   # smaller float32 npz
python main.py --format both --step 5 --path /tmp/x  # every 5th point on each axis
python main.py                                       # no flags -> no export, as before
```

Or set `"concentration_output"` in `simulation_config.json`.

Sanity-check a file against the workbench without the server:

```python
from data_analysis import datasets
from data_analysis.grids import pivot_gridded
df = datasets.load_csv(open("grid.csv", "rb").read())     # or datasets.load_npz(...)
pivot_gridded(df, "x_m", "y_m", "concentration_mgL")      # no error == accepted
```

---

## 6. Gotchas

- **Flat module directory.** `src/` has no `__init__.py`; it must be on
  `sys.path`, and imports are `from at_grid_export import ...`, never
  `from src.at_grid_export import ...`.

- **Queued runs are pickled.** If the AEM run goes through the site's
  `numerical_jobs` queue, the result is pickled (`numerical_jobs.py:141`), so a
  **live `ATSimulation` will not survive** — don't call `sim.grid_npz_bytes()`
  on the far side of the queue. Either have the worker set
  `concentration_output` / `export_path` and write the file *before* it pickles,
  or make the pickled object carry `xaxis`, `yaxis`, `result` and use the free
  function `grid_npz_bytes(x, y, result)`.

- **NPZ shape.** `x_m`/`y_m` are 1D axes; `concentration_mgL` is 2D
  `(len(y), len(x))`. Long-form CSV, by contrast, is one row per point. Don't
  cross the two up.

- **Renaming `GRID_COLUMNS`** breaks the workbench's axis labels and its
  `load_npz` keys. See section 0.

- **MathJax.** The workbench's LaTeX axis labels only typeset when Panel starts
  with `pn.extension(..., "mathjax", ...)`. `panel_server.py` currently reads
  `pn.extension("tabulator")` — add `"mathjax"` when you register the workbench
  app, or every axis label shows raw TeX (`\mathr…`).

- **Version pins.** Per the workbench's own `INTEGRATION.md §3`, Panel ≥ 1.5 /
  Bokeh ≥ 3.5 silently drop dynamically-added figures — which is how the
  Scientific tab renders. Keep `panel==1.4.5` / `bokeh==3.4.3`.
