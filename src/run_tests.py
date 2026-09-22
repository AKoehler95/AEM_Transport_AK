#Code written by Alvin Yadav
"""
AEM Solver Test Runner

Batch-runs every JSON configuration in a directory through the AEM
simulation pipeline and collects the outputs into summary files.

Each test loads a config, runs the full simulation (solve, grid
evaluation, plots, statistics), and records the timing, solver
diagnostics, and key outputs (L_max, concentration range, etc.).

Outputs (in ./batch_results/):
  test_results.json   — machine-readable per-test data
  test_summary.txt    — human-readable formatted table + analytics

Usage:
    python run_tests.py [config_dir]

Default config_dir is ./batch_configs/ relative to this script.
Batch output (test_results.json, test_summary.txt) goes to ./batch_results/.
"""

import argparse
import sys
import os
import json
import time
import threading
import traceback
import logging
import io
import math
import re
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# ── Run-button defaults ──────────────────────────────────────────────────────
# Edit these three when running from an IDE Run button (Spyder / PyCharm), which
# launches the file with no command-line arguments. Command-line flags, when
# given, override every one of them (see _parse_args), so the CLI still works
# exactly as before.
#
#   CONFIG_DIR     : folder of *.json configs to run. An absolute path, or a
#                    name/relative path resolved against this script's folder.
#   USE_PREDICTOR  : True to seed dom_xmax from the learned XmaxPredictor
#                    instead of each config's built-in guess.
#   LEARN          : True to append each clean run to the xmax dataset and
#                    refit the model afterwards.
CONFIG_DIR    = "mc_batch_008"
USE_PREDICTOR = True
LEARN         = True
# ─────────────────────────────────────────────────────────────────────────────

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
# Resolve CONFIG_DIR against the script folder unless it's already absolute, so
# a bare name like "batch_configs" works regardless of the IDE's working dir.
if not os.path.isabs(CONFIG_DIR):
    CONFIG_DIR = os.path.join(SCRIPT_DIR, CONFIG_DIR)   # overridden in main() from argv
RESULTS_DIR = os.path.join(SCRIPT_DIR, "batch_results/Batch_008")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset path for the learned dom_xmax predictor (xmax_features.py /
# xmax_predictor.py) — canonical definition lives in xmax_predictor so
# train_xmax_from_runs.py shares the same default.
from xmax_predictor import DEFAULT_DATASET_PATH as LMAX_DATASET_PATH

# Maximum wall-clock time per test before it is marked TIMEOUT.
TEST_TIMEOUT = 500

# Log capture
log_capture = io.StringIO()
log_handler = logging.StreamHandler(log_capture)
log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log_handler)


def fmt_f(v, spec=".2f"):
    """Format a numeric value with the given format spec, or '-' if None."""
    return f"{v:{spec}}" if v is not None else "-"


def _parse_config_meta(config_name):
    """Extract archetype letter and cluster count from a config filename.

    Expected format: {A-J}{num}_{n_elems}circ_{n_clusters}cl[.json]
    e.g. A001_45circ_3cl.json -> ('A', 3)
    Returns ('?', None) for unrecognised names.
    """
    arch = config_name[0].upper() if config_name else "?"
    if arch not in "ABCDEFGHIJ":
        arch = "?"
    m = re.search(r'_(\d+)cl', config_name)
    n_cl = int(m.group(1)) if m else None
    return arch, n_cl


def _run_one_body(config_path, result, stdout_buf, use_predictor=False, predictor=None):
    """
    Execute a single configuration through the full simulation pipeline
    and populate the result dict with outputs and diagnostics.

    Runs inside a worker thread so the outer run_one() can enforce a
    timeout. Redirects sim.run()'s stdout/stderr into stdout_buf, then
    parses it for solver-method information (coupled vs decoupled,
    condition number, group count).

    If use_predictor is True, config.dom_xmax is overwritten with the
    learned predictor's suggestion before the run; the existing dynamic
    domain-extension loop in ATSimulation.run() is untouched either way and
    remains the correctness guarantee against under-prediction.
    """
    from at_config import ATConfiguration
    from at_simulation import ATSimulation
    import xmax_features

    config = ATConfiguration.from_json(config_path)
    result["num_elements"] = len(config.elements)
    result["orientation"]  = config.orientation

    features = xmax_features.extract_features(config)
    result["_features"] = features
    result["_config_hash"] = xmax_features.config_hash_from_path(config_path)

    if use_predictor:
        pred = predictor.predict(features)
        config.dom_xmax = pred["dom_xmax"]
        result["predictor_trained"] = pred["trained"]
        result["L_pred_q"] = pred["L_pred_q"]
    result["dom_xmax_initial"] = config.dom_xmax

    sim = ATSimulation(config)
    with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
        sim.run()

    stdout_text = stdout_buf.getvalue()
    result["stdout_capture"] = stdout_text
    result["log_capture"]    = log_capture.getvalue()
    result["num_elements_after_mirroring"] = len(sim.config.elements)

    # Extract phase timings from the simulation's built-in timer
    if hasattr(sim, 'timer') and hasattr(sim.timer, 'phases'):
        result["phase_timings"] = {k: round(v, 3) for k, v in sim.timer.phases.items()}

    # Solver diagnostics come from the simulation itself; the terminal output
    # only says "Coupled solve" / "Decoupled solve".
    info = getattr(sim, "solve_info", None) or {}
    if info.get("decision"):
        result["solve_method"] = info["decision"]
    if info.get("cond") is not None:
        result["condition_number"] = float(info["cond"])
    groups = info.get("groups")
    result["num_groups"] = len(groups) if groups else 1

    # Parse validation result
    for ln in (stdout_text or "").splitlines():
        s = ln.strip()
        if s.startswith("VALIDATION PASS"):
            result["validation_passed"] = True
        elif s.startswith("VALIDATION FAIL"):
            result["validation_passed"] = False

    # Record simulation outputs
    result["L_max"]            = sim.L_max
    result["interface_length"] = getattr(sim, "interface_length", None)
    result["result_min"]       = float(np.min(sim.result))
    result["result_max"]       = float(np.max(sim.result))
    result["status"]           = "DONE"

    # Domain-sizing diagnostics for the xmax dataset (see xmax_predictor.py).
    # "censored" mirrors ATSimulation.check_domain_adequacy: the plume ran
    # into the domain edge, so L_max is only a lower bound and must not be
    # used as a training label.
    dom_xmax_final = sim.config.dom_xmax
    n_extensions = sum(
        1 for k in (result.get("phase_timings") or {}) if k.startswith("conc_array_ext")
    )
    domain_width = dom_xmax_final - sim.config.dom_xmin
    censored_x = True
    if sim.L_max is not None and domain_width > 0:
        max_x_distance = sim.L_max - sim.config.dom_xmin
        censored_x = max_x_distance >= 0.95 * domain_width

    # Vertical capping clips the part of the envelope that reaches furthest in
    # x, so L_max is a lower bound there too (check_vertical_adequacy). Without
    # this, a plume pressed against dom_ymin was recorded as a clean run.
    domain_height = sim.config.dom_ymax - sim.config.dom_ymin
    z_min_reached = getattr(sim, "z_min_reached", None)
    censored_z = False
    if z_min_reached is not None and domain_height > 0:
        censored_z = (z_min_reached - sim.config.dom_ymin) <= 0.05 * domain_height

    result["dom_xmax_final"] = dom_xmax_final
    result["dom_ymin_final"] = sim.config.dom_ymin
    result["n_extensions"]   = n_extensions
    result["z_min_reached"]  = z_min_reached
    result["censored_x"]     = censored_x
    result["censored_z"]     = censored_z
    result["censored"]       = censored_x or censored_z


def _append_dataset_row(path, row):
    """Append one JSON line to the xmax training dataset, creating dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_one(config_path, use_predictor=False, predictor=None, learn=False,
            dataset_path=LMAX_DATASET_PATH):
    """
    Run a single test configuration with timeout enforcement.

    Spawns a worker thread to execute _run_one_body. If the thread
    doesn't complete within TEST_TIMEOUT seconds, the result is marked
    TIMEOUT. If the worker raises an exception, the result is marked
    ERROR with the exception details recorded.

    use_predictor: seed dom_xmax from the learned XmaxPredictor instead of
        the value baked into the config file. The dynamic extension loop
        stays active regardless, so a bad prediction only costs time.
    learn: after a clean (non-censored) DONE run, append the labelled row
        to the xmax dataset and refit the predictor on it. Independent of
        use_predictor — you can collect training data while still running
        with the default domain sizing.

    Returns the result dict with all fields populated.
    """
    result = dict(
        config=os.path.basename(config_path),
        status="UNKNOWN",
        num_elements=0,
        num_elements_after_mirroring=0,
        orientation="",
        condition_number=None,
        solve_method="",
        num_groups=None,
        L_max=None,
        interface_length=None,
        validation_passed=None,
        result_min=None,
        result_max=None,
        phase_timings=None,
        wall_time_s=None,
        error=None,
        stdout_capture="",
        log_capture="",
        dom_xmax_initial=None,
        dom_xmax_final=None,
        n_extensions=None,
        censored=None,
        censored_x=None,
        censored_z=None,
        z_min_reached=None,
        dom_ymin_final=None,
        predictor_trained=False,
        L_pred_q=None,
    )

    stdout_buf = io.StringIO()
    log_capture.truncate(0)
    log_capture.seek(0)
    thread_exc = [None]

    def worker():
        try:
            _run_one_body(config_path, result, stdout_buf,
                           use_predictor=use_predictor, predictor=predictor)
        except Exception as e:
            thread_exc[0] = e

    t0 = time.time()
    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=TEST_TIMEOUT)
    result["wall_time_s"] = round(time.time() - t0, 2)

    if t.is_alive():
        result["status"] = "TIMEOUT"
        result["error"]  = f"Exceeded {TEST_TIMEOUT}s"
        result["stdout_capture"] = stdout_buf.getvalue()
        result.pop("_features", None)
        result.pop("_config_hash", None)
        return result

    if thread_exc[0] is not None:
        e = thread_exc[0]
        result["status"] = "ERROR"
        result["error"]  = f"{type(e).__name__}: {e}"
        result["stdout_capture"] = stdout_buf.getvalue()
        result["log_capture"] = log_capture.getvalue()
        traceback.print_exception(type(e), e, e.__traceback__)

    features = result.pop("_features", None)
    config_hash = result.pop("_config_hash", None)
    if learn and result["status"] == "DONE" and features is not None:
        _append_dataset_row(dataset_path, {
            "config_name": result["config"],
            "config_hash": config_hash,
            **features,
            "L_max": result["L_max"],
            "dom_xmax_initial": result["dom_xmax_initial"],
            "dom_xmax_final": result["dom_xmax_final"],
            "n_extensions": result["n_extensions"],
            "censored": result["censored"],
            "wall_time_s": result["wall_time_s"],
        })
        if not result["censored"] and result["L_max"] is not None:
            predictor.update(features, result["L_max"])
            predictor.refit()
            predictor.save()

    return result


def _analyse_results(all_results):
    """
    Derive a structured analytics report from all test results.

    Examines solver behaviour, timing breakdown, plume size distribution,
    and concentration range anomalies. Returns a list of text lines ready
    to append to the summary file.

    Analytics sections:
      1. Solver strategy breakdown — how often coupled vs decoupled fires,
         and whether decoupled solve is reliably faster or slower.
      2. Timing hotspots — which pipeline phase dominates total runtime,
         and which individual tests consume disproportionate time.
      3. L_max distribution — histogram-style bucketing of plume lengths
         and identification of the longest / shortest cases.
      4. Concentration range anomalies — flags cases where result_min or
         result_max fall far outside the physically expected window, which
         typically indicates numerical blowup or solver divergence.
      5. Grid time vs plume length — correlation analysis: long plumes
         require multiple domain extensions and dominate conc_array time.
      6. Conditioning vs solve time — whether high condition number
         predicts slow solves.
    """
    done = [r for r in all_results if r["status"] == "DONE"]
    if not done:
        return ["No completed tests to analyse."]

    lines = []
    sep = "-" * 80

    # Solver strategy breakdown
    lines.append("\n" + "=" * 80)
    lines.append("ANALYTICS")
    lines.append("=" * 80)

    # Validation summary
    v_passed  = [r for r in done if r.get("validation_passed") is True]
    v_failed  = [r for r in done if r.get("validation_passed") is False]
    v_unknown = [r for r in done if r.get("validation_passed") is None]

    lines.append("\n0. Solution validation  (result_max > ca  AND  result_min < -ca/2)")
    lines.append(sep)
    lines.append(f"  PASS    : {len(v_passed):3d} / {len(done)}  ({100*len(v_passed)/len(done):.0f}%)")
    lines.append(f"  FAIL    : {len(v_failed):3d} / {len(done)}  ({100*len(v_failed)/len(done):.0f}%)")
    if v_unknown:
        lines.append(f"  Unknown : {len(v_unknown):3d}")
    if v_failed:
        lines.append("\n  Failed runs:")
        for r in v_failed:
            lines.append(
                f"    {r['config']:<45}  "
                f"method={r.get('solve_method','?'):<10}  "
                f"cond={fmt_f(r.get('condition_number'), '.2e')}  "
                f"result_max={fmt_f(r.get('result_max'), '.2f')}"
            )

    coupled   = [r for r in done if r.get("solve_method") == "coupled"]
    decoupled = [r for r in done if r.get("solve_method") == "decoupled"]
    unknown   = [r for r in done if not r.get("solve_method")]

    lines.append("\n1. Solver strategy")
    lines.append(sep)
    lines.append(f"  Coupled   : {len(coupled):3d} / {len(done)}  ({100*len(coupled)/len(done):.0f}%)")
    lines.append(f"  Decoupled : {len(decoupled):3d} / {len(done)}  ({100*len(decoupled)/len(done):.0f}%)")
    if unknown:
        lines.append(f"  Unknown   : {len(unknown):3d} (condition number not captured)")

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    if coupled and decoupled:
        ct_times = [r["wall_time_s"] for r in coupled  if r["wall_time_s"]]
        dt_times = [r["wall_time_s"] for r in decoupled if r["wall_time_s"]]
        lines.append(f"\n  Wall time  — coupled mean: {_mean(ct_times):.1f}s  "
                     f"decoupled mean: {_mean(dt_times):.1f}s")

        ct_solve = [r["phase_timings"].get("solve_system", 0)
                    for r in coupled if r.get("phase_timings")]
        dt_solve = [r["phase_timings"].get("solve_system", 0)
                    for r in decoupled if r.get("phase_timings")]
        lines.append(f"  Solve time — coupled mean: {_mean(ct_solve):.1f}s  "
                     f"decoupled mean: {_mean(dt_solve):.1f}s")

    # Pass/fail breakdown by solver method
    for method_label, group in [("Coupled", coupled), ("Decoupled", decoupled)]:
        if not group:
            continue
        gp = sum(1 for r in group if r.get("validation_passed") is True)
        gf = sum(1 for r in group if r.get("validation_passed") is False)
        lines.append(f"  {method_label} pass/fail: {gp} PASS  {gf} FAIL  "
                     f"({100*gp/len(group):.0f}% pass rate)")

    # Timing hotspots
    lines.append("\n2. Timing breakdown")
    lines.append(sep)

    phase_names = set()
    for r in done:
        if r.get("phase_timings"):
            phase_names.update(r["phase_timings"].keys())

    # Total grid time (initial + extensions)
    grid_phases = [p for p in phase_names if "conc_array" in p]
    total_grid = []
    for r in done:
        if r.get("phase_timings"):
            tg = sum(r["phase_timings"].get(p, 0) for p in grid_phases)
            total_grid.append(tg)

    total_wall = [r["wall_time_s"] for r in done if r["wall_time_s"]]
    if total_grid and total_wall:
        lines.append(f"  Grid evaluation (summed across all passes): "
                     f"mean={_mean(total_grid):.1f}s  "
                     f"mean wall={_mean(total_wall):.1f}s  "
                     f"(extensions multiply grid time for long plumes)")

    # Extension frequency
    ext_counts = {1: 0, 2: 0, 3: 0}
    for r in done:
        if r.get("phase_timings"):
            for k in range(1, 4):
                if r["phase_timings"].get(f"conc_array_ext{k}", 0) > 0:
                    ext_counts[k] += 1
    lines.append(f"  Domain extension frequency: "
                 f"≥1 ext: {ext_counts[1]} tests  "
                 f"≥2 ext: {ext_counts[2]} tests  "
                 f"≥3 ext: {ext_counts[3]} tests")

    # Slowest tests
    slowest = sorted(done, key=lambda r: r["wall_time_s"] or 0, reverse=True)[:5]
    lines.append(f"\n  5 slowest tests:")
    for r in slowest:
        pt = r.get("phase_timings") or {}
        tg = sum(pt.get(p, 0) for p in grid_phases)
        lines.append(f"    {r['config']:<45}  "
                     f"wall={fmt_f(r['wall_time_s'])}s  "
                     f"grid={tg:.1f}s  "
                     f"solve={pt.get('solve_system',0):.1f}s  "
                     f"L_max={fmt_f(r['L_max'])}")

    # Near-timeout cases (>= 90% of TEST_TIMEOUT)
    near_timeout = [r for r in done if r["wall_time_s"] and r["wall_time_s"] >= 0.9 * TEST_TIMEOUT]
    if near_timeout:
        lines.append(f"\n  Near-timeout cases ({len(near_timeout)} tests at ≥{0.9*TEST_TIMEOUT:.0f}s):")
        for r in near_timeout:
            lines.append(f"    {r['config']:<45}  L_max={fmt_f(r['L_max'])}  "
                         f"#elems={r['num_elements']}")

    # L_max distribution
    lines.append("\n3. Plume length (L_max) distribution")
    lines.append(sep)

    lmaxes = [r["L_max"] for r in done if r["L_max"] is not None]
    if lmaxes:
        sl = sorted(lmaxes)
        n = len(sl)
        lines.append(f"  N={n}  min={sl[0]}  p25={sl[n//4]}  "
                     f"median={sl[n//2]}  p75={sl[3*n//4]}  max={sl[-1]}  "
                     f"mean={_mean(lmaxes):.1f} m")

        # Histogram buckets
        buckets = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 9999)]
        lines.append("  Distribution:")
        for lo, hi in buckets:
            count = sum(1 for v in lmaxes if lo <= v < hi)
            bar = "█" * count
            lines.append(f"    {lo:>4}–{hi:<4} m : {count:3d}  {bar}")

        # Longest plumes (most expensive grid evaluation)
        longest = sorted(done, key=lambda r: r["L_max"] or 0, reverse=True)[:5]
        lines.append(f"\n  5 longest plumes:")
        for r in longest:
            lines.append(f"    {r['config']:<45}  L_max={r['L_max']}  "
                         f"#elems={r['num_elements']}  "
                         f"method={r.get('solve_method','?')}")

        # Shared helpers for 3a–3d
        ARCH_NAMES = {
            "A": "vert.stack",   "B": "horiz.spread", "C": "top-heavy",
            "D": "bot-heavy",    "E": "tight-pack",   "F": "scattered",
            "G": "diagonal",     "H": "two-groups",   "I": "outlier",
            "J": "dense-ctr",    "?": "unknown",
        }
        arch_buckets = arch_buckets = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 9999)]
        arch_bucket_labels = ["0–100", "100–300", "300–600", "600–1000", "1000+"]

        def _std(vals):
            if len(vals) < 2:
                return 0.0
            m = _mean(vals)
            return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

        # Per-archetype L_max summary
        arch_lmax = {}
        for r in done:
            if r.get("L_max") is None:
                continue
            arch, _ = _parse_config_meta(r["config"])
            arch_lmax.setdefault(arch, []).append(r["L_max"])

        if arch_lmax:
            lines.append("\n  3a. Mean L_max per archetype")
            lines.append(
                f"  {'Arch':<4} {'Name':<12} {'N':>3}  "
                f"{'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}  {'Median':>7}"
            )
            lines.append("  " + "-" * 60)
            for arch in sorted(arch_lmax.keys()):
                vals = sorted(arch_lmax[arch])
                n = len(vals)
                median = vals[n // 2]
                lines.append(
                    f"  {arch:<4} {ARCH_NAMES.get(arch, '?'):<12} {n:>3}  "
                    f"{_mean(vals):>7.1f}  {_std(vals):>7.1f}  "
                    f"{vals[0]:>7.1f}  {vals[-1]:>7.1f}  {median:>7.1f}"
                )

        # Per-archetype L_max histogram
        if arch_lmax:
            lines.append("\n  3b. L_max histogram per archetype")
            lines.append(
                "  " + f"{'Arch':<6}" +
                "  ".join(f"{lbl:>8}" for lbl in arch_bucket_labels)
            )
            lines.append("  " + "-" * (6 + 10 * len(arch_bucket_labels)))
            for arch in sorted(arch_lmax.keys()):
                vals = arch_lmax[arch]
                row = f"  {arch:<6}"
                for lo, hi in arch_buckets:
                    count = sum(1 for v in vals if lo <= v < hi)
                    row += f"{'█' * count:>8}  "
                lines.append(row)

        # Mean L_max by cluster count
        cl_lmax = {}
        for r in done:
            if r.get("L_max") is None:
                continue
            _, n_cl = _parse_config_meta(r["config"])
            if n_cl is not None:
                cl_lmax.setdefault(n_cl, []).append(r["L_max"])

        if cl_lmax:
            lines.append("\n  3c. Mean L_max by cluster count")
            lines.append(
                f"  {'n_cl':>4}  {'N':>3}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}"
            )
            lines.append("  " + "-" * 46)
            for n_cl in sorted(cl_lmax.keys()):
                vals = cl_lmax[n_cl]
                lines.append(
                    f"  {n_cl:>4}  {len(vals):>3}  {_mean(vals):>7.1f}  "
                    f"{_std(vals):>7.1f}  {min(vals):>7.1f}  {max(vals):>7.1f}"
                )

        #Mean L_max by element count (pre-mirroring)
        elem_bins = [(0, 30), (30, 60), (60, 100), (100, 999999)]
        elem_labels = ["<30", "30–60", "60–100", "100+"]

        if any(r.get("num_elements") is not None and r.get("L_max") is not None
               for r in done):
            lines.append("\n  3d. Mean L_max by element count (pre-mirroring)")
            lines.append(
                f"  {'Elements':>10}  {'N':>3}  {'Mean':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}"
            )
            lines.append("  " + "-" * 52)
            for (lo, hi), lbl in zip(elem_bins, elem_labels):
                vals = [r["L_max"] for r in done
                        if r.get("L_max") is not None
                        and r.get("num_elements") is not None
                        and lo <= r["num_elements"] < hi]
                if not vals:
                    continue
                lines.append(
                    f"  {lbl:>10}  {len(vals):>3}  {_mean(vals):>7.1f}  "
                    f"{_std(vals):>7.1f}  {min(vals):>7.1f}  {max(vals):>7.1f}"
                )

    # Concentration range anomalies
    lines.append("\n4. Concentration range anomalies")
    lines.append(sep)

    # Expected background is around -8 mg/l; flag anything far below -12
    extreme_min = sorted(
        [r for r in done if r.get("result_min") is not None and r["result_min"] < -12],
        key=lambda r: r["result_min"]
    )
    lines.append(f"  result_min < -50 mg/l: {len(extreme_min)} cases")
    for r in extreme_min[:5]:
        lines.append(f"    {r['config']:<45}  result_min={r['result_min']:.1f}  "
                     f"method={r.get('solve_method','?')}  cond={fmt_f(r.get('condition_number'), '.2e')}")

    # Physically impossible positives (result_max >> max donor conc)
    # Max donor is typically 50 mg/l — flag anything > 60
    extreme_max = sorted(
        [r for r in done if r.get("result_max") is not None and r["result_max"] > 60],
        key=lambda r: -r["result_max"]
    )
    lines.append(f"  result_max > 100 mg/l: {len(extreme_max)} cases")
    for r in extreme_max[:5]:
        lines.append(f"    {r['config']:<45}  result_max={r['result_max']:.1f}")

    # Grid time vs plume length
    lines.append("\n5. Grid time vs plume length")
    lines.append(sep)
    lines.append("  Total grid time grows rapidly with L_max because long plumes")
    lines.append("  trigger domain extensions (each adding a full grid pass).")

    buckets_lmax = [(0, 100), (100, 300), (300, 600), (600, 1000), (1000, 9999)]
    for lo, hi in buckets_lmax:
        group = [r for r in done
                 if r.get("L_max") is not None and lo <= r["L_max"] < hi
                 and r.get("phase_timings")]
        if not group:
            continue
        grids = [sum(r["phase_timings"].get(p, 0) for p in grid_phases) for r in group]
        walls = [r["wall_time_s"] for r in group if r["wall_time_s"]]
        label = f"{lo}–{hi if hi < 9999 else '∞'} m"
        lines.append(f"  L_max {label:<12} ({len(group):2d} tests) "
                     f"mean grid={_mean(grids):.0f}s  mean wall={_mean(walls):.0f}s")

    return lines


def write_summary(all_results, path):
    """
    Write a formatted text summary of all test results to `path`.

    Contains a header with timestamp and config directory, a table row
    per test with key diagnostics, aggregate statistics, and an analytics
    section derived from cross-test patterns.
    """
    col_w = 42
    header = (
        f"{'Config':<{col_w}} {'St':>6} {'#El':>3} {'Ori':>5} "
        f"{'Cond':>10} {'Method':>10} {'L_max':>6} "
        f"{'ResMin':>8} {'ResMax':>8} {'Val':>5} {'Time':>7}"
    )
    sep = "-" * len(header)
    lines = []

    lines.append(f"AEM Test Suite Results")
    lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Config dir: {CONFIG_DIR}")
    lines.append(sep)

    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append(
        f"Total: {len(all_results)}  |  "
        + "  ".join(f"{s}: {c}" for s, c in sorted(counts.items()))
    )
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in all_results:
        name = r["config"].replace(".json", "")
        if len(name) > col_w:
            name = name[:col_w-1] + "…"

        cond_str = f"{r['condition_number']:.2e}" if r["condition_number"] else "-"

        vp = r.get("validation_passed")
        val_str = "PASS" if vp is True else ("FAIL" if vp is False else "-")
        lines.append(
            f"{name:<{col_w}} "
            f"{r['status']:>6} "
            f"{r['num_elements']:>3} "
            f"{r['orientation'][:5]:>5} "
            f"{cond_str:>10} "
            f"{(r['solve_method'] or '-')[:10]:>10} "
            f"{fmt_f(r['L_max']):>6} "
            f"{fmt_f(r['result_min'], '.2f'):>8} "
            f"{fmt_f(r['result_max'], '.2f'):>8} "
            f"{val_str:>5} "
            f"{fmt_f(r['wall_time_s'])}s"
        )

    lines.append(sep)

    # Aggregate statistics over completed tests
    done = [r for r in all_results if r["status"] == "DONE"]
    if done:
        lines.append("\nStatistics over completed tests:")
        for field, label in [
            ("wall_time_s",  "Wall time [s]"),
            ("L_max",        "L_max [m]"),
        ]:
            vals = [r[field] for r in done if r[field] is not None]
            if vals:
                lines.append(
                    f"  {label:<22}  "
                    f"min={min(vals):.4g}  max={max(vals):.4g}  "
                    f"mean={sum(vals)/len(vals):.4g}"
                )

    # Phase timing aggregation
    phase_names = set()
    for r in done:
        if r.get("phase_timings"):
            phase_names.update(r["phase_timings"].keys())
    if phase_names:
        lines.append("\nAggregate phase timings:")
        for phase in sorted(phase_names):
            vals = [r["phase_timings"].get(phase, 0)
                    for r in done if r.get("phase_timings")]
            if vals:
                lines.append(
                    f"  {phase:<30}  "
                    f"mean={sum(vals)/len(vals):.2f}s  max={max(vals):.2f}s"
                )

    # Error/timeout details
    problems = [r for r in all_results if r["status"] in ("ERROR", "TIMEOUT")]
    if problems:
        lines.append(f"\nERROR/TIMEOUT details:")
        lines.append(sep)
        for r in problems:
            lines.append(f"\n[{r['status']}] {r['config']}")
            if r.get("error"):
                lines.append(f"  {r['error']}")

    # Analytics section
    lines.extend(_analyse_results(all_results))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="AEM Solver Test Runner")
    parser.add_argument(
        "config_dir", nargs="?", default=CONFIG_DIR,
        help="Directory of JSON configs to run (default: ./batch_configs)",
    )
    # Defaults come from the module-level constants above so the IDE Run button
    # (no args) honours them; passing a flag overrides the corresponding one.
    predictor_group = parser.add_mutually_exclusive_group()
    predictor_group.add_argument(
        "--use-predictor", dest="use_predictor", action="store_true", default=USE_PREDICTOR,
        help="Seed dom_xmax from the learned XmaxPredictor instead of the config's "
             "built-in guess (dynamic extension loop stays on as a safety net).",
    )
    predictor_group.add_argument(
        "--no-predictor", dest="use_predictor", action="store_false",
        help="Use the config's default dom_xmax (legacy behaviour).",
    )
    learn_group = parser.add_mutually_exclusive_group()
    learn_group.add_argument(
        "--learn", dest="learn", action="store_true", default=LEARN,
        help="After each clean (non-censored) run, append it to the xmax dataset "
             "and refit the predictor.",
    )
    learn_group.add_argument(
        "--no-learn", dest="learn", action="store_false",
        help="Don't log runs or update the predictor.",
    )
    return parser.parse_args()


def main():
    """
    Discover all JSON configs, run each through the simulation, print
    progress to the console, and write summary files.
    """
    global CONFIG_DIR
    args = _parse_args()
    CONFIG_DIR = args.config_dir

    predictor = None
    if args.use_predictor or args.learn:
        from xmax_predictor import XmaxPredictor
        predictor = XmaxPredictor()

    configs = sorted(
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR) if f.endswith(".json")
    )
    if not configs:
        print(f"No .json configs found in {CONFIG_DIR}")
        sys.exit(1)

    print(f"Running {len(configs)} tests from {CONFIG_DIR}")
    print(f"Results -> {RESULTS_DIR}")
    print(f"Predictor: {'ON' if args.use_predictor else 'off'}  "
          f"(trained on {len(predictor.y) if predictor else 0} rows)  "
          f"Learn: {'ON' if args.learn else 'off'}\n")

    all_results = []
    for i, cfg_path in enumerate(configs, 1):
        name = os.path.basename(cfg_path)
        print(f"[{i:3d}/{len(configs)}] {name} ... ", end="", flush=True)

        result = run_one(cfg_path, use_predictor=args.use_predictor,
                          predictor=predictor, learn=args.learn,
                          dataset_path=LMAX_DATASET_PATH)
        all_results.append(result)

        # Compact one-line status with phase breakdown
        phase_str = ""
        if result.get("phase_timings"):
            pt = result["phase_timings"]
            phase_str = (
                f"  solve={pt.get('solve_system',0):.1f}s "
                f"grid={pt.get('conc_array',0):.1f}s "
                f"stats={pt.get('print_statistics',0):.1f}s "
                f"plot={pt.get('plot_result',0):.1f}s"
            )

        print(f"{result['status']:<7}  t={fmt_f(result['wall_time_s'])}s"
              f"  L_max={fmt_f(result.get('L_max'))}{phase_str}",
              flush=True)

    # Write JSON (excluding verbose stdout/log captures)
    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(
            [{k: v for k, v in r.items()
              if k not in ("stdout_capture", "log_capture")}
             for r in all_results],
            f, indent=2, default=str
        )

    # Write human-readable summary
    summary_path = os.path.join(RESULTS_DIR, "test_summary.txt")
    write_summary(all_results, summary_path)

    # Final tally
    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_results)} tests")
    for s, c in sorted(counts.items()):
        print(f"  {s:<10} {c}")

    done = [r for r in all_results if r["status"] == "DONE"]
    if done:
        n_ext = [r["n_extensions"] for r in done if r["n_extensions"] is not None]
        n_capped = sum(1 for r in done if r.get("censored"))
        n_cap_x = sum(1 for r in done if r.get("censored_x"))
        n_cap_z = sum(1 for r in done if r.get("censored_z"))
        if n_ext:
            print(f"  mean extensions: {sum(n_ext)/len(n_ext):.2f}")
        print(f"  capped L_max: {n_capped}/{len(done)} "
              f"(x-boundary: {n_cap_x}, z-boundary: {n_cap_z})")
    if predictor is not None:
        print(f"  predictor rows after this batch: {len(predictor.y)}")

    print(f"\nOutputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()