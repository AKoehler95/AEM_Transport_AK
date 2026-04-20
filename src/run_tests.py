#!/usr/bin/env python3
"""
AEM Solver Test Runner
======================
Batch-runs every JSON configuration in a directory through the AEM
simulation pipeline and collects the outputs into summary files.

Each test loads a config, runs the full simulation (solve, grid
evaluation, plots, statistics), and records the timing, solver
diagnostics, and key outputs (L_max, concentration range, etc.).

Outputs (in ./results/):
  test_results.json   — machine-readable per-test data
  test_summary.txt    — human-readable formatted table

Usage:
    python run_tests.py [config_dir]

Default config_dir is ./configs_test_batch/ relative to this script.
"""

import sys
import os
import json
import time
import threading
import traceback
import logging
import io
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "configs_test_batch")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Maximum wall-clock time per test before it is marked TIMEOUT.
TEST_TIMEOUT = 600

# Log capture
log_capture = io.StringIO()
log_handler = logging.StreamHandler(log_capture)
log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log_handler)


def fmt_f(v, spec=".2f"):
    """Format a numeric value with the given format spec, or '-' if None."""
    return f"{v:{spec}}" if v is not None else "-"


def _run_one_body(config_path, result, stdout_buf):
    """
    Execute a single configuration through the full simulation pipeline
    and populate the result dict with outputs and diagnostics.

    Runs inside a worker thread so the outer run_one() can enforce a
    timeout. Redirects sim.run()'s stdout/stderr into stdout_buf, then
    parses it for solver-method information (coupled vs decoupled,
    condition number, group count).
    """
    from at_config import ATConfiguration
    from at_simulation import ATSimulation

    config = ATConfiguration.from_json(config_path)
    result["num_elements"] = len(config.elements)
    result["orientation"]  = config.orientation

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

    # Parse solver diagnostics from captured stdout
    for ln in (stdout_text or "").splitlines():
        s = ln.strip()
        if s.startswith("Coupled solve: OK") and "cond=" in s:
            result["solve_method"] = "coupled"
            try:
                result["condition_number"] = float(
                    s.split("cond=")[1].strip().rstrip(")"))
            except Exception:
                pass
        elif "ill-conditioned" in s and "cond=" in s:
            result["solve_method"] = "decoupled"
            try:
                result["condition_number"] = float(
                    s.split("cond=")[1].split(")")[0].strip())
            except Exception:
                pass
        elif s.startswith("Split into"):
            try:
                result["num_groups"] = int(s.split()[2])
            except Exception:
                pass

    # Record simulation outputs
    result["L_max"]            = sim.L_max
    result["interface_length"] = getattr(sim, "interface_length", None)
    result["result_min"]       = float(np.min(sim.result))
    result["result_max"]       = float(np.max(sim.result))
    result["status"]           = "DONE"


def run_one(config_path):
    """
    Run a single test configuration with timeout enforcement.

    Spawns a worker thread to execute _run_one_body. If the thread
    doesn't complete within TEST_TIMEOUT seconds, the result is marked
    TIMEOUT. If the worker raises an exception, the result is marked
    ERROR with the exception details recorded.

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
        result_min=None,
        result_max=None,
        phase_timings=None,
        wall_time_s=None,
        error=None,
        stdout_capture="",
        log_capture="",
    )

    stdout_buf = io.StringIO()
    log_capture.truncate(0)
    log_capture.seek(0)
    thread_exc = [None]

    def worker():
        try:
            _run_one_body(config_path, result, stdout_buf)
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
        return result

    if thread_exc[0] is not None:
        e = thread_exc[0]
        result["status"] = "ERROR"
        result["error"]  = f"{type(e).__name__}: {e}"
        result["stdout_capture"] = stdout_buf.getvalue()
        result["log_capture"] = log_capture.getvalue()
        traceback.print_exception(type(e), e, e.__traceback__)

    return result


def write_summary(all_results, path):
    """
    Write a formatted text summary of all test results to `path`.

    Contains a header with timestamp and config directory, a table row
    per test with key diagnostics, and aggregate statistics over all
    completed tests at the bottom.
    """
    col_w = 42
    header = (
        f"{'Config':<{col_w}} {'St':>6} {'#El':>3} {'Ori':>5} "
        f"{'Cond':>10} {'Method':>10} {'L_max':>6} "
        f"{'ResMin':>8} {'ResMax':>8} {'Time':>7}"
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

        lines.append(
            f"{name:<{col_w}} "
            f"{r['status']:>6} "
            f"{r['num_elements_after_mirroring']:>3} "
            f"{r['orientation'][:5]:>5} "
            f"{cond_str:>10} "
            f"{(r['solve_method'] or '-')[:10]:>10} "
            f"{fmt_f(r['L_max']):>6} "
            f"{fmt_f(r['result_min'], '.2f'):>8} "
            f"{fmt_f(r['result_max'], '.2f'):>8} "
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

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """
    Discover all JSON configs, run each through the simulation, print
    progress to the console, and write summary files.
    """
    configs = sorted(
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR) if f.endswith(".json")
    )
    if not configs:
        print(f"No .json configs found in {CONFIG_DIR}")
        sys.exit(1)

    print(f"Running {len(configs)} tests from {CONFIG_DIR}")
    print(f"Results -> {RESULTS_DIR}\n")

    all_results = []
    for i, cfg_path in enumerate(configs, 1):
        name = os.path.basename(cfg_path)
        print(f"[{i:3d}/{len(configs)}] {name} ... ", end="", flush=True)

        result = run_one(cfg_path)
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
    print(f"\nOutputs in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
