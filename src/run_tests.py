#!/usr/bin/env python3
"""
AEM Solver Test Suite
=====================
Runs all test configurations from ./configs/ and collects diagnostics.
Saves three output files to ./results/:
  - test_results.json      : full per-test data (machine-readable)
  - test_summary.txt       : formatted summary table + statistics (human-readable)
  - test_summary_fails.txt : only the failed/error/marginal tests with details

Usage:
    cd <project_dir>   # where at_simulation.py, at_config.py etc. live
    python <path_to>/run_tests.py [config_dir]
"""

import sys
import os
import json
import time
import signal
import traceback
import logging
import io
import math
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "configs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Per-test timeout (seconds)
TEST_TIMEOUT = 500

# Thresholds for PASS / MARGINAL / FAIL
BOUNDARY_ERROR_PASS     = 0.10   # relative error ≤ 10 % → PASS
BOUNDARY_ERROR_MARGINAL = 0.25   # 10–25 % → MARGINAL, > 25 % → FAIL
CONC_OUTSIDE_PCT_PASS   = 1.0    # ≤ 1 % of grid points outside expected range → OK


# Timeout helper
class TestTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TestTimeout(f"Test exceeded {TEST_TIMEOUT}s timeout")


# Log capture
log_capture = io.StringIO()
log_handler = logging.StreamHandler(log_capture)
log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log_handler)


# Result field helpers
def fmt_f(v, spec=".2f"):
    return f"{v:{spec}}" if v is not None else "-"

def fmt_cond(v):
    return f"{v:.2e}" if v is not None else "-"


# Single-test runner
def run_one(config_path):
    from at_config import ATConfiguration
    from at_simulation import ATSimulation

    result = dict(
        config                       = os.path.basename(config_path),
        status                       = "UNKNOWN",
        num_elements                 = 0,
        num_elements_after_mirroring = 0,
        orientation                  = "",
        condition_number             = None,
        solve_method                 = "",
        num_groups                   = None,
        group_sizes                  = None,
        decoupled_iterations         = None,
        L_max                        = None,
        interface_length             = None,
        result_min                   = None,
        result_max                   = None,
        conc_range_ok                = None,
        worst_boundary_error         = None,
        wall_time_s                  = None,
        error                        = None,
        stdout_capture               = "",
        log_capture                  = "",
    )

    stdout_buf = io.StringIO()
    log_capture.truncate(0)
    log_capture.seek(0)

    t0 = time.time()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(TEST_TIMEOUT)

    try:
        config = ATConfiguration.from_json(config_path)
        result["num_elements"]   = len(config.elements)
        result["orientation"]    = config.orientation

        sim = ATSimulation(config)
        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            sim.run()

        result["wall_time_s"] = round(time.time() - t0, 2)
        signal.alarm(0)

        stdout_text = stdout_buf.getvalue()
        result["stdout_capture"]               = stdout_text
        result["log_capture"]                  = log_capture.getvalue()
        result["num_elements_after_mirroring"] = len(sim.config.elements)

        # ── Parse solver stdout ──────────────────────────────────────────
        for ln in stdout_text.splitlines():
            s = ln.strip()
            if s.startswith("Coupled solve: OK"):
                result["solve_method"] = "coupled"
                if "cond=" in s:
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
            elif "iterations" in s and "rel_change" in s:
                try:
                    result["decoupled_iterations"] = int(
                        s.split("after")[1].split("iteration")[0].strip())
                except Exception:
                    pass

        # ── Simulation output ────────────────────────────────────────────
        result["L_max"]            = sim.L_max
        result["interface_length"] = getattr(sim, "interface_length", None)
        result["result_min"]       = float(np.min(sim.result))
        result["result_max"]       = float(np.max(sim.result))

        # ── Concentration range check ────────────────────────────────────
        elem_concs    = [e.c for e in sim.config.elements]
        max_expected  = max(elem_concs)
        min_expected  = min(-8.0, min(elem_concs))
        tol           = 0.01 * max(abs(min_expected), abs(max_expected))
        outside       = int(np.sum(
            (sim.result > max_expected + tol) |
            (sim.result < min_expected - tol)
        ))
        pct_outside   = outside / sim.result.size * 100
        result["conc_range_ok"] = pct_outside <= CONC_OUTSIDE_PCT_PASS

        # ── Boundary accuracy ────────────────────────────────────────────
        from at_element import ATElementType
        from math import cos, sin, pi

        worst_rel = 0.0
        for elem in sim.config.elements:
            if elem.id.lower().startswith("image_") or \
               elem.id.lower().startswith("zero_"):
                continue
            if elem.kind == ATElementType.Circle:
                phi    = np.linspace(0, 2*pi, 120, endpoint=False)
                x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
                y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)
            elif elem.kind == ATElementType.Line:
                s      = np.linspace(-elem.r + 1e-9, elem.r - 1e-9, 120)
                x_test = elem.x + s * cos(elem.theta)
                y_test = elem.y + s * sin(elem.theta)
            elif elem.kind == ATElementType.Ellipse:
                t      = np.linspace(0, 2*pi, 120, endpoint=False)
                b      = elem.b if elem.b is not None else elem.r
                ct, st = cos(elem.theta), sin(elem.theta)
                x_test = elem.x + elem.r*np.cos(t)*ct - b*np.sin(t)*st
                y_test = elem.y + elem.r*np.cos(t)*st + b*np.sin(t)*ct
            else:
                continue

            vals     = np.array([sim.calc_c(x, y)
                                 for x, y in zip(x_test, y_test)])
            mean_err = float(np.mean(np.abs(vals - elem.c)))
            denom    = max(abs(elem.c), 1e-10)
            rel_err  = mean_err / denom
            worst_rel = max(worst_rel, rel_err)

        result["worst_boundary_error"] = round(worst_rel, 6)

        # ── Status ───────────────────────────────────────────────────────
        if worst_rel <= BOUNDARY_ERROR_PASS and result["conc_range_ok"]:
            result["status"] = "PASS"
        elif worst_rel <= BOUNDARY_ERROR_MARGINAL:
            result["status"] = "MARGINAL"
        else:
            result["status"] = "FAIL"

    except TestTimeout as e:
        signal.alarm(0)
        result["status"]         = "TIMEOUT"
        result["error"]          = str(e)
        result["wall_time_s"]    = round(time.time() - t0, 2)
        result["stdout_capture"] = stdout_buf.getvalue()

    except Exception as e:
        signal.alarm(0)
        result["status"]         = "ERROR"
        result["error"]          = f"{type(e).__name__}: {e}"
        result["stdout_capture"] = stdout_buf.getvalue()
        result["log_capture"]    = log_capture.getvalue()
        traceback.print_exc()

    return result


# Summary writers
def write_summary(all_results, path, fails_only=False):
    """Write a human-readable summary table to a text file."""

    rows = [r for r in all_results
            if not fails_only or r["status"] not in ("PASS",)]

    STATUS_ICON = {"PASS": "OK", "MARGINAL": "~~", "FAIL": "XX",
                   "ERROR": "!!", "TIMEOUT": "TT", "UNKNOWN": "??"}

    col_w = 42

    header = (
        f"{'Config':<{col_w}} {'St':>4} {'#El':>3} {'Ori':>5} "
        f"{'Cond':>10} {'Method':>10} {'#Grp':>4} {'BdyErr':>8} "
        f"{'ResMin':>8} {'ResMax':>8} {'L_max':>6} "
        f"{'ConcOK':>6} {'Time':>7}"
    )
    sep = "-" * len(header)

    lines = []
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(f"AEM Test Suite — {'Failures only' if fails_only else 'Full summary'}")
    lines.append(f"Run at: {run_time}")
    lines.append(f"Config dir: {CONFIG_DIR}")
    lines.append(sep)

    # Counts
    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = len(all_results)
    lines.append(
        f"Total: {total}  |  "
        + "  ".join(f"{s}: {counts.get(s,0)}"
                    for s in ("PASS","MARGINAL","FAIL","ERROR","TIMEOUT"))
    )
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for r in rows:
        name = r["config"].replace(".json", "")
        if len(name) > col_w:
            name = name[:col_w-1] + "…"
        icon = STATUS_ICON.get(r["status"], "??")
        lines.append(
            f"{name:<{col_w}} "
            f"{icon:>4} "
            f"{r['num_elements_after_mirroring']:>3} "
            f"{r['orientation'][:5]:>5} "
            f"{fmt_cond(r['condition_number']):>10} "
            f"{(r['solve_method'] or '-')[:10]:>10} "
            f"{str(r['num_groups'] or '-'):>4} "
            f"{fmt_f(r['worst_boundary_error'], '.4f'):>8} "
            f"{fmt_f(r['result_min'], '.2f'):>8} "
            f"{fmt_f(r['result_max'], '.2f'):>8} "
            f"{fmt_f(r['L_max']):>6} "
            f"{str(r['conc_range_ok']):>6} "
            f"{fmt_f(r['wall_time_s'])}s"
        )

    lines.append(sep)

    # Statistics on numeric fields for PASS tests
    pass_results = [r for r in all_results if r["status"] == "PASS"]
    if pass_results:
        lines.append("\nStatistics over PASS tests:")
        for field, label in [
            ("worst_boundary_error", "Boundary error"),
            ("wall_time_s",          "Wall time [s]"),
            ("L_max",                "L_max [m]"),
        ]:
            vals = [r[field] for r in pass_results if r[field] is not None]
            if vals:
                lines.append(
                    f"  {label:<22}  min={min(vals):.4g}  "
                    f"max={max(vals):.4g}  mean={sum(vals)/len(vals):.4g}"
                )

    # Detail block for non-PASS
    failures = [r for r in all_results if r["status"] not in ("PASS",)]
    if failures and not fails_only:
        lines.append("\nNon-PASS details:")
        lines.append(sep)
        for r in failures:
            lines.append(f"\n[{r['status']}] {r['config']}")
            if r.get("error"):
                lines.append(f"  Error   : {r['error']}")
            lines.append(f"  BdyErr  : {fmt_f(r['worst_boundary_error'], '.4f')}")
            lines.append(f"  ConcOK  : {r['conc_range_ok']}")
            lines.append(f"  Method  : {r['solve_method']}  cond={fmt_cond(r['condition_number'])}")
            if r.get("stdout_capture"):
                # Include last 10 lines of stdout for context
                tail = "\n".join(r["stdout_capture"].splitlines()[-10:])
                lines.append(f"  Stdout (tail):\n    " +
                              tail.replace("\n", "\n    "))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# Main
def main():
    configs = sorted(
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".json")
    )
    if not configs:
        print(f"No .json configs found in {CONFIG_DIR}")
        sys.exit(1)

    print(f"Running {len(configs)} tests from {CONFIG_DIR}")
    print(f"Results will be written to {RESULTS_DIR}\n")

    all_results = []
    for i, cfg_path in enumerate(configs, 1):
        name = os.path.basename(cfg_path)
        print(f"[{i:3d}/{len(configs)}] {name} ... ", end="", flush=True)
        result = run_one(cfg_path)
        all_results.append(result)

        icon = {"PASS":"OK","MARGINAL":"~~","FAIL":"XX",
                "ERROR":"!!","TIMEOUT":"TT"}.get(result["status"],"??")
        print(f"{icon} {result['status']:<8}  "
              f"cond={fmt_cond(result['condition_number'])}  "
              f"bdy={fmt_f(result['worst_boundary_error'],'.4f')}  "
              f"t={fmt_f(result['wall_time_s'])}s")

    # Write outputs
    # 1. Raw JSON
    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    serialisable = []
    for r in all_results:
        row = {k: v for k, v in r.items()
               if k not in ("stdout_capture", "log_capture")}
        serialisable.append(row)
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    # 2. Full summary text
    summary_path = os.path.join(RESULTS_DIR, "test_summary.txt")
    write_summary(all_results, summary_path, fails_only=False)

    # 3. Failures-only summary text
    fails_path = os.path.join(RESULTS_DIR, "test_summary_fails.txt")
    write_summary(all_results, fails_path, fails_only=True)

    # Print counts to stdout
    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_results)} tests")
    for s in ("PASS", "MARGINAL", "FAIL", "ERROR", "TIMEOUT"):
        if counts.get(s, 0):
            print(f"  {s:<10} {counts[s]}")
    print(f"\nOutputs written to {RESULTS_DIR}/:")
    print(f"  test_results.json        (full per-test data)")
    print(f"  test_summary.txt         (formatted table, all tests)")
    print(f"  test_summary_fails.txt   (non-PASS tests only)")


if __name__ == "__main__":
    main()