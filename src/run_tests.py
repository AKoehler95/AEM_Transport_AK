#!/usr/bin/env python3
"""
Runs all test configs from ./configs/ and collects diagnostics.

Usage:
    cd <project_dir>           # where at_simulation.py lives
    python <path>/run_tests.py

The multiprocessing guard is essential — without it, spawned workers
try to re-import this script and crash.
"""

import sys
import os
import json
import time
import traceback
import logging
import io
import math
import re
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, "configs")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def setup():
    """Imports and logging — called only from __main__."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    PROJECT_DIR = os.getcwd()
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

    import matplotlib
    matplotlib.use("Agg")

    # Logging capture
    log_buf = io.StringIO()
    h = logging.StreamHandler(log_buf)
    h.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.DEBUG)
    return log_buf


def evaluate_boundary_accuracy(sim):
    """
    Evaluate concentration on each element boundary vs target.
    Image and zero-isoline elements are flagged separately since their
    C-space boundary value differs from elem.c by design.
    """
    from at_element import ATElementType

    results = []
    for idx, elem in enumerate(sim.config.elements):
        n_pts = 80

        # Classify element
        eid = (elem.id or "").lower()
        is_image = eid.startswith("image_")
        is_zero = eid.startswith("zero_")
        elem_type = "image" if is_image else ("zero" if is_zero else "real")

        if elem.kind == ATElementType.Circle:
            phi = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
            x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
            y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)

        elif elem.kind == ATElementType.Ellipse:
            t = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
            a, b = float(elem.r), float(elem.b if elem.b is not None else elem.r)
            ct, st = math.cos(elem.theta), math.sin(elem.theta)
            x_test = elem.x + a * np.cos(t) * ct - b * np.sin(t) * st
            y_test = elem.y + a * np.cos(t) * st + b * np.sin(t) * ct

        elif elem.kind == ATElementType.Line:
            s = np.linspace(-elem.r + 1e-9, elem.r - 1e-9, n_pts)
            x_test = elem.x + s * math.cos(elem.theta)
            y_test = elem.y + s * math.sin(elem.theta)

        else:
            continue

        conc_values = []
        for x, y in zip(x_test, y_test):
            try:
                conc_values.append(sim.calc_c(x, y))
            except Exception:
                conc_values.append(np.nan)

        arr = np.array(conc_values)
        valid = arr[~np.isnan(arr)]

        if len(valid) == 0:
            results.append(dict(elem_idx=idx, elem_id=elem.id, elem_type=elem_type,
                                target_c=elem.c,
                                mean_conc=np.nan, std_conc=np.nan,
                                mean_abs_error=np.nan, max_abs_error=np.nan))
            continue

        abs_err = np.abs(valid - elem.c)
        results.append(dict(
            elem_idx=idx,
            elem_id=elem.id,
            elem_type=elem_type,
            target_c=elem.c,
            mean_conc=round(float(np.mean(valid)), 6),
            std_conc=round(float(np.std(valid)), 6),
            mean_abs_error=round(float(np.mean(abs_err)), 6),
            max_abs_error=round(float(np.max(abs_err)), 6),
        ))
    return results


def run_single_test(config_path, log_buf):
    """Run one config and return diagnostics dict."""
    import matplotlib.pyplot as plt
    from at_config import ATConfiguration
    from at_simulation import ATSimulation

    result = dict(
        config=os.path.basename(config_path),
        status="UNKNOWN",
        num_elements=0,
        num_elements_after_mirroring=0,
        orientation="",
        condition_number=None,
        solve_method="",
        num_groups=None,
        decoupled_iterations=None,
        L_max=None,
        interface_length=None,
        result_min=None,
        result_max=None,
        conc_range_ok=None,
        boundary_stats=[],
        worst_boundary_error=None,
        wall_time_s=None,
        error=None,
    )

    stdout_buf = io.StringIO()
    log_buf.truncate(0)
    log_buf.seek(0)

    t0 = time.time()
    try:
        config = ATConfiguration.from_json(config_path)
        result["num_elements"] = len(config.elements)
        result["orientation"] = config.orientation

        sim = ATSimulation(config)

        with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
            sim.run()

        result["wall_time_s"] = round(time.time() - t0, 2)
        stdout_text = stdout_buf.getvalue()
        result["num_elements_after_mirroring"] = len(sim.config.elements)

        # ---- Parse solver prints ----
        for ln in stdout_text.split("\n"):
            s = ln.strip()
            if s.startswith("Coupled solve: OK"):
                result["solve_method"] = "coupled"
                m = re.search(r"cond=([\d.eE+\-]+)", s)
                if m:
                    try: result["condition_number"] = float(m.group(1))
                    except: pass

            elif s.startswith("Coupled solve failed validation"):
                result["solve_method"] = "decoupled"
                m = re.search(r"cond=([\d.eE+\-]+)", s)
                if m:
                    try: result["condition_number"] = float(m.group(1))
                    except: pass

            elif s.startswith("Coupled solve ill-conditioned"):
                result["solve_method"] = "decoupled"
                m = re.search(r"cond=([\d.eE+\-]+)", s)
                if m:
                    try: result["condition_number"] = float(m.group(1))
                    except: pass

            elif s.startswith("Cannot decouple"):
                result["solve_method"] = "coupled_fallback"

            elif s.startswith("Could not split further"):
                result["solve_method"] = "coupled_fallback"

            elif s.startswith("Decoupled into") or s.startswith("Split into"):
                m = re.search(r"(\d+) groups", s)
                if m:
                    result["num_groups"] = int(m.group(1))

            elif "converged after" in s:
                m = re.search(r"after (\d+)", s)
                if m:
                    result["decoupled_iterations"] = int(m.group(1))

            elif s.startswith("Decoupled solve:") and "iterations" in s:
                m = re.search(r"(\d+) iterations", s)
                if m:
                    result["decoupled_iterations"] = int(m.group(1))

        # ---- Grid results ----
        result["L_max"] = sim.L_max
        if sim.interface_length is not None:
            result["interface_length"] = round(sim.interface_length, 3)
        result["result_min"] = round(float(np.min(sim.result)), 4)
        result["result_max"] = round(float(np.max(sim.result)), 4)

        # ---- Conc range check ----
        elem_concs = [e.c for e in sim.config.elements]
        min_exp = min(-8.0, min(elem_concs))
        max_exp = max(elem_concs)
        tol = 0.01 * max(abs(min_exp), abs(max_exp))
        outside = np.sum((sim.result > max_exp + tol) |
                         (sim.result < min_exp - tol))
        pct = (outside / sim.result.size) * 100
        result["conc_range_ok"] = pct <= 1.0

        # ---- Boundary accuracy ----
        bstats = evaluate_boundary_accuracy(sim)
        result["boundary_stats"] = bstats

        # Only consider REAL elements (not image/zero) for pass/fail
        worst = 0.0
        for bs in bstats:
            if bs.get("elem_type", "real") != "real":
                continue  # skip image and zero-isoline elements
            v = bs["max_abs_error"]
            if v is not None and not np.isnan(v):
                worst = max(worst, v)
        result["worst_boundary_error"] = round(worst, 6)

        # ---- Overall status ----
        if result["conc_range_ok"] and worst < 1.0:
            result["status"] = "PASS"
        elif result["conc_range_ok"] and worst < 5.0:
            result["status"] = "MARGINAL"
        else:
            result["status"] = "FAIL"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["wall_time_s"] = round(time.time() - t0, 2)
        traceback.print_exc()

    finally:
        plt.close("all")

    return result


# =====================================================================
def fmt_cond(c):
    return "-" if c is None else f"{c:.1e}"

def fmt_f(v, f=".2f"):
    return "-" if v is None else f"{v:{f}}"


def main():
    log_buf = setup()

    config_files = sorted(
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR) if f.endswith(".json")
    )
    n = len(config_files)
    print(f"Found {n} test configurations in {CONFIG_DIR}")
    print("=" * 140)

    all_results = []

    for i, cfg_path in enumerate(config_files):
        name = os.path.basename(cfg_path).replace(".json", "")
        print(f"\n[{i+1}/{n}] {name} ...", end=" ", flush=True)

        r = run_single_test(cfg_path, log_buf)
        all_results.append(r)

        tag = {"PASS": "OK", "MARGINAL": "~~", "FAIL": "XX",
               "ERROR": "!!"}.get(r["status"], "??")
        print(f"{tag} {r['status']}  cond={fmt_cond(r['condition_number'])}  "
              f"method={r['solve_method']}  worst_bdy={fmt_f(r['worst_boundary_error'],'.4f')}  "
              f"L_max={fmt_f(r['L_max'])}  time={fmt_f(r['wall_time_s'])}s")

    # ==================== SUMMARY TABLE ====================
    print("\n\n" + "=" * 155)
    print("SUMMARY TABLE")
    print("=" * 155)
    hdr = (f"{'Test':<40} {'St':>4} {'#El':>3} {'Ori':>6} "
           f"{'Cond':>12} {'Method':>14} {'#Grp':>4} {'Iter':>4} "
           f"{'WorstBdyE':>12} {'ResMin':>10} {'ResMax':>10} "
           f"{'L_max':>6} {'ConcOK':>6} {'Time':>7}")
    print(hdr)
    print("-" * 155)

    for r in all_results:
        nm = r["config"].replace(".json", "")[:38]
        print(f"{nm:<40} "
              f"{r['status']:>4} "
              f"{r['num_elements_after_mirroring']:>3} "
              f"{r['orientation'][:6]:>6} "
              f"{fmt_cond(r['condition_number']):>12} "
              f"{r['solve_method']:>14} "
              f"{str(r['num_groups'] or '-'):>4} "
              f"{str(r['decoupled_iterations'] or '-'):>4} "
              f"{fmt_f(r['worst_boundary_error'], '.4f'):>12} "
              f"{fmt_f(r['result_min'], '.3f'):>10} "
              f"{fmt_f(r['result_max'], '.3f'):>10} "
              f"{fmt_f(r['L_max']):>6} "
              f"{str(r['conc_range_ok']):>6} "
              f"{fmt_f(r['wall_time_s'])}s")

    # ==================== FAILURES ====================
    failures = [r for r in all_results
                if r["status"] in ("FAIL", "ERROR", "MARGINAL")]
    if failures:
        print(f"\n\n{'='*120}")
        print(f"DETAILED FAILURE / MARGINAL REPORT ({len(failures)} cases)")
        print("=" * 120)
        for r in failures:
            print(f"\n--- {r['config']} [{r['status']}] ---")
            if r["error"]:
                print(f"  Error: {r['error']}")
            print(f"  Condition number: {fmt_cond(r['condition_number'])}")
            print(f"  Solve method:     {r['solve_method']}")
            print(f"  Result range:     [{fmt_f(r['result_min'],'.3f')}, "
                  f"{fmt_f(r['result_max'],'.3f')}]")
            print(f"  Conc range OK:    {r['conc_range_ok']}")
            print(f"  Boundary stats:")
            for bs in r["boundary_stats"]:
                etype = bs.get("elem_type", "real")
                marker = "" if etype == "real" else f" [{etype}]"
                print(f"    Elem {bs['elem_idx']} ({bs['elem_id']}){marker}: "
                      f"target={bs['target_c']}, mean={bs['mean_conc']}, "
                      f"mean_err={bs['mean_abs_error']}, "
                      f"max_err={bs['max_abs_error']}")

    # ==================== STATISTICS ====================
    print(f"\n\n{'='*80}")
    print("OVERALL STATISTICS")
    print("=" * 80)
    total = len(all_results)
    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Total: {total}")
    for s in ("PASS", "MARGINAL", "FAIL", "ERROR"):
        print(f"  {s:10s}: {counts.get(s, 0)}")

    coupled   = [r for r in all_results if r["solve_method"] == "coupled"]
    decoupled = [r for r in all_results if r["solve_method"] == "decoupled"]
    fallback  = [r for r in all_results if r["solve_method"] == "coupled_fallback"]

    print(f"\nSolve method distribution:")
    print(f"  Coupled:          {len(coupled)}")
    print(f"  Decoupled:        {len(decoupled)}")
    print(f"  Coupled fallback: {len(fallback)}")

    if coupled:
        conds = [r["condition_number"] for r in coupled
                 if r["condition_number"] is not None]
        if conds:
            print(f"\nCoupled-accepted condition numbers:")
            print(f"  Min: {min(conds):.2e}   Max: {max(conds):.2e}")

        # Show which coupled solves FAILED (accepted by threshold but bad result)
        bad_coupled = [r for r in coupled if r["status"] in ("FAIL", "MARGINAL")]
        if bad_coupled:
            print(f"\n  *** {len(bad_coupled)} coupled solves accepted "
                  f"but produced FAIL/MARGINAL results! ***")
            for r in bad_coupled:
                print(f"    {r['config']}: cond={fmt_cond(r['condition_number'])}, "
                      f"worst_bdy={fmt_f(r['worst_boundary_error'],'.4f')}")

    if decoupled:
        conds = [r["condition_number"] for r in decoupled
                 if r["condition_number"] is not None]
        if conds:
            print(f"\nDecoupled-trigger condition numbers:")
            print(f"  Min: {min(conds):.2e}   Max: {max(conds):.2e}")

    # ==================== THRESHOLD ANALYSIS ====================
    print(f"\n\n{'='*90}")
    print("CONDITION NUMBER vs DISTANCE (threshold analysis)")
    print("=" * 90)

    sweep_categories = [
        ("A01_2circ_horiz",    "A: Two circles horizontal"),
        ("B01_2line_horiz",    "B: Two lines horizontal"),
        ("C01_circ_line",      "C1: Circle + line horizontal"),
        ("C02_circ_ellipse",   "C2: Circle + ellipse horizontal"),
        ("D01_2circ_vert",     "D1: Two circles vertical (x-dist)"),
        ("D02_2line_vert",     "D2: Two lines vertical (x-dist)"),
        ("D03_2circ_vert_dy",  "D3: Two circles vertical (y-dist)"),
        ("E01_donor_acc",      "E: Donor + acceptor horizontal"),
    ]

    for prefix, label in sweep_categories:
        subset = [r for r in all_results
                  if r["config"].replace(".json", "").startswith(prefix)]
        if not subset:
            continue

        print(f"\n  {label}:")
        print(f"  {'Param':>8} {'Cond':>12} {'Method':>14} "
              f"{'Status':>8} {'WorstBdyE':>12} {'ResRange':>22}")

        for r in sorted(subset, key=lambda x: x["config"]):
            name = r["config"].replace(".json", "")
            # Extract the numeric suffix (d60, dx5, dy4 etc.)
            m = re.search(r'_d[xy]?(\d+)$', name)
            param = m.group(1) if m else "?"
            rng = (f"[{fmt_f(r['result_min'],'.1f')}, "
                   f"{fmt_f(r['result_max'],'.1f')}]")
            print(f"  {param:>8} {fmt_cond(r['condition_number']):>12} "
                  f"{r['solve_method']:>14} {r['status']:>8} "
                  f"{fmt_f(r['worst_boundary_error'], '.4f'):>12} "
                  f"{rng:>22}")

    # ==================== COMPOSITE SHAPE SUMMARY ====================
    print(f"\n\n{'='*90}")
    print("CLOSELY PACKED / COMPOSITE SHAPE RESULTS")
    print("=" * 90)
    composite = [r for r in all_results if r["config"].startswith("G")]
    if composite:
        print(f"  {'Test':<35} {'St':>4} {'#El':>3} {'Cond':>12} "
              f"{'Method':>14} {'WorstBdyE':>12} {'ResRange':>22}")
        for r in composite:
            nm = r["config"].replace(".json", "")[:33]
            rng = (f"[{fmt_f(r['result_min'],'.1f')}, "
                   f"{fmt_f(r['result_max'],'.1f')}]")
            print(f"  {nm:<35} {r['status']:>4} "
                  f"{r['num_elements_after_mirroring']:>3} "
                  f"{fmt_cond(r['condition_number']):>12} "
                  f"{r['solve_method']:>14} "
                  f"{fmt_f(r['worst_boundary_error'], '.4f'):>12} "
                  f"{rng:>22}")

    # ==================== SAVE JSON ====================
    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nFull results saved to: {json_path}")


# =====================================================================
# CRITICAL: This guard prevents spawned multiprocessing workers from
# re-executing the test suite.
# =====================================================================
if __name__ == "__main__":
    main()
