#!/usr/bin/env python3
"""
AEM Solver Test Suite
=====================
Batch-runs every JSON configuration in a directory through the AEM simulation
pipeline and classifies each result as PASS / PASS_CAPPED / MARGINAL / FAIL /
ERROR / TIMEOUT based on physics-based validation metrics.

Each test:
  1. Loads the configuration via ATConfiguration.from_json
  2. Runs the full simulation (solve, grid eval, plots, stats)
  3. Validates the resulting concentration field against five criteria
  4. Records phase timings, solver diagnostics, and any failure reasons

Outputs three files in ./results/:
  test_results.json       — machine-readable per-test data
  test_summary.txt        — human-readable table + aggregate statistics
  test_summary_fails.txt  — same format, filtered to non-PASS tests

Validation criteria (see _validate_result for details):
  1. Concentration range    — grid values within physically expected bounds
  2. Source interior check  — mean concentration inside source elements ≈ C0
  3. Far-field decay        — concentration approaches -Ca at domain edges
  4. L_max plausibility     — plume not capped by the domain boundary
  5. Centerline monotonicity — no spurious oscillations downstream

These target the physical correctness of the result rather than whether
individual AEM basis functions satisfy their exact boundary conditions.
For composite shapes made of tightly packed sub-elements, per-element
boundary checks are unreliable (interpolation-limited at coarse grid
resolutions) but the aggregate plume can still be correct.

Usage:
    python run_tests.py [config_dir]

Default config_dir is ./configs_v8/ relative to this script.
"""

import sys, os, json, time, threading, traceback, logging, io, math
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

# Paths
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRIPT_DIR, "configs_v8")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Test-level parameters
# Maximum wall-clock time per test. Tests that exceed this are marked
# TIMEOUT and the worker thread is abandoned (its result is discarded).
TEST_TIMEOUT = 600

# Validation thresholds
# Mean relative error of interior-point concentrations vs. prescribed C0.
# Above this, the source zone is not being rendered correctly.
SOURCE_INTERIOR_REL_ERROR = 0.15

# Mean relative error of far-field concentrations vs. background (-Ca).
# Above this, the solution hasn't decayed properly at the domain edge.
FAR_FIELD_REL_ERROR = 0.10

# Maximum percentage of grid points allowed outside the physically
# expected concentration range. Values above indicate numerical blowup.
CONC_OUTSIDE_PCT_PASS = 2.0

# Maximum fraction of centerline points that may show concentration
# increasing downstream of the peak (would indicate solver oscillation).
MONOTONICITY_VIOLATIONS = 0.10

# L_max within this fraction of the domain width is treated as capped
# by the domain boundary — a configuration issue, not a solver failure.
LMAX_CAPPED_THRESHOLD = 0.95

# Log capture
# The at_simulation module uses Python logging for debug info; we
# redirect it to an in-memory buffer and attach it to each test's result.
log_capture = io.StringIO()
log_handler = logging.StreamHandler(log_capture)
log_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(log_handler)


def fmt_f(v, spec=".2f"):
    """Format a numeric value with the given format spec, or '-' if None."""
    return f"{v:{spec}}" if v is not None else "-"


def fmt_cond(v):
    """Format a condition number in scientific notation, or '-' if None."""
    return f"{v:.2e}" if v is not None else "-"


def _validate_result(sim, result):
    """
    Run five physics-based validation checks on a completed simulation
    and populate the result dict with diagnostics and a final status.

    Checks, in order:

      1. Concentration range — counts grid points whose concentration
         falls outside [min_element_c, max_element_c] (with -8 as a
         background floor). More than CONC_OUTSIDE_PCT_PASS percent
         outside flags a range violation.

      2. Source interior — for each source element, samples 12 points
         at 50% of the element's radius (well inside, not on boundary),
         looks up the interpolated concentration at each, and compares
         the mean to the prescribed c. A grid-resolution-tolerant way
         to verify that source zones actually carry their C0 value.

      3. Far-field decay — averages the concentration over the rightmost
         10% of the domain and compares to -Ca (background acceptor).
         Large deviations indicate the plume hasn't resolved or the
         solver diverged in the far field.

      4. L_max plausibility — flags results where L_max is within
         LMAX_CAPPED_THRESHOLD of the domain edge. These produce a
         PASS_CAPPED status (solver succeeded, domain was too small).

      5. Centerline monotonicity — extracts the concentration profile
         along the y = (average source y) line, finds the peak, and
         counts points downstream of the peak where concentration
         increases. Above MONOTONICITY_VIOLATIONS indicates solver
         oscillation.

    Status assignment:
      - PASS         : no issues
      - PASS_CAPPED  : only issue is L_max capped (domain sizing, not solver)
      - MARGINAL     : only far-field or L_max issues (peripheral problems)
      - FAIL         : interior or monotonicity or range issues
    """
    from scipy.interpolate import RegularGridInterpolator
    from at_element import ATElementType

    config = sim.config
    issues = []

    # 1. Concentration range
    elem_concs = [e.c for e in config.elements]
    max_expected = max(elem_concs)
    min_expected = min(-8.0, min(elem_concs))
    tol = 0.01 * max(abs(min_expected), abs(max_expected))
    outside = int(np.sum(
        (sim.result > max_expected + tol) | (sim.result < min_expected - tol)))
    pct_outside = outside / sim.result.size * 100
    result["conc_range_ok"] = pct_outside <= CONC_OUTSIDE_PCT_PASS
    result["conc_outside_pct"] = round(pct_outside, 3)
    if pct_outside > CONC_OUTSIDE_PCT_PASS:
        issues.append(f"conc_range:{pct_outside:.1f}%")

    # 2. Source interior check (interpolated from grid)
    # Sample points at 50% of each element's radius — these are well
    # inside the source, so interpolation artifacts at the boundary
    # don't pollute the measurement.
    interp = RegularGridInterpolator(
        (sim.yaxis, sim.xaxis), sim.result,
        method='linear', bounds_error=False, fill_value=np.nan)

    source_elements = [e for e in config.elements
                       if not (e.id or "").lower().startswith("image_")
                       and not (e.id or "").lower().startswith("zero_")]

    interior_errors = []
    for elem in source_elements:
        if elem.c == 0:
            continue  # skip pure zero-isoline elements (rare in practice)
        n_pts = 12
        if elem.kind == ATElementType.Circle:
            phi = np.linspace(0, 2*math.pi, n_pts, endpoint=False)
            x_pts = elem.x + elem.r * 0.5 * np.cos(phi)
            y_pts = elem.y + elem.r * 0.5 * np.sin(phi)
        elif elem.kind == ATElementType.Line:
            s = np.linspace(-elem.r * 0.8, elem.r * 0.8, n_pts)
            x_pts = elem.x + s * math.cos(elem.theta)
            y_pts = elem.y + s * math.sin(elem.theta)
        elif elem.kind == ATElementType.Ellipse:
            phi = np.linspace(0, 2*math.pi, n_pts, endpoint=False)
            a = elem.r * 0.5
            b = (elem.b or elem.r) * 0.5
            ct, st = math.cos(elem.theta), math.sin(elem.theta)
            x_pts = elem.x + a * np.cos(phi) * ct - b * np.sin(phi) * st
            y_pts = elem.y + a * np.cos(phi) * st + b * np.sin(phi) * ct
        else:
            continue
        pts = np.column_stack([y_pts, x_pts])
        vals = interp(pts)
        valid = ~np.isnan(vals)
        if np.any(valid):
            rel_err = abs(float(np.mean(vals[valid])) - elem.c) / max(abs(elem.c), 1e-10)
            interior_errors.append(rel_err)

    if interior_errors:
        result["source_interior_mean_rel"] = round(float(np.mean(interior_errors)), 4)
        result["source_interior_worst_rel"] = round(float(max(interior_errors)), 4)
        if result["source_interior_mean_rel"] > SOURCE_INTERIOR_REL_ERROR:
            issues.append(f"src_interior:{result['source_interior_mean_rel']:.3f}")

    # 3. Far-field decay
    ca = config.ca
    x_far = config.dom_xmin + 0.9 * (config.dom_xmax - config.dom_xmin)
    x_idx = np.searchsorted(sim.xaxis, x_far)
    if x_idx < len(sim.xaxis):
        far = sim.result[:, x_idx:]
        mean_far = float(np.mean(far))
        expected = -ca
        far_err = abs(mean_far - expected) / max(abs(expected), 1e-10)
        result["far_field_rel_err"] = round(far_err, 4)
        if far_err > FAR_FIELD_REL_ERROR:
            issues.append(f"far_field:{far_err:.3f}")

    # 4. L_max capped
    result["L_max"] = sim.L_max
    lmax_capped = False
    if sim.L_max is not None:
        dw = config.dom_xmax - config.dom_xmin
        if (sim.L_max - config.dom_xmin) >= LMAX_CAPPED_THRESHOLD * dw:
            lmax_capped = True
            issues.append("L_max_capped")
    result["lmax_capped"] = lmax_capped

    # 5. Centerline monotonicity
    # Extract concentration along y = mean(source_y), find the downstream
    # peak, and count significant positive slopes past the peak.
    source_ys = [e.y for e in source_elements if e.c > 0]
    if source_ys and len(sim.xaxis) > 10:
        y_center = np.mean(source_ys)
        y_idx = np.argmin(np.abs(sim.yaxis - y_center))
        centerline = sim.result[y_idx, :]
        peak_idx = np.argmax(centerline)
        downstream = centerline[peak_idx:]
        if len(downstream) > 5:
            diffs = np.diff(downstream)
            threshold = 0.01 * max(abs(float(np.max(downstream))), 1e-10)
            violations = np.sum(diffs > threshold)
            v_pct = violations / max(len(diffs), 1) * 100
            result["monotonicity_violations_pct"] = round(v_pct, 2)
            if v_pct > MONOTONICITY_VIOLATIONS * 100:
                issues.append(f"monotonicity:{v_pct:.1f}%")

    # Status determination
    result["validation_issues"] = issues
    if not issues:
        result["status"] = "PASS"
    elif len(issues) == 1 and lmax_capped:
        result["status"] = "PASS_CAPPED"
    elif all("far_field" in i or "lmax" in i.lower() for i in issues):
        result["status"] = "MARGINAL"
    else:
        result["status"] = "FAIL"


def _run_one_body(config_path, result, stdout_buf):
    """
    Execute a single configuration through the full simulation pipeline
    and populate the result dict with all diagnostic fields.

    Runs inside a worker thread so the outer run_one() can enforce a
    timeout. Redirects sim.run()'s stdout/stderr into stdout_buf to keep
    it out of the parent process's console; the captured text is then
    parsed for solver-method information (coupled vs decoupled, condition
    number, group count) and attached to the result.

    After the simulation completes, calls _validate_result to apply the
    physics-based checks and set result["status"].
    """
    from at_config import ATConfiguration
    from at_simulation import ATSimulation

    config = ATConfiguration.from_json(config_path)
    result["num_elements"] = len(config.elements)
    result["orientation"]  = config.orientation

    sim = ATSimulation(config)
    with redirect_stdout(stdout_buf), redirect_stderr(stdout_buf):
        sim.run()

    result["stdout_capture"] = stdout_buf.getvalue()
    result["log_capture"]    = log_capture.getvalue()
    result["num_elements_after_mirroring"] = len(sim.config.elements)

    if hasattr(sim, 'timer') and hasattr(sim.timer, 'phases'):
        result["phase_timings"] = {k: round(v, 3) for k, v in sim.timer.phases.items()}

    # Extract solver diagnostics from the captured stdout.
    for ln in (result["stdout_capture"] or "").splitlines():
        s = ln.strip()
        if s.startswith("Coupled solve: OK") and "cond=" in s:
            result["solve_method"] = "coupled"
            try: result["condition_number"] = float(s.split("cond=")[1].strip().rstrip(")"))
            except: pass
        elif "ill-conditioned" in s and "cond=" in s:
            result["solve_method"] = "decoupled"
            try: result["condition_number"] = float(s.split("cond=")[1].split(")")[0].strip())
            except: pass
        elif s.startswith("Split into"):
            try: result["num_groups"] = int(s.split()[2])
            except: pass

    result["interface_length"] = getattr(sim, "interface_length", None)
    result["result_min"] = float(np.min(sim.result))
    result["result_max"] = float(np.max(sim.result))

    _validate_result(sim, result)


def run_one(config_path):
    """
    Run a single test configuration with timeout enforcement.

    Spawns a worker thread to execute _run_one_body. If the thread
    doesn't complete within TEST_TIMEOUT seconds, the result is marked
    TIMEOUT and the worker is abandoned (Python threads cannot be
    forcibly killed — the OS will reclaim resources when the main
    process exits). If the worker raises, the result is marked ERROR
    with the exception type and message recorded.

    Returns the result dict with all fields populated.
    """
    result = dict(
        config=os.path.basename(config_path), status="UNKNOWN",
        num_elements=0, num_elements_after_mirroring=0, orientation="",
        condition_number=None, solve_method="", num_groups=None,
        L_max=None, interface_length=None, result_min=None, result_max=None,
        conc_range_ok=None, conc_outside_pct=None,
        source_interior_worst_rel=None, source_interior_mean_rel=None,
        far_field_rel_err=None, lmax_capped=None,
        monotonicity_violations_pct=None, validation_issues=None,
        phase_timings=None, wall_time_s=None,
        error=None, stdout_capture="", log_capture="",
    )

    stdout_buf = io.StringIO()
    log_capture.truncate(0); log_capture.seek(0)
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
        result["error"] = f"Exceeded {TEST_TIMEOUT}s"
        result["stdout_capture"] = stdout_buf.getvalue()
        return result
    if thread_exc[0] is not None:
        e = thread_exc[0]
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["stdout_capture"] = stdout_buf.getvalue()
        result["log_capture"] = log_capture.getvalue()
        traceback.print_exception(type(e), e, e.__traceback__)
    return result


def write_summary(all_results, path, fails_only=False):
    """
    Write a formatted text summary of all test results to `path`.

    The summary contains:
      - A header with timestamp, config directory, and status counts
      - A table row per test showing: config name, status icon, element
        count, orientation, source/far-field error, monotonicity %,
        L_max, capped flag, and wall time
      - A detailed breakdown of non-PASS tests (when fails_only=False)
        listing error messages, validation issues, and phase timings

    If fails_only is True, only rows with status other than PASS or
    PASS_CAPPED are included — useful for a quick failures-only file.
    """
    rows = [r for r in all_results
            if not fails_only or r["status"] not in ("PASS", "PASS_CAPPED")]
    STATUS_ICON = {"PASS":"OK","PASS_CAPPED":"OC","MARGINAL":"~~",
                   "FAIL":"XX","ERROR":"!!","TIMEOUT":"TT","UNKNOWN":"??"}
    col_w = 42
    header = (f"{'Config':<{col_w}} {'St':>4} {'#El':>3} {'Ori':>5} "
              f"{'SrcErr':>7} {'FarErr':>7} {'Mono%':>6} {'L_max':>6} "
              f"{'Capped':>6} {'Time':>7}")
    sep = "-" * len(header)
    lines = []
    lines.append(f"AEM Test Suite — {'Failures' if fails_only else 'Full summary'}")
    lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Config dir: {CONFIG_DIR}")
    lines.append(sep)

    counts = {}
    for r in all_results: counts[r["status"]] = counts.get(r["status"], 0) + 1
    lines.append(f"Total: {len(all_results)}  |  " +
        "  ".join(f"{s}:{counts.get(s,0)}" for s in
                  ("PASS","PASS_CAPPED","MARGINAL","FAIL","ERROR","TIMEOUT")))
    lines.append(sep); lines.append(header); lines.append(sep)

    for r in rows:
        name = r["config"].replace(".json","")
        if len(name) > col_w: name = name[:col_w-1] + "…"
        icon = STATUS_ICON.get(r["status"], "??")
        lines.append(
            f"{name:<{col_w}} {icon:>4} "
            f"{r['num_elements_after_mirroring']:>3} "
            f"{r['orientation'][:5]:>5} "
            f"{fmt_f(r.get('source_interior_mean_rel'),'.4f'):>7} "
            f"{fmt_f(r.get('far_field_rel_err'),'.4f'):>7} "
            f"{fmt_f(r.get('monotonicity_violations_pct'),'.1f'):>6} "
            f"{fmt_f(r.get('L_max')):>6} "
            f"{'YES' if r.get('lmax_capped') else 'no':>6} "
            f"{fmt_f(r['wall_time_s'])}s")
    lines.append(sep)

    # Non-PASS details: error messages, validation issues, phase timings
    failures = [r for r in all_results if r["status"] not in ("PASS","PASS_CAPPED")]
    if failures and not fails_only:
        lines.append("\nNon-PASS details:")
        for r in failures:
            lines.append(f"\n[{r['status']}] {r['config']}")
            if r.get("error"): lines.append(f"  Error: {r['error']}")
            if r.get("validation_issues"): lines.append(f"  Issues: {r['validation_issues']}")
            if r.get("phase_timings"): lines.append(f"  Phases: {r['phase_timings']}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    """
    Test suite entry point.

    Discovers every .json file in CONFIG_DIR, runs each through run_one,
    prints a one-line status update per test to the console (including
    phase timings and any validation issues for quick diagnosis), then
    writes three output files: a machine-readable JSON dump and two
    human-readable text summaries (full and failures-only).

    Console output is flushed after every test so long-running suites
    remain visible to the user in real time, even when individual tests
    take many seconds.
    """
    configs = sorted(os.path.join(CONFIG_DIR, f)
                     for f in os.listdir(CONFIG_DIR) if f.endswith(".json"))
    if not configs:
        print(f"No .json configs found in {CONFIG_DIR}"); sys.exit(1)

    print(f"Running {len(configs)} tests from {CONFIG_DIR}")
    print(f"Results -> {RESULTS_DIR}\n")

    STATUS_ICON = {"PASS":"OK","PASS_CAPPED":"OC","MARGINAL":"~~",
                   "FAIL":"XX","ERROR":"!!","TIMEOUT":"TT"}

    all_results = []
    for i, cfg_path in enumerate(configs, 1):
        name = os.path.basename(cfg_path)
        print(f"[{i:3d}/{len(configs)}] {name} ... ", end="", flush=True)
        result = run_one(cfg_path)
        all_results.append(result)

        icon = STATUS_ICON.get(result["status"], "??")

        # Compact phase breakdown for inline diagnosis
        phase_str = ""
        if result.get("phase_timings"):
            pt = result["phase_timings"]
            phase_str = (f"  solve={pt.get('solve_system',0):.1f}s "
                         f"grid={pt.get('conc_array',0):.1f}s "
                         f"stats={pt.get('print_statistics',0):.1f}s")

        issues_str = ""
        if result.get("validation_issues"):
            issues_str = f"  [{', '.join(result['validation_issues'])}]"

        print(f"{icon} {result['status']:<12} t={fmt_f(result['wall_time_s'])}s"
              f"{phase_str}{issues_str}", flush=True)

    # Write JSON output (excluding the verbose stdout/log capture fields)
    json_path = os.path.join(RESULTS_DIR, "test_results.json")
    with open(json_path, "w") as f:
        json.dump([{k:v for k,v in r.items() if k not in ("stdout_capture","log_capture")}
                   for r in all_results], f, indent=2, default=str)

    write_summary(all_results, os.path.join(RESULTS_DIR, "test_summary.txt"))
    write_summary(all_results, os.path.join(RESULTS_DIR, "test_summary_fails.txt"), True)

    # Final status tally to the console
    counts = {}
    for r in all_results: counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\n{'='*60}\nRESULTS: {len(all_results)} tests")
    for s in ("PASS","PASS_CAPPED","MARGINAL","FAIL","ERROR","TIMEOUT"):
        if counts.get(s,0): print(f"  {s:<12} {counts[s]}")


if __name__ == "__main__":
    main()