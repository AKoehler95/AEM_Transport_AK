#!/usr/bin/env python3
import os
import sys

import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt

from at_config import ATConfiguration
from at_element import ATElement, ATElementType
from at_simulation import ATSimulation, create_mirrored_element

class TeeOutput:
    """Write to both file and console simultaneously"""

    def __init__(self, file_handle, console_handle):
        self.file_handle = file_handle
        self.console_handle = console_handle

    def write(self, text):
        self.file_handle.write(text)
        self.console_handle.write(text)
        self.file_handle.flush()
        self.console_handle.flush()

    def flush(self):
        self.file_handle.flush()
        self.console_handle.flush()


def compute_lmax(alpha, r, C0, Ca, gamma, orientation, elem_kind, target_Lmax):
    cfg = ATConfiguration.from_json("simulation_config.json")
    cfg.alpha_t = alpha
    cfg.ca = Ca
    cfg.gamma = gamma
    cfg.orientation = orientation
    cfg.dom_inc = 1.0
    if target_Lmax is not None:
        cfg.dom_xmax = max(cfg.dom_xmax, target_Lmax * 1.1)

    if orientation == "vertical":
        if elem_kind == "Circle":
            base = ATElement(ATElementType.Circle, x=0.0, y=-(r + 0.01), c=C0, r=r)
        elif elem_kind == "Line":
            base = ATElement(ATElementType.Line, x=0.0, y=-(r + 0.01), c=C0, r=r)
        else:
            raise ValueError(f"Unknown element type: {elem_kind}")
        img = create_mirrored_element(base)
        img.id = f"image_{base.id}"
        cfg.elements = [base, img]
        cfg.dom_ymax = 0.0
    else:
        if elem_kind == "Circle":
            base = ATElement(ATElementType.Circle, x=0.0, y=0.0, c=C0, r=r)
        else:
            base = ATElement(ATElementType.Line, x=0.0, y=0.0, c=C0, r=r)
        cfg.elements = [base]

    for e in cfg.elements:
        e.calc_d_q(cfg.alpha_t, cfg.alpha_l, cfg.beta)
        e.set_outline(cfg.num_cp)

    sim = ATSimulation(cfg)
    sim.solve_system(cfg.alpha_l, cfg.alpha_t, cfg.beta, cfg.gamma, cfg.ca, cfg.num_terms, cfg.num_cp)
    sim.conc_array(cfg.dom_xmin, cfg.dom_ymin, cfg.dom_xmax, cfg.dom_ymax, cfg.dom_inc)
    sim.calculate_lmax()
    return sim.L_max, sim


def save_statistics(sim, line_index, statsfile):
    """
    Save boundary error statistics to 'find_alpha_stats.txt'
    """
    phi = np.linspace(0, 2 * pi, 360)
    stats_file = statsfile

    with open(stats_file, "a") as f:
        f.write(f"\n=== Stats for Line {line_index} ===\n")
        for idx, elem in enumerate(sim.config.elements):
            if elem.kind == ATElementType.Circle:
                x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
                y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)
            elif elem.kind == ATElementType.Line:
                half_len = elem.r
                s = np.linspace(-half_len + 1e-9, half_len - 1e-9, 360)
                theta = elem.theta
                x_test = elem.x + s * cos(theta)
                y_test = elem.y + s * sin(theta)
            else:
                raise ValueError(f"Unknown element type: {elem.kind}")

            Err = [sim.calc_c(x, y) for x, y in zip(x_test, y_test)]
            min_val = round(np.min(Err), 9)
            max_val = round(np.max(Err), 9)
            mean_val = round(np.mean(Err), 9)
            std_val = round(np.std(Err), 9)

            f.write(f"Element {idx + 1}:\n")
            f.write(f"  Min = {min_val} mg/l\n")
            f.write(f"  Max = {max_val} mg/l\n")
            f.write(f"  Mean = {mean_val} mg/l\n")
            f.write(f"  Standard Deviation = {std_val} mg/l\n")


def find_alpha(
        radius, C0, Ca, gamma, target_Lmax,
        orientation="horizontal",
        elem_kind="Circle",
        alpha_start=0.02,
        step=0.001,
        tolerance=1e-3,
        max_alpha=0.2,
        max_stagnation=5
):
    def eval_L(a):
        at = a
        multiplier = 1
        count = 0
        while at <= max_alpha:
            try:
                L, sim = compute_lmax(at, radius, C0, Ca, gamma, orientation, elem_kind, target_Lmax)
                if L is not None:
                    return L, sim
            except Exception:
                pass
            at += step * multiplier
            count += 1
            if count > 5:
                multiplier *= 5
                count = 0
        raise RuntimeError(f"Cannot compute L at αₜ ∈ {a, max_alpha}")

    L_lo, sim = eval_L(alpha_start)
    print(f"L_max(⍺_min) = {L_lo}")
    if L_lo < target_Lmax:
        raise RuntimeError(f"L_max(⍺_min) = {L_lo} < target {target_Lmax}")
    elif abs(L_lo - target_Lmax) < tolerance:
        return alpha_start, L_lo, sim

    L_hi, sim = eval_L(max_alpha)
    print(f"L_max(⍺_max) = {L_hi}")
    if L_hi > target_Lmax:
        raise RuntimeError(f"L_max(⍺_max)={L_hi} > target {target_Lmax}")
    elif abs(L_hi - target_Lmax) < tolerance:
        return max_alpha, L_hi, sim

    a_lo, a_hi = 0.0, max_alpha
    prev_L, stagn = None, 0

    while True:
        a_mid = 0.5 * (a_lo + a_hi)
        L_mid, sim = eval_L(a_mid)
        diff = L_mid - target_Lmax
        print(f"[dbg] α_mid={a_mid:.6f}, L_mid={L_mid:.6f}, diff={diff:.6f}")

        if abs(diff) <= tolerance:
            print(f"[break] within tolerance |{L_mid}-{target_Lmax}|={abs(diff):.6f} ≤ {tolerance}")
            return a_mid, L_mid, sim

        if prev_L is not None and abs(L_mid - prev_L) < tolerance:
            stagn += 1
            print(f"[stagnate] count={stagn}")
            if stagn >= max_stagnation:
                raise RuntimeError(f"L stalled at {L_mid} for {max_stagnation} steps")
        else:
            stagn = 0

        prev_L = L_mid

        if diff > 0:
            a_lo = a_mid
        else:
            a_hi = a_mid

'''
def process_input_file(
        input_file,
        output_file,
        stats_file,
        orientation: str ="horizontal",
        elem_kind: str ="Circle",
        source_thickness_modifier: float =1.0,
        alpha_start: float = 0.02,
        step: float = 0.001,
        tolerance: float = 1e-3,
        max_alpha: float = 0.2,
        max_stagnation: int = 5
):
    open(stats_file, "w").close()
    lines = [ln.split() for ln in open(input_file) if ln.strip()]
    results, skipped = [], []

    for i, params in enumerate(lines, 1):
        r, C0, Ca, gamma, target = map(float, params)
        print("=" * 65)
        print(f"Processing line {i}: target_Lmax = {target}")
        try:
            α, L, sim = find_alpha(
                r * source_thickness_modifier, C0, Ca, gamma, target,
                orientation,
                elem_kind,
                alpha_start,
                step,
                tolerance,
                max_alpha,
                max_stagnation
            )

            warning = " [WARNING: αₜ > 0.1]" if α > 0.1 else ""
            results.append(
                f"Line {i}: aq = {r}, r={r * source_thickness_modifier: .3f}, C0={C0}, Ca={Ca}, γ={gamma}, "
                f"target={target} → αₜ={α:.6f}, Lmax={L:.6f}{warning}"
            )

            if sim is not None:
                save_statistics(sim, i, stats_file)

        except Exception as e:
            print(f"  [ERROR] {e}")
            skipped.append(f"Line {i}: params={params}  ERROR: {e}")

    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")
        f.write("\nSkipped Cases:\n")
        for line in skipped:
            f.write(line + "\n")
    print(f"Wrote {len(results)} results, {len(skipped)} skipped to '{output_file}'")
'''

def process_input_file_with_logging(
        input_file,
        output_file,
        statsfile,
        orientation="horizontal",
        elem_kind="Circle",
        source_thickness_modifier=1.0,
        alpha_start: float = 0.02,
        step: float = 0.001,
        tolerance: float = 1e-3,
        max_alpha: float = 0.2,
        max_stagnation: int = 5
):
    """Function that logs parameters and outputs per input line interleaved"""
    import datetime
    log_file = "findalpha_log.txt"

    with open(log_file, 'a') as f:
        # Add timestamp header for new run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write("\n" + "*" * 80 + "\n")
        f.write(f"NEW RUN STARTED AT: {timestamp}\n")
        f.write("*" * 80 + "\n")
        # Write parameters
        f.write("FINDALPHA MODE PARAMETERS:\n")
        f.write("=" * 80 + "\n")
        f.write(f"INPUT_FILE: {input_file}, OUTPUT_FILE: {output_file}, STATSFILE: {statsfile}\n")
        f.write(f", ORIENTATION: {orientation}, ELEM_KIND: {elem_kind}, SOURCE_THICKNESS_MODIFIER: {source_thickness_modifier}, \n")
        f.write(f" ALPHA_START: {alpha_start}, STEP: {step}, TOLERANCE: {tolerance}, MAX_ALPHA: {max_alpha}, MAX_STAGNATION: {max_stagnation}\n")

        # Set up tee output to write to both file and console
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            # Create tee objects
            tee_stdout = TeeOutput(f, original_stdout)
            tee_stderr = TeeOutput(f, original_stderr)
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            # Clear stats file
            open(statsfile, "w").close()
            lines = [ln.strip() for ln in open(input_file) if ln.strip()]
            results, skipped = [], []

            for i, line in enumerate(lines, 1):
                f.write("=" * 80 + "\n")
                f.write(f"Input Line {i}: {line}\n")
                params = list(map(float, line.split()))
                r, C0, Ca, gamma, target = params

                print("=" * 80)
                print(f"Processing line {i}: target_Lmax = {target}")
                try:
                    α, L, sim = find_alpha(
                        r * source_thickness_modifier, C0, Ca, gamma, target,
                        orientation,
                        elem_kind,
                        alpha_start,
                        step,
                        tolerance,
                        max_alpha,
                        max_stagnation
                    )

                    warning = " [WARNING: αₜ > 0.1]" if α > 0.1 else ""
                    result = (
                        f"Line {i}: aq = {r}, r={r * source_thickness_modifier: .3f}, C0={C0}, Ca={Ca}, γ={gamma}, "
                        f"target={target} → αₜ={α:.6f}, Lmax={L:.6f}{warning}"
                    )
                    results.append(result)

                    if sim is not None:
                        save_statistics(sim, i, statsfile)

                except Exception as e:
                    err_msg = f"Line {i}: params={params}  ERROR: {e}"
                    print(f"  [ERROR] {e}")
                    skipped.append(err_msg)

            with open(output_file, "w") as out_f:
                for line in results:
                    out_f.write(line + "\n")
                out_f.write("\nSkipped Cases:\n")
                for line in skipped:
                    out_f.write(line + "\n")
            print(f"Wrote {len(results)} results, {len(skipped)} skipped to '{output_file}'")

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
