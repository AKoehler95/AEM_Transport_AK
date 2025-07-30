#!/usr/bin/env python3
import os

import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt

from at_config import ATConfiguration
from at_element import ATElement, ATElementType
from at_simulation import ATSimulation, create_mirrored_element


def compute_lmax(alpha, r, C0, Ca, gamma, orientation, target_Lmax):
    cfg = ATConfiguration.from_json("simulation_config.json")
    cfg.alpha_t = alpha
    cfg.ca = Ca
    cfg.gamma = gamma
    cfg.orientation = orientation
    cfg.dom_inc = 1.0
    if target_Lmax is not None:
        cfg.dom_xmax = max(cfg.dom_xmax, target_Lmax * 1.1)

    if orientation == "vertical":
        base = ATElement(ATElementType.Circle, x=0.0, y=-(r + 0.1), c=C0, r=r)
        img = create_mirrored_element(base)
        img.id = f"image_{base.id}"
        cfg.elements = [base, img]
        cfg.dom_ymax = 0.0
    else:
        base = ATElement(ATElementType.Circle, x=0.0, y=0.0, c=C0, r=r)
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
                L, sim = compute_lmax(at, radius, C0, Ca, gamma, orientation, target_Lmax)
                if L is not None:
                    return L, sim
            except Exception:
                pass
            at += step * multiplier
            count +=1
            if count > 5:
                multiplier *= 5
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


def process_input_file(
    input_file,
    output_file,
    statsfile,
    orientation="horizontal",
    source_thickness_modifier = 1.0,
    alpha_start: float=0.02,
    step: float=0.001,
    tolerance: float=1e-3,
    max_alpha: float=0.2,
    max_stagnation: int=5
):
    open(statsfile, "w").close()
    lines = [ln.split() for ln in open(input_file) if ln.strip()]
    results, skipped = [], []

    for i, params in enumerate(lines, 1):
        r, C0, Ca, gamma, target = map(float, params)
        print("=" * 65)
        print(f"Processing line {i}: target_Lmax = {target}")
        try:
            α, L, sim = find_alpha(
                r*source_thickness_modifier, C0, Ca, gamma, target,
                orientation,
                alpha_start,
                step,
                tolerance,
                max_alpha,
                max_stagnation
            )

            warning = " [WARNING: αₜ > 0.1]" if α > 0.1 else ""
            results.append(
                f"Line {i}: aq = {r}, r={r*source_thickness_modifier: .3f}, C0={C0}, Ca={Ca}, γ={gamma}, "
                f"target={target} → αₜ={α:.6f}, Lmax={L:.6f}{warning}"
            )

            if sim is not None:
                save_statistics(sim, i, statsfile)

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
