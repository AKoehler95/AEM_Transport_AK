#!/usr/bin/env python3
from traceback import print_tb

import matplotlib.pyplot as plt

from at_config    import ATConfiguration
from at_element   import ATElement, ATElementType
from at_simulation import ATSimulation, create_mirrored_element


def compute_lmax(alpha, r, C0, Ca, gamma, orientation, target_Lmax):
    """
    Run ATSimulation up through calculate_lmax() and return sim.L_max.
    Automatically enlarges dom_xmax to 1.1×target if provided.
    """
    cfg = ATConfiguration.from_json("simulation_config.json")
    cfg.alpha_t     = alpha
    cfg.ca          = Ca
    cfg.gamma       = gamma
    cfg.orientation = orientation
    cfg.dom_inc     = 1.0
    if target_Lmax is not None:
        cfg.dom_xmax = max(cfg.dom_xmax, target_Lmax * 1.1)

    # place element(s)
    if orientation == "vertical":
        base = ATElement(ATElementType.Circle, x=0.0, y=-(r+0.1), c=C0, r=r)
        img  = create_mirrored_element(base)
        img.id = f"image_{base.id}"
        cfg.elements = [base, img]
        cfg.dom_ymax  = 0.0
    else:
        base = ATElement(ATElementType.Circle, x=0.0, y=0.0, c=C0, r=r)
        cfg.elements = [base]

    # solve system
    for e in cfg.elements:
        e.calc_d_q(cfg.alpha_t, cfg.alpha_l, cfg.beta)
        e.set_outline(cfg.num_cp)

    sim = ATSimulation(cfg)
    sim.solve_system(
        cfg.alpha_l, cfg.alpha_t, cfg.beta,
        cfg.gamma, cfg.ca,
        cfg.num_terms, cfg.num_cp
    )
    sim.conc_array(
        cfg.dom_xmin, cfg.dom_ymin,
        cfg.dom_xmax, cfg.dom_ymax,
        cfg.dom_inc
    )
    sim.calculate_lmax()
    return sim.L_max


def find_alpha(
    r, C0, Ca, gamma, target_Lmax,
    orientation="horizontal",
    alpha_start=0.02,
    step=0.001,
    tolerance=1e-3,
    max_alpha=0.2,
    max_stagnation=5
):
    """
    Solve L_max(α)=target_Lmax by monotonic bisection on [0, max_alpha]:
      • Tolerance break when |L_mid − target_Lmax| ≤ tolerance
      • Stagnation break if L_mid changes < tolerance for max_stagnation steps
      • Crash‐nudge by stepping α by step until compute succeeds
    """

    def eval_L(a):
        at = a
        while at <= max_alpha:
            try:
                L = compute_lmax(at, r, C0, Ca, gamma,
                                 orientation, target_Lmax)
                if L is not None:
                    return L
            except Exception:
                pass
            at += step
        raise RuntimeError(f"Cannot compute L at αₜ ∈ {a, max_alpha}")

    # endpoints
    L_lo = eval_L(alpha_start)
    print(f"L_max(⍺_min) = {L_lo}")
    if L_lo < target_Lmax:
        raise RuntimeError(f"L_max(⍺_min) = {L_lo} < target {target_Lmax}")
    elif abs(L_lo- target_Lmax) < tolerance:
        return alpha_start, L_lo

    L_hi = eval_L(max_alpha)
    print(f"L_max(⍺_max) = {L_hi}")
    if L_hi > target_Lmax:
        raise RuntimeError(f"L_max(⍺_max)={L_hi} > target {target_Lmax}")
    elif abs(L_hi- target_Lmax) < tolerance:
        return max_alpha, L_hi

    a_lo, a_hi = 0.0, max_alpha
    prev_L, stagn = None, 0

    while True:
        a_mid = 0.5*(a_lo + a_hi)
        L_mid = eval_L(a_mid)
        diff  = L_mid - target_Lmax
        print(f"[dbg] α_mid={a_mid:.6f}, L_mid={L_mid:.6f}, diff={diff:.6f}")

        # tolerance break
        if abs(diff) <= tolerance:
            print(f"[break] within tolerance |{L_mid}-{target_Lmax}|={abs(diff):.6f} ≤ {tolerance}")
            return a_mid, L_mid

        # stagnation check
        if prev_L is not None and abs(L_mid - prev_L) < tolerance:
            stagn += 1
            print(f"[stagnate] count={stagn}")
            if stagn >= max_stagnation:
                raise RuntimeError(f"L stalled at {L_mid} for {max_stagnation} steps")
        else:
            stagn = 0

        prev_L = L_mid

        # monotonic bisection
        if diff > 0:
            a_lo = a_mid
        else:
            a_hi = a_mid

def process_input_file(
    input_file, output_file,
    orientation="horizontal",
    alpha_start=0.02,
    step=0.001,
    tolerance=1e-3,
    max_alpha=0.2,
    max_stagnation=5
):
    """
    Read lines 'r C0 Ca gamma target_Lmax' from input_file,
    run find_alpha, and write 'αₜ,L_max' to output_file.
    """
    lines = [ln.split() for ln in open(input_file) if ln.strip()]
    results, skipped = [], []

    for i, params in enumerate(lines, 1):
        r, C0, Ca, gamma, target = map(float, params)
        print("="*65)
        print(f"Processing line {i}: target_Lmax = {target}")
        try:
            α, L = find_alpha(
                r, C0, Ca, gamma, target,
                orientation,
                alpha_start,
                step,
                tolerance,
                max_alpha,
                max_stagnation
            )
            warning = " [WARNING: αₜ > 0.1]" if α > 0.1 else ""
            results.append(
                f"Line {i}: r={r}, C0={C0}, Ca={Ca}, γ={gamma}, "
                f"target={target} → αₜ={α:.6f}, Lmax={L:.6f}{warning}"
            )
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
