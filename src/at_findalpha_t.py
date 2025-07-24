#!/usr/bin/env python3
import sys
import argparse
import matplotlib.pyplot as plt_temp

from at_config import ATConfiguration
from at_element import ATElement, ATElementType
from at_simulation import ATSimulation, create_mirrored_element


def compute_lmax_with_sim(
    alpha_t, r, C0, Ca, gamma,
    orientation="horizontal",
    target_Lmax=None      # ← take the target in here
):
    # 1) load JSON config
    config = ATConfiguration.from_json("simulation_config.json")

    # 2) override chemistry & orientation
    config.alpha_t     = alpha_t
    config.ca          = Ca
    config.gamma       = gamma
    config.orientation = orientation
    config.dom_inc = 1.0

    # 3) ensure the domain is big enough to capture the plume
    if target_Lmax is not None:
        # add a 10% margin
        config.dom_xmax = max(config.dom_xmax, target_Lmax * 1.1) #add tolerance

    # 4) build your element(s) exactly as before...
    if orientation == "vertical":
        base = ATElement(ATElementType.Circle, x=0.0, y=-(r+0.1), c=C0, r=r)
        mirror = create_mirrored_element(base)
        mirror.id = f"image_{base.id}"
        config.elements = [base, mirror]
        config.dom_ymax = 0.0
    else:
        base = ATElement(ATElementType.Circle, x=0.0, y=0.0, c=C0, r=r)
        config.elements = [base]

    # 5) precompute outlines & solve
    for e in config.elements:
        e.calc_d_q(config.alpha_t, config.alpha_l, config.beta)
        e.set_outline(config.num_cp)

    sim = ATSimulation(config)
    sim.solve_system(
        config.alpha_l, config.alpha_t, config.beta,
        config.gamma, config.ca,
        config.num_terms, config.num_cp
    )
    sim.conc_array(
        config.dom_xmin, config.dom_ymin,
        config.dom_xmax, config.dom_ymax,
        config.dom_inc
    )

    # 6) now delegate to calculate_lmax()
    sim.calculate_lmax()
    return sim.L_max

    #contour = plt.contour(sim.result, levels=[0])
    #paths = contour.get_paths()
    #plt.close()  # close the temporary figure
    #if not paths or len(paths[0].vertices) == 0:
    #    return None

    #Lmax_val = int(paths[0].vertices[:, 0].max() * config.dom_inc)
    #return Lmax_val

def find_alpha_t(
    target_Lmax: float,
    r: float,
    C0: float,
    Ca: float,
    gamma: float,
    orientation: str,
    alpha_start: float,
    step: float,
    tolerance: float,
    max_alpha: float,
    max_stagnation: int
) -> tuple[float, float]:
    """
    Find αₜ so L_max ≈ target_Lmax via:
      1) Bracket with stepping (and early breaks on undershoot or tolerance)
      2) Bisection refinement (with tolerance, undershoot, stagnation)
    """
    print("="*65)
    print(f"target_Lmax = {target_Lmax}")

    def safe_lmax(a: float):
        at = a
        while at <= max_alpha:
            try:
                L = compute_lmax_with_sim(
                    at, r, C0, Ca, gamma,
                    orientation, target_Lmax=target_Lmax
                )
                if L is not None:
                    print (f"Safe L_max has been found at L, a= {L}, {at}")
                    return L, at
            except Exception:
                pass
            print(f"Safe L_max has not been found for alpha = {at}")
            at += 0.01
        raise RuntimeError(f"Cannot eval L_max for αₜ∈[{a}, {max_alpha}]")

    def calc_lmax(a: float):
        at = a
        while at <= max_alpha:
            try:
                L = compute_lmax_with_sim(
                    at, r, C0, Ca, gamma,
                    orientation, target_Lmax=target_Lmax
                )
                if L is not None:
                    return L, at
            except Exception:
                pass
            at += 0.01
        raise RuntimeError(f"Cannot eval L_max for αₜ∈[{a}, {max_alpha}]")

    # ─── 1) Bracket phase ────────────────────────────────────────────────
    L_lo, a_lo = safe_lmax(alpha_start)
    # early break on starting value
    if L_lo < target_Lmax or abs(L_lo - target_Lmax) <= tolerance:
        print("Tolerance Break or Undershoot Break")
        return a_lo, L_lo

    a_hi = a_lo + step
    while a_hi <= max_alpha:
        L_hi, a_hi = calc_lmax(a_hi)

        # early tolerance break
        if abs(L_hi - target_Lmax) <= tolerance:
            print(f"Tolerance Break")
            return a_hi, L_hi

        # classic bracket check
        if (L_lo - target_Lmax) * (L_hi - target_Lmax) <= 0:
            print(f"Bracket found between {a_lo} and {a_hi}.")
            break

        L_lo, a_lo = L_hi, a_hi
        a_hi += step
    else:
        raise RuntimeError(
            f"Failed to bracket target {target_Lmax:.6f} in αₜ∈[{alpha_start}, {max_alpha}]"
        )

    # ─── 2) Bisection + stagnation + tolerance + undershoot ─────────────
    prev_L   = None
    stagnate = 0

    while True:
        a_mid = 0.5 * (a_lo + a_hi)
        if a_mid > max_alpha:
            raise RuntimeError(f"αₜ exceeded max_alpha={max_alpha}")

        L_mid, a_mid = calc_lmax(a_mid)
        diff = L_mid - target_Lmax
        print(f"[dbg] α_mid={a_mid:.6f}, L_mid={L_mid:.6f}, diff={diff:.6f}")

        # tolerance break
        if abs(diff) <= tolerance:
            print(f"[break] within tolerance |{L_mid}-{target_Lmax}|={abs(diff):.6f} ≤ {tolerance}")
            return a_mid, L_mid

        # stagnation break
        if prev_L is not None and abs(L_mid - prev_L) < tolerance:
            stagnate += 1
            if stagnate >= max_stagnation:
                raise RuntimeError(
                    f"L_max stagnated at {L_mid:.6f} for {max_stagnation} steps"
                )
        else:
            stagnate = 0

        prev_L = L_mid

        # narrow bracket
        if (L_lo - target_Lmax) * (L_mid - target_Lmax) <= 0:
            a_hi, L_hi = a_mid, L_mid
        else:
            a_lo, L_lo = a_mid, L_mid


def process_input_file(
    input_file: str,
    output_file: str,
    orientation: str,
    alpha_start: float,
    step: float,
    tolerance: float,
    max_alpha: float,
    max_stagnation: int
) -> None:
    """
    Reads rows of 'r C0 Ca gamma target_Lmax' from input_file,
    computes αₜ for each, and writes results & any skips to output_file.
    """
    with open(input_file, 'r') as f:
        rows = [ln.split() for ln in f if ln.strip()]

    results, skipped = [], []
    for idx, params in enumerate(rows, start=1):
        try:
            r, C0, Ca, gamma, target = map(float, params)
            α, Lout = find_alpha_t(
                target, r, C0, Ca, gamma, orientation,
                alpha_start, step, tolerance, max_alpha, max_stagnation
            )
            results.append(
                f"Line {idx}: r={r}, C0={C0}, Ca={Ca}, γ={gamma}, "
                f"target={target} → αₜ={α:.6f}, Lmax={Lout:.6f}"
            )
        except Exception as e:
            skipped.append(f"Line {idx}: params={params}  ERROR: {e}")

    with open(output_file, 'w') as f:
        f.write("\n".join(results))
        if skipped:
            f.write("\n\n# Skipped:\n")
            f.write("\n".join(skipped))

    print(f"[done] Wrote {len(results)} lines to '{output_file}'"
          + (f", {len(skipped)} skipped." if skipped else ""))

