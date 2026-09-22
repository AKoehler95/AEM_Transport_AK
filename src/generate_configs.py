"""
AEM Composite Source Config Generator

Generates a batch of random source configurations for the AEM transport model.
Each config places circular source elements inside a source zone of width WS and
height y_span, in one of several spatial patterns (archetypes), filled to a
random packing fraction.

Usage
-----
1. Edit the USER CONFIG block below (batch size, output folder, factor ranges).
2. Run:  python generate_configs.py
3. Outputs written to OUTPUT_DIR/:
     <name>.json    one simulation config each (loadable by at_config.from_json)
     manifest.csv   one row per config with all its parameters
     factors.json   the factor definitions and the list of varying factors

To change how a parameter is sampled, change its FORM in FACTORS:
     8.0            -> constant
     (4.0, 12.0)    -> uniform random in [4, 12]
     [4.0, 8.0]     -> random choice from a list
No other code changes are needed.
"""

import csv
import json
import math
import os
import random
import numpy as np

#USER CONFIG

N_CONFIGS   = 200
OUTPUT_DIR  = "mc_batch_008"
RANDOM_SEED = 31051994              # int for reproducibility, or None
ORIENTATION = "vertical"        # "horizontal" | "vertical"

# How random samples are spread over the factor ranges:
#   "lhs"         Latin-hypercube, evenly stratified coverage
#   "independent" plain independent random draws
SAMPLING_MODE = "lhs"

# Circle radii available to the fill, as multiples of WS. Each pool spans a
# range of sizes: the large ones set the structure, the small ones fill gaps.
# Larger circles => fewer elements (the AEM solve slows and loses conditioning
# past a few hundred). Hard limit: radius <= WS/2.
RADIUS_FACTORS = {
    "FINE":   [0.06, 0.10, 0.16, 0.24],
    "MED":    [0.09, 0.15, 0.24, 0.36],
    "COARSE": [0.12, 0.20, 0.30, 0.42],
    "BROAD":  [0.08, 0.14, 0.22, 0.32, 0.42],
}
CD_POOL = np.linspace(10, 500, 50, dtype=int).tolist() #[3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0]   # donor conc. [mg/L] 

GAMMA_BOUNDS = (3.5, 3.5) #(2, 20.0)      # valid range for the derived gamma


class D:
    """Marker for a derived factor: fn receives the resolved factors so far."""

    def __init__(self, fn):
        self.fn = fn


# Factor registry. Change a value's FORM (see module docstring) to change how it
# is sampled. Constants and derived factors are excluded from the design matrix.
FACTORS = {
    # Geometry
    "ws":          10.0,                       # source width [m]
    "y_span":      (1.0, 3.0),               # source height [m]
    "packing":     (0.10, 0.80),              # fraction of WS*y_span covered by circles
    "n_clusters":  {"choice": [2, 3, 4, 5], "weights": [1, 2, 3, 2]},
    "archetype":   {"choice": list("CEFH")},    # spatial pattern; see ARCHETYPES "ACDEFHIJ"
    "radius_pool": {"choice": ["FINE", "MED", "COARSE"]}, #, "BROAD"

    # Transport
    "alpha_l":     (0.5, 2.0),                # longitudinal dispersivity
    "alpha_ratio": (0.05, 0.20),              # alpha_t / alpha_l
    "alpha_t":     (0.0005, 0.1), #D(lambda f: f["alpha_l"] * f["alpha_ratio"]),

    # Chemical Term: specify any three of T, ca, cd, gamma; derive the fourth by
    # moving the D(...) wrapper. T = 1 + gamma*cd/ca.
    "T":           (2.0, 50.0),               # retardation-like ratio
    "ca":          8.0,                       # acceptor conc. [mg/L]
    "cd":          CD_POOL,                   # donor conc. [mg/L]
    "gamma":       3.5, #D(lambda f: (f["T"] - 1.0) * f["ca"] / f["cd"]),

    # Numerics
    "num_cp":      50,
    "num_terms":   15,                         # keep at 5; dense sources are unstable higher
    "dom_inc":     5.0,
}

# A config is rejected and resampled if any constraint returns False.
CONSTRAINTS = [
    lambda f: GAMMA_BOUNDS[0] <= f["gamma"] <= GAMMA_BOUNDS[1],
    lambda f: f["alpha_t"] < f["alpha_l"],
]

# Reject configs with more than this many source elements (each is mirrored for
# the vertical boundary, so the solve sees ~2x). Keeps every config solvable.
MAX_ELEMENTS_PER_CONFIG = 250

# A cluster should read as a cluster, so radii whose area would eat more than
# 1/this of the cluster's budget are dropped from its pool.
MIN_CIRCLES_PER_CLUSTER = 5

# END USER CONFIG

# LHS cannot resample a single rejected row without breaking its stratification,
# so the design is over-generated and rows are consumed in order, skipping
# rejects. This is the over-generation factor.
_DESIGN_OVERSAMPLE = 2.0


# Factor resolution and sampling

class FactorSpec:
    """A factor normalised to one form: const / uniform / loguniform / choice /
    derived. `varies` is True for the forms that consume a random variate."""

    def __init__(self, kind, args=None, weights=None, fn=None):
        self.kind = kind
        self.args = args if args is not None else []
        self.weights = weights
        self.fn = fn

    @property
    def varies(self):
        return self.kind in ("uniform", "loguniform", "choice")

    @property
    def derived(self):
        return self.kind == "derived"

    def as_dict(self):
        out = {"kind": self.kind, "varies": self.varies, "derived": self.derived}
        if not self.derived:
            out["args"] = list(self.args)
        if self.weights is not None:
            out["weights"] = list(self.weights)
        return out


def resolve_registry(factors):
    """Normalise every FACTORS entry into a FactorSpec, raising on a bad form."""
    resolved = {}
    for name, spec in factors.items():
        if isinstance(spec, D):
            resolved[name] = FactorSpec("derived", fn=spec.fn)
        elif isinstance(spec, FactorSpec):
            resolved[name] = spec
        elif isinstance(spec, tuple):
            if len(spec) != 2:
                raise ValueError(f"Factor '{name}': tuple must be (lo, hi).")
            resolved[name] = FactorSpec("uniform", args=list(spec))
        elif isinstance(spec, list):
            if not spec:
                raise ValueError(f"Factor '{name}': pool list is empty.")
            resolved[name] = FactorSpec("choice", args=list(spec))
        elif isinstance(spec, dict):
            if "choice" in spec:
                choices = list(spec["choice"])
                if not choices:
                    raise ValueError(f"Factor '{name}': 'choice' is empty.")
                weights = spec.get("weights")
                if weights is not None:
                    weights = list(weights)
                    if len(weights) != len(choices):
                        raise ValueError(
                            f"Factor '{name}': weights/choices length mismatch.")
                    if any(w < 0 for w in weights) or sum(weights) <= 0:
                        raise ValueError(
                            f"Factor '{name}': weights must be non-negative.")
                resolved[name] = FactorSpec("choice", args=choices, weights=weights)
            elif "loguniform" in spec:
                lo, hi = spec["loguniform"]
                if lo <= 0:
                    raise ValueError(f"Factor '{name}': loguniform needs lo > 0.")
                resolved[name] = FactorSpec("loguniform", args=[lo, hi])
            elif "uniform" in spec:
                resolved[name] = FactorSpec("uniform", args=list(spec["uniform"]))
            else:
                raise ValueError(f"Factor '{name}': unknown dict spec {sorted(spec)}.")
        elif isinstance(spec, (int, float, str)):
            resolved[name] = FactorSpec("const", args=[spec])
        else:
            raise ValueError(f"Factor '{name}': unrecognised form {spec!r}.")
    return resolved


RESOLVED = resolve_registry(FACTORS)

# The varying factors, in order — these are the design-matrix columns.
ACTIVE = [name for name, spec in RESOLVED.items() if spec.varies]


def sample_one(spec, u):
    """Map a uniform variate u in [0,1) to a factor value via its inverse CDF."""
    if spec.kind == "const":
        return spec.args[0]
    if spec.kind == "uniform":
        lo, hi = spec.args
        return lo + u * (hi - lo)
    if spec.kind == "loguniform":
        lo, hi = spec.args
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if spec.kind == "choice":
        choices = spec.args
        weights = spec.weights or [1.0] * len(choices)
        total = float(sum(weights))
        acc = 0.0
        for choice, w in zip(choices, weights):
            acc += w / total
            if u < acc:
                return choice
        return choices[-1]
    raise ValueError(f"sample_one: cannot sample kind '{spec.kind}'.")


def build_design(n, k):
    """Return an (n, k) list of rows of uniforms in [0,1) per SAMPLING_MODE."""
    if k == 0:
        return [[] for _ in range(n)]
    if SAMPLING_MODE == "lhs":
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=k, seed=RANDOM_SEED)
            return [list(row) for row in sampler.random(n)]
        except ImportError:
            print("WARNING: scipy unavailable; using independent sampling.")
        except Exception as err:
            print(f"WARNING: LHS failed ({err}); using independent sampling.")
    elif SAMPLING_MODE != "independent":
        raise ValueError(f"SAMPLING_MODE must be 'lhs' or 'independent'.")
    return [[random.random() for _ in range(k)] for _ in range(n)]


def realise_factors(row):
    """Turn one design row into a full factor dict (sampled then derived)."""
    if len(row) != len(ACTIVE):
        raise ValueError(f"row has {len(row)} values, expected {len(ACTIVE)}.")

    factors = {name: sample_one(RESOLVED[name], u) for name, u in zip(ACTIVE, row)}
    for name, spec in RESOLVED.items():
        if spec.kind == "const":
            factors[name] = spec.args[0]

    pending = [name for name, spec in RESOLVED.items() if spec.derived]
    while pending:
        progressed = []
        for name in pending:
            try:
                factors[name] = RESOLVED[name].fn(factors)
            except KeyError:
                continue
            progressed.append(name)
        if not progressed:
            raise ValueError(f"Cannot resolve derived factors {sorted(pending)} "
                             f"(dependency cycle or unknown reference).")
        pending = [n for n in pending if n not in progressed]
    return factors


# Packing: grow blob-shaped clusters of mixed-size circles

def _cluster_targets(ls_list, packing, ws, y_span):
    """Split the total covered-area budget (packing*WS*y_span) across clusters
    in proportion to their heights."""
    total_ls = sum(ls_list)
    if total_ls <= 0:
        return [0.0 for _ in ls_list]
    budget = packing * ws * y_span
    return [budget * ls / total_ls for ls in ls_list]


def _blob_axes(ls, ws, target_area, r_min, r_max, fill_eff):
    """Semi-axes (a, b) of the ellipse that holds target_area at fill_eff
    coverage.

    b is never allowed below the radius of an equivalent circular blob, so a
    thin cluster slot cannot squash the blob into a single row of circles;
    where ls is genuinely tall, b follows ls and the blob comes out narrow and
    elongated instead. a is capped so the blob fits inside [0, WS].
    """
    if target_area <= 0:
        r = max(r_min, min(r_max, ws / 2.0))
        return r, max(ls / 2.0, r)
    r_circ = math.sqrt(target_area / (math.pi * fill_eff))
    b = max(ls / 2.0, r_circ)
    a = target_area / (math.pi * b * fill_eff)
    return max(r_min, min(a, ws / 2.0)), b


def _pack_blob(cx, cy, ls, ws, rpool, cd, target_area, tag, start_idx,
               clearance=1.02, fill_eff=0.65, existing=None):
    """Grow a blob of circles outward from (cx, cy) until the covered area
    reaches target_area.

    The blob is an ellipse with half-height ls/2; its half-width is sized to
    hold target_area (see _blob_half_width), so low packing fractions give
    narrow, locally clustered blobs rather than full-width layers.

    Candidate centres sit on a hex lattice and are consumed in order of
    distance from the centre, so the blob fills from the middle outward. Each
    position draws a random radius and falls back to smaller ones if that does
    not fit, which lets big circles set the structure and small ones fill the
    gaps. Centres stay inside [r, WS-r] (the WS constraint) and never overlap.

    cx=None draws a random horizontal centre that keeps the blob in the zone.
    `existing` is the circles already placed by earlier clusters; they are
    treated as obstacles so clusters cannot overlap each other.

    Returns (elements, covered_area).
    """
    radii = sorted(set(rpool), reverse=True)
    if not radii or ls <= 0:
        return [], 0.0

    # Drop radii too big for this cluster's area budget: a single circle of
    # radius r covers pi*r^2, so anything above the cap would spend the whole
    # budget at once and leave a "cluster" of one or two circles. The smallest
    # radius is always kept so the pool can never empty.
    if target_area > 0 and len(radii) > 1:
        r_cap = math.sqrt(target_area / (math.pi * MIN_CIRCLES_PER_CLUSTER))
        eligible = [r for r in radii if r <= r_cap]
        # Keep at least the two smallest so a cluster always has a size mix.
        radii = eligible if len(eligible) >= 2 else radii[-2:]
    r_max, r_min = radii[0], radii[-1]

    a, b = _blob_axes(ls, ws, target_area, r_min, r_max, fill_eff)

    if cx is None:
        lo, hi = a, ws - a
        cx = random.uniform(lo, hi) if hi > lo else ws / 2.0
    cx = min(max(cx, a), max(a, ws - a))

    cell = 2.0 * max(r_max, max((e["r"] for e in (existing or ())), default=0.0))
    grid = {}

    def remember(x, y, r):
        grid.setdefault((int(x / cell), int(y / cell)), []).append((x, y, r))

    for e in (existing or ()):
        remember(e["x"], e["y"], e["r"])

    def overlaps(x, y, r):
        gx, gy = int(x / cell), int(y / cell)
        for ax in (gx - 1, gx, gx + 1):
            for ay in (gy - 1, gy, gy + 1):
                for px, py, pr in grid.get((ax, ay), ()):
                    if math.hypot(x - px, y - py) < r + pr - 1e-12:
                        return True
        return False

    pitch = 2.0 * r_min * clearance
    row_pitch = pitch * math.sqrt(3) / 2.0
    n_rows = int(math.ceil(b / row_pitch)) + 1
    n_cols = int(math.ceil(a / pitch)) + 2

    candidates = []
    for ri in range(-n_rows, n_rows + 1):
        y = cy + ri * row_pitch
        x_off = (pitch / 2.0) if ri % 2 else 0.0
        for ci in range(-n_cols, n_cols + 1):
            x = cx + ci * pitch + x_off
            d = math.hypot((x - cx) / a, (y - cy) / b)
            if d <= 1.0:
                candidates.append((d, x, y))
    candidates.sort()

    elements = []
    covered = 0.0
    for _, x, y in candidates:
        if target_area > 0 and covered >= target_area:
            break
        r_first = random.choice(radii)
        for r in [r_first] + [rr for rr in radii if rr < r_first]:
            if x < r or x > ws - r:
                continue
            if overlaps(x, y, r):
                continue
            remember(x, y, r)
            elements.append({
                "kind": "circle",
                "x": round(x, 5),
                "y": round(y, 5),
                "r": round(r, 5),
                "c": round(cd, 2),
                "id": f"{tag}_{start_idx + len(elements)}",
            })
            covered += math.pi * r * r
            break
    return elements, covered


def make_cluster(cy, ls, rpool, cd, target_area, ws, tag, start_idx=0, cx=None,
                 existing=None):
    """Build one blob-shaped cluster of height ls centred vertically on cy.

    cx defaults to a random horizontal centre, so each cluster lands in its own
    place across the source zone instead of spanning it end to end.
    """
    return _pack_blob(cx, cy, ls, ws, rpool, cd, target_area, tag, start_idx,
                      existing=existing)


# Archetypes: place clusters vertically in a characteristic pattern
# (each cluster's horizontal centre is drawn at random by make_cluster)

class _Accumulator:
    """Collects elements across an archetype's clusters."""

    def __init__(self):
        self.elements = []
        self.covered = 0.0

    def add(self, result):
        elems, covered = result
        self.elements.extend(elems)
        self.covered += covered

    @property
    def start(self):
        return len(self.elements)

    def place(self, cy, ls, rpool, cd, target_area, ws, tag, cx=None):
        """Add one cluster, passing the already-placed circles through as
        obstacles so clusters cannot overlap one another."""
        self.add(make_cluster(cy, ls, rpool, cd, target_area, ws, tag,
                              start_idx=self.start, cx=cx,
                              existing=self.elements))

    def result(self):
        return self.elements, self.covered


def sample_cluster_ls(n_clusters, total_span, ws):
    """Cluster height: sized so heights sum to most of the span, leaving modest
    gaps for vertical structure."""
    cap = 0.85 * total_span / n_clusters
    lo = min(0.3 * ws, cap * 0.5)
    return random.uniform(lo, max(lo * 1.1, cap))


def expo_gaps(n, budget):
    """n exponentially-distributed gaps rescaled to sum to budget."""
    raw = [random.expovariate(1.0) + 0.05 for _ in range(n)]
    s = sum(raw)
    return [max(0.02, g / s * budget) for g in raw]


def _stack(ls_list, gaps, targets, ws, rpool, cd, tag_prefix, cfg_id):
    """Stack clusters top-to-bottom with the given heights and gaps."""
    acc = _Accumulator()
    y = (sum(ls_list) + sum(gaps)) / 2.0
    for k in range(len(ls_list)):
        cy = y - ls_list[k] / 2.0
        acc.place(cy, ls_list[k], rpool, cd, targets[k], ws, f"{tag_prefix}{k}_{cfg_id}")
        y -= ls_list[k] + (gaps[k] if k < len(ls_list) - 1 else 0.0)
    return acc.result()


def archetype_vertical_stack(cfg_id, n_clusters, f):
    """A: evenly stacked clusters with exponentially-distributed gaps."""
    ws, span = f["ws"], f["y_span"]
    ls_list = [sample_cluster_ls(n_clusters, span, ws) for _ in range(n_clusters)]
    gaps = expo_gaps(n_clusters - 1, max(span - sum(ls_list), 0.3 * (n_clusters - 1)))
    targets = _cluster_targets(ls_list, f["packing"], ws, span)
    return _stack(ls_list, gaps, targets, ws, f["_rpool"], f["cd"], "vs", cfg_id)


def archetype_top_heavy(cfg_id, n_clusters, f):
    """C: all clusters in the upper half, lower half empty."""
    ws, span = f["ws"], f["y_span"]
    half = span / 2.0
    ls_list = [sample_cluster_ls(n_clusters, half, ws) for _ in range(n_clusters)]
    gaps = expo_gaps(n_clusters - 1, max(half - sum(ls_list), 0.2 * (n_clusters - 1)))
    targets = _cluster_targets(ls_list, f["packing"], ws, span)
    acc = _Accumulator()
    y = half
    for k in range(n_clusters):
        cy = y - ls_list[k] / 2.0
        acc.place(cy, ls_list[k], f["_rpool"], f["cd"], targets[k], ws, f"th{k}_{cfg_id}")
        y -= ls_list[k] + (gaps[k] if k < n_clusters - 1 else 0.0)
    return acc.result()


def archetype_bottom_heavy(cfg_id, n_clusters, f):
    """D: top-heavy mirrored into the lower half."""
    elems, covered = archetype_top_heavy(cfg_id, n_clusters, f)
    for e in elems:
        e["y"] = round(-e["y"], 5)
    return elems, covered


def archetype_tight_pack(cfg_id, n_clusters, f):
    """E: clusters very close together (small gaps)."""
    ws, span = f["ws"], f["y_span"]
    ls_list = [sample_cluster_ls(n_clusters, span, ws) for _ in range(n_clusters)]
    gaps = [random.uniform(0.02 * ws, 0.1 * span) for _ in range(n_clusters - 1)]
    targets = _cluster_targets(ls_list, f["packing"], ws, span)
    return _stack(ls_list, gaps, targets, ws, f["_rpool"], f["cd"], "tp", cfg_id)


def archetype_scattered(cfg_id, n_clusters, f):
    """F: clusters at random vertical positions."""
    ws, span = f["ws"], f["y_span"]
    min_sep = span / (n_clusters * 2.5)
    cy_list = sorted(random.uniform(-span / 2, span / 2) for _ in range(n_clusters))
    for i in range(1, len(cy_list)):
        if cy_list[i] - cy_list[i - 1] < min_sep:
            cy_list[i] = cy_list[i - 1] + min_sep
    ls_list = [sample_cluster_ls(n_clusters, span, ws) for _ in range(n_clusters)]
    targets = _cluster_targets(ls_list, f["packing"], ws, span)
    acc = _Accumulator()
    for k, cy in enumerate(cy_list):
        acc.place(cy, ls_list[k], f["_rpool"], f["cd"], targets[k], ws, f"sc{k}_{cfg_id}")
    return acc.result()


def archetype_two_groups(cfg_id, n_clusters, f):
    """H: two tight sub-groups with a large void between them."""
    ws, span = f["ws"], f["y_span"]
    n_top = max(1, n_clusters // 2)
    n_bot = n_clusters - n_top
    void = random.uniform(0.3 * span, 0.7 * span)
    top_span = random.uniform(0.1 * span, 0.3 * span)
    bot_span = random.uniform(0.1 * span, 0.3 * span)

    ls_all = ([top_span / max(n_top, 1)] * n_top
              + [bot_span / max(n_bot, 1)] * n_bot)
    targets = _cluster_targets(ls_all, f["packing"], ws, span)
    idx = [0]
    acc = _Accumulator()

    def pack_group(n, group_span, y_top, prefix):
        ls_each = group_span / max(n, 1)
        tiny = [random.uniform(0.01, 0.1 * ws) for _ in range(n - 1)]
        y = y_top
        for k in range(n):
            cy = y - ls_each / 2.0
            acc.place(cy, ls_each, f["_rpool"], f["cd"], targets[idx[0]], ws, f"{prefix}{k}_{cfg_id}")
            idx[0] += 1
            y -= ls_each + (tiny[k] if k < n - 1 else 0.0)

    y_top = (top_span + void + bot_span) / 2.0
    pack_group(n_top, top_span, y_top, "tg")
    pack_group(n_bot, bot_span, y_top - top_span - void, "bg")
    return acc.result()


def archetype_outlier(cfg_id, n_clusters, f):
    """I: a main group plus one isolated cluster far away."""
    ws, span = f["ws"], f["y_span"]
    n_main = max(1, n_clusters - 1)
    main_span = random.uniform(0.1 * span, 0.3 * span)
    outlier_gap = random.uniform(0.4 * span, 0.9 * span)
    outlier_ls = sample_cluster_ls(1, main_span, ws)
    ls_each = main_span / max(n_main, 1)

    targets = _cluster_targets([ls_each] * n_main + [outlier_ls], f["packing"], ws, span)
    acc = _Accumulator()
    y = main_span / 2.0
    for k in range(n_main):
        cy = y - ls_each / 2.0
        acc.place(cy, ls_each, f["_rpool"], f["cd"], targets[k], ws, f"om{k}_{cfg_id}")
        y -= ls_each + random.uniform(0.05, 0.2 * ws)
    cy_out = -main_span / 2.0 - outlier_gap
    acc.place(cy_out, outlier_ls, f["_rpool"], f["cd"], targets[-1], ws, f"oo_{cfg_id}")
    return acc.result()


def archetype_dense_centre(cfg_id, n_clusters, f):
    """J: clusters bunched near y=0, spacing grows outward."""
    ws, span = f["ws"], f["y_span"]
    base_gap = random.uniform(0.1 * ws, 0.5 * ws)
    growth = random.uniform(1.5, 3.0)
    positions, gap = [0.0], base_gap
    for _ in range(n_clusters - 1):
        positions.append(positions[-1] + gap)
        positions.append(positions[0] - gap)
        gap *= growth
    positions = sorted(set(round(p, 5) for p in positions))[:n_clusters]

    ls_list = [sample_cluster_ls(n_clusters, span, ws) for _ in range(len(positions))]
    targets = _cluster_targets(ls_list, f["packing"], ws, span)
    acc = _Accumulator()
    for k, cy in enumerate(positions):
        acc.place(cy, ls_list[k], f["_rpool"], f["cd"], targets[k], ws, f"dc{k}_{cfg_id}")
    return acc.result()


ARCHETYPES = {
    "A": archetype_vertical_stack,
    "C": archetype_top_heavy,
    "D": archetype_bottom_heavy,
    "E": archetype_tight_pack,
    "F": archetype_scattered,
    "H": archetype_two_groups,
    "I": archetype_outlier,
    "J": archetype_dense_centre,
}
ARCHETYPE_NAMES = {
    "A": "vertical-stack", "C": "top-heavy", "D": "bottom-heavy",
    "E": "tight-pack", "F": "scattered", "H": "two-groups",
    "I": "outlier", "J": "dense-centre",
}


# Config assembly and outputs

def base_config(elements, factors):
    """Build the simulation config dict. beta is not written — at_config derives
    it from alpha_l."""
    max_r = max(e["r"] for e in elements)

    if ORIENTATION == "vertical":
        # Shift the stack down so every element sits below the y=0 boundary.
        shift = max(e["y"] for e in elements) + max_r + 0.15
        if shift > 0:
            for e in elements:
                e["y"] = round(e["y"] - shift, 5)

    all_x = [e["x"] for e in elements]
    all_y = [e["y"] for e in elements]
    dom_ymax = 0.0 if ORIENTATION == "vertical" else round(max(all_y) + max_r + 5.0, 3)

    return {
        "alpha_l": round(factors["alpha_l"], 6),
        "alpha_t": round(factors["alpha_t"], 6),
        "ca": round(factors["ca"], 6),
        "gamma": round(factors["gamma"], 6),
        "dom_xmin": round(min(all_x) - max_r - 2.0, 3),
        "dom_xmax": round(max(all_x) + 150.0, 1),
        "dom_ymin": round(min(all_y) - max_r - 5.0, 3),
        "dom_ymax": dom_ymax,
        "dom_inc": factors["dom_inc"],
        "num_cp": int(factors["num_cp"]),
        "num_terms": int(factors["num_terms"]),
        "orientation": ORIENTATION,
        "plot_aspect": "",
        "elements": elements,
    }


def save_config(name, cfg, output_dir):
    with open(os.path.join(output_dir, f"{name}.json"), "w") as fh:
        json.dump(cfg, fh, indent=4)


def write_factors_json(output_dir, n_configs):
    """Write the resolved registry and the list of varying (active) factors."""
    out = {name: spec.as_dict() for name, spec in RESOLVED.items()}
    out["active"] = list(ACTIVE)
    out["sampling_mode"] = SAMPLING_MODE
    out["seed"] = RANDOM_SEED
    out["n_configs"] = n_configs
    with open(os.path.join(output_dir, "factors.json"), "w") as fh:
        json.dump(out, fh, indent=4)


def write_manifest(output_dir, rows):
    if not rows:
        return
    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_batch(n_configs=N_CONFIGS, output_dir=OUTPUT_DIR, verbose=True):
    """Generate the batch and write configs, manifest.csv and factors.json.
    Returns a summary dict."""
    random.seed(RANDOM_SEED)
    os.makedirs(output_dir, exist_ok=True)

    design = build_design(int(math.ceil(n_configs * _DESIGN_OVERSAMPLE)), len(ACTIVE))

    manifest_rows = []
    generated = 0
    rejected = {"constraint": 0, "geometry": 0, "bounds": 0, "count": 0}
    archetype_counts = {}

    for row_index, row in enumerate(design):
        if generated >= n_configs:
            break

        f = realise_factors(row)
        if not all(c(f) for c in CONSTRAINTS):
            rejected["constraint"] += 1
            continue

        ws = f["ws"]
        f["_rpool"] = [round(ws * x, 6) for x in RADIUS_FACTORS[f["radius_pool"]]]
        archetype_key = f["archetype"]
        n_clusters = int(f["n_clusters"])

        try:
            elems, covered = ARCHETYPES[archetype_key](row_index + 1, n_clusters, f)
        except Exception as err:
            if verbose:
                print(f"  WARN {archetype_key} row {row_index}: {err}")
            rejected["geometry"] += 1
            continue

        if len(elems) < 3:
            rejected["geometry"] += 1
            continue
        if len(elems) > MAX_ELEMENTS_PER_CONFIG:
            rejected["count"] += 1
            continue

        xs = [e["x"] for e in elems]
        if max(xs) > ws + 1e-5 or min(xs) < -1e-5:
            rejected["bounds"] += 1
            continue

        cfg = base_config(elems, f)
        packing_global = covered / (ws * f["y_span"]) if f["y_span"] > 0 else 0.0
        config_name = f"{archetype_key}{generated + 1:03d}_{len(elems)}circ_{n_clusters}cl"

        meta = {
            "config_name": config_name,
            "archetype": archetype_key,
            "n_clusters": n_clusters,
            "n_elements": len(elems),
            "seed": RANDOM_SEED,
            "ws": ws,
            "y_span": round(f["y_span"], 5),
            "alpha_l": round(f["alpha_l"], 6),
            "alpha_t": round(f["alpha_t"], 6),
            "alpha_ratio": round(f["alpha_ratio"], 6),
            "T": round(f["T"], 6),
            "ca": round(f["ca"], 6),
            "cd": round(f["cd"], 6),
            "gamma": round(f["gamma"], 6),
            "packing": round(f["packing"], 6),          # target
            "packing_global": round(packing_global, 6),  # achieved
            "total_covered_area": round(covered, 8),
            "radius_pool": f["radius_pool"],
            "num_cp": int(f["num_cp"]),
            "num_terms": int(f["num_terms"]),
            "dom_inc": f["dom_inc"],
            "design_row_index": row_index,
        }
        cfg["_meta"] = meta

        save_config(config_name, cfg, output_dir)
        manifest_rows.append(dict(meta))
        archetype_counts[archetype_key] = archetype_counts.get(archetype_key, 0) + 1
        generated += 1

    write_manifest(output_dir, manifest_rows)
    write_factors_json(output_dir, generated)

    summary = {
        "generated": generated,
        "requested": n_configs,
        "rows_available": len(design),
        "rejected": rejected,
        "archetype_counts": archetype_counts,
        "manifest_rows": manifest_rows,
    }
    if verbose:
        print_summary(summary, output_dir)
    return summary


def print_summary(summary, output_dir):
    rows = summary["manifest_rows"]
    print(f"\nGenerated {summary['generated']} / {summary['requested']} "
          f"configs in '{output_dir}/'")
    print(f"Sampling: {SAMPLING_MODE}, seed={RANDOM_SEED}, orientation={ORIENTATION}")
    r = summary["rejected"]
    print(f"Rejected — constraints: {r['constraint']}, geometry: {r['geometry']}, "
          f"bounds: {r['bounds']}, over-count: {r['count']}")
    if summary["generated"] < summary["requested"]:
        print(f"WARNING: only {summary['generated']} configs produced — raise "
              f"N_CONFIGS headroom or relax the constraints/element cap.")
    if not rows:
        return

    print("\nRealised ranges (varying factors):")
    for name in ACTIVE:
        if name not in rows[0]:
            continue
        values = [row[name] for row in rows]
        if all(isinstance(v, (int, float)) for v in values):
            print(f"  {name:14s} {min(values):>10.4f} .. {max(values):>10.4f}")
        else:
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            print(f"  {name:14s} {dict(sorted(counts.items()))}")

    errs = [abs(row["packing_global"] - row["packing"]) for row in rows]
    print(f"\nPacking target vs achieved: mean|delta|={sum(errs) / len(errs):.4f}, "
          f"max={max(errs):.4f}")
    print(f"\nArchetype counts: {dict(sorted(summary['archetype_counts'].items()))}")
    for key, name in ARCHETYPE_NAMES.items():
        print(f"  {key}={name}", end="")
    print()


if __name__ == "__main__":
    generate_batch()
