"""
Generate 100 composite source configs — circles only, Ws=0.5m constraint.

Physical scaling:
  Total vertical source span: 5 - 25 m
  Ws = 0.5 m  =>  all circles fit within x-width <= 0.5 m
  r << Ws     =>  radii in [0.008, 0.04] m

Key design: HIGH VARIANCE in cluster vertical placement:
  - Non-uniform gap distribution (exponential, clustered-at-ends, etc.)
  - Some configs have all clusters bunched at top or bottom
  - Some configs have one outlier cluster far from the rest
  - Some configs have two tight groups separated by a large void
  - Gaps drawn from heavy-tailed distributions

Categories (100 total):
  M2  – 2 clusters   15 configs
  M3  – 3 clusters   20 configs
  M4  – 4 clusters   25 configs
  M5  – 5 clusters   25 configs
  B   – big/dense    15 configs
"""

import json, os, math, random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "configs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WS = 0.5

def save_config(name, cfg):
    with open(os.path.join(OUTPUT_DIR, f"{name}.json"), "w") as f:
        json.dump(cfg, f, indent=4)


def make_cluster(cx, cy, n_target, radii_pool, conc_pool,
                 id_prefix, start_idx=0, ls=None):
    r_max = max(radii_pool)
    col_spacing = 2.0 * r_max * 1.03
    row_spacing  = col_spacing * math.sqrt(3) / 2
    n_cols = max(1, int(WS / col_spacing))
    n_rows = max(n_target * 4, 80)

    col_xs = []
    for ci in range(n_cols):
        x = cx - (n_cols - 1) * col_spacing / 2 + ci * col_spacing
        x = max(r_max, min(WS - r_max, x))
        col_xs.append(x)

    candidates = []
    for ci, x in enumerate(col_xs):
        for ri in range(-n_rows // 2, n_rows // 2 + 1):
            y = cy + ri * row_spacing + (ci % 2) * row_spacing * 0.5
            if ls is not None and abs(y - cy) > ls / 2:
                continue
            dist = math.hypot(x - cx, y - cy)
            candidates.append((dist, x, y))

    candidates.sort(key=lambda t: t[0])

    placed, elements = [], []
    for _, x, y in candidates:
        if len(elements) >= n_target:
            break
        r = random.choice(radii_pool)
        c = random.choice(conc_pool)
        if not any(math.hypot(x - px, y - py) < r + pr - 1e-9
                   for px, py, pr in placed):
            placed.append((x, y, r))
            elements.append({
                "kind": "circle",
                "x": round(x, 4),
                "y": round(y, 4),
                "r": round(r, 4),
                "c": round(c, 2),
                "id": f"{id_prefix}_{start_idx + len(elements)}"
            })
    return elements


def base_config(elements):
    all_x = [e["x"] for e in elements]
    all_y = [e["y"] for e in elements]
    max_r = max(e["r"] for e in elements)
    return {
        "alpha_l": 2.0,
        "alpha_t": 0.2,
        "ca": 8.0,
        "gamma": 3.5,
        "dom_xmin": round(min(all_x) - max_r - 2.0, 3),
        "dom_xmax": round(max(all_x) + 150.0, 1),
        "dom_ymin": round(min(all_y) - max_r - 5.0, 3),
        "dom_ymax": round(max(all_y) + max_r + 5.0, 3),
        "dom_inc": 1.0,
        "num_cp": 40,
        "num_terms": 5,
        "orientation": "horizontal",
        "plot_aspect": "",
        "elements": elements
    }


# Radius / concentration pools
TINY  = [0.008, 0.010, 0.012]
SMALL = [0.012, 0.015, 0.018]
MED   = [0.018, 0.022, 0.025]
LARGE = [0.028, 0.032, 0.038, 0.040]
MIXED = [0.008, 0.012, 0.018, 0.025, 0.032, 0.040]

LOW_C  = [3.0, 5.0, 8.0]
MID_C  = [10.0, 15.0, 20.0]
HIGH_C = [25.0, 35.0, 50.0]
FULL_C = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 35.0, 50.0]

ALL_RPOOLS = [TINY, SMALL, MED, LARGE, MIXED]
ALL_CPOOLS = [LOW_C, MID_C, HIGH_C, FULL_C]

random.seed(2026)


# ─── Gap generation strategies ───────────────────────────────────────────────

def gaps_uniform(n, total_gap):
    """Equal spacing — baseline."""
    g = total_gap / n
    return [g] * n

def gaps_exponential(n, total_gap):
    """Exponentially increasing gaps — clusters bunch at one end."""
    raw = [math.exp(i) for i in range(n)]
    s = sum(raw)
    return [r / s * total_gap for r in raw]

def gaps_one_large(n, total_gap):
    """One randomly chosen gap is dominant; rest are tiny."""
    big_idx = random.randint(0, n - 1)
    small = total_gap * 0.05 / max(n - 1, 1)
    big   = total_gap - small * (n - 1)
    return [big if i == big_idx else small for i in range(n)]

def gaps_two_groups(n, total_gap):
    """Clusters split into two tight groups with a large void between them."""
    if n < 2:
        return [total_gap]
    split = random.randint(1, n - 1)
    tiny  = total_gap * 0.03
    large_void = total_gap - tiny * (n - 1)
    gaps = [tiny] * n
    gaps[split - 1] = large_void
    return gaps

def gaps_random_heavy(n, total_gap):
    """Gaps drawn from exponential dist — highly variable."""
    raw = [random.expovariate(1.0) + 0.01 for _ in range(n)]
    s = sum(raw)
    return [r / s * total_gap for r in raw]

def gaps_bunched_at_end(n, total_gap):
    """Most clusters near one end, one outlier far away."""
    # All gaps tiny except the last, which is large
    tiny  = total_gap * 0.04
    large = total_gap - tiny * (n - 1)
    gaps = [tiny] * (n - 1) + [large]
    if random.random() < 0.5:
        gaps = gaps[::-1]  # flip: outlier at top instead
    return gaps

def gaps_irregular(n, total_gap):
    """Fully random with no structure."""
    raw = [random.uniform(0.05, 1.0) for _ in range(n)]
    s = sum(raw)
    return [r / s * total_gap for r in raw]

GAP_STRATEGIES = [
    gaps_exponential,
    gaps_one_large,
    gaps_two_groups,
    gaps_random_heavy,
    gaps_bunched_at_end,
    gaps_irregular,
    gaps_uniform,       # keep some uniform for contrast
]

def pick_gap_strategy():
    # Weight away from uniform so variance configs dominate
    weights = [3, 3, 3, 3, 3, 3, 1]
    return random.choices(GAP_STRATEGIES, weights=weights, k=1)[0]


# ─── Core builder ────────────────────────────────────────────────────────────

def build_multi_cluster(n_clusters, n_range, total_span_range, cat_prefix, cfg_id):
    """
    Build n_clusters stacked vertically.
    Gaps between clusters chosen by a random strategy for high variance.
    Cluster sizes (ls) also varied non-uniformly.
    """
    total_span = random.uniform(*total_span_range)

    # Cluster vertical spans — varied sizes, not uniform
    ls_list = [random.uniform(0.3, min(5.0, total_span * 0.4))
               for _ in range(n_clusters)]

    total_cluster_height = sum(ls_list)
    total_gap_budget = max(total_span - total_cluster_height, 0.2 * (n_clusters - 1))

    # Pick gap strategy and generate gaps
    strategy = pick_gap_strategy()
    gaps = strategy(n_clusters - 1, total_gap_budget)
    # Ensure no negative gaps
    gaps = [max(g, 0.05) for g in gaps]

    # Recompute actual span
    actual_span = total_cluster_height + sum(gaps)

    # Stack top-to-bottom, centred on y=0
    y = actual_span / 2
    centres = []
    for k in range(n_clusters):
        cy = y - ls_list[k] / 2
        cx = random.uniform(0.05, WS - 0.05)
        centres.append((cx, cy, ls_list[k]))
        y -= ls_list[k] + (gaps[k] if k < n_clusters - 1 else 0)

    # Build elements
    all_elems = []
    start = 0
    for k, (cx, cy, ls) in enumerate(centres):
        rpool = random.choice(ALL_RPOOLS)
        cpool = random.choice(ALL_CPOOLS)
        n = random.randint(*n_range)
        cluster = make_cluster(cx, cy, n, rpool, cpool,
                               f"{cat_prefix}{k}_{cfg_id}",
                               start_idx=start, ls=ls)
        all_elems.extend(cluster)
        start += len(cluster)

    return all_elems, actual_span, strategy.__name__


# ─── Generate configs ─────────────────────────────────────────────────────────

idx = 0

# M2: 2 clusters, span 5-25 m  (15 configs)
for _ in range(15):
    idx += 1
    elems, span, strat = build_multi_cluster(
        2, (6, 25), (5.0, 25.0), "m2", idx)
    if len(elems) < 3: continue
    save_config(f"M2_{idx:03d}_{len(elems)}circ_span{span:.0f}m",
                base_config(elems))

# M3: 3 clusters, span 5-25 m  (20 configs)
for _ in range(20):
    idx += 1
    elems, span, strat = build_multi_cluster(
        3, (5, 20), (5.0, 25.0), "m3", idx)
    if len(elems) < 3: continue
    save_config(f"M3_{idx:03d}_{len(elems)}circ_span{span:.0f}m",
                base_config(elems))

# M4: 4 clusters, span 8-25 m  (25 configs)
for _ in range(25):
    idx += 1
    elems, span, strat = build_multi_cluster(
        4, (5, 18), (8.0, 25.0), "m4", idx)
    if len(elems) < 4: continue
    save_config(f"M4_{idx:03d}_{len(elems)}circ_span{span:.0f}m",
                base_config(elems))

# M5: 5 clusters, span 10-25 m  (25 configs)
for _ in range(25):
    idx += 1
    elems, span, strat = build_multi_cluster(
        5, (4, 15), (10.0, 25.0), "m5", idx)
    if len(elems) < 5: continue
    save_config(f"M5_{idx:03d}_{len(elems)}circ_span{span:.0f}m",
                base_config(elems))

# B: big/dense 2-5 clusters, span 5-25 m  (15 configs)
for _ in range(15):
    idx += 1
    nc = random.randint(2, 5)
    elems, span, strat = build_multi_cluster(
        nc, (10, 20), (5.0, 25.0), "b", idx)
    if len(elems) < 5: continue
    save_config(f"B_{idx:03d}_{len(elems)}circ_span{span:.0f}m",
                base_config(elems))


# ─── Summary ─────────────────────────────────────────────────────────────────
configs = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json"))
print(f"\nGenerated {len(configs)} configs in {OUTPUT_DIR}/")

total, mx, xspans, yspans = 0, 0, [], []
for cf in configs:
    d = json.load(open(os.path.join(OUTPUT_DIR, cf)))
    elems = d["elements"]
    xs = [e["x"] for e in elems]
    ys = [e["y"] for e in elems]
    total += len(elems); mx = max(mx, len(elems))
    xspans.append(max(xs) - min(xs))
    yspans.append(max(ys) - min(ys))

print(f"Elements : total={total}, max={mx}, avg={total/len(configs):.1f}")
print(f"X-span   : {min(xspans):.3f} - {max(xspans):.3f} m  (limit 0.5 m)")
print(f"Y-span   : {min(yspans):.1f} - {max(yspans):.1f} m")
print(f"X violations: {sum(1 for s in xspans if s > WS)}")

print("\nY-span distribution:")
for lo, hi in [(0,5),(5,10),(10,15),(15,20),(20,26),(26,40)]:
    n = sum(1 for s in yspans if lo <= s < hi)
    print(f"  {lo:2d}-{hi:2d} m: {n:3d} configs  {'#'*n}")

cats = {}
for cf in configs:
    cats[cf.split('_')[0]] = cats.get(cf.split('_')[0], 0) + 1
print(f"\nCategories: {cats}")