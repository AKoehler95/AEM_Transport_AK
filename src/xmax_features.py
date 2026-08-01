# Written by Alvin Yadav

"""
Pure feature extraction for the learned dom_xmax predictor (xmax_predictor.py).

extract_features(config) turns an ATConfiguration into a flat, ordered dict of
physics + source-geometry features. No feature may depend on solver output
(L_max, sim results, etc.) — this module must be importable and callable
before a simulation has ever been run.
"""

import hashlib
import math

from at_element import ATElementType

# Fixed, ordered feature key list. Both extract_features() and the predictor
# rely on this exact order — do not reorder without retraining/persisting a
# new model.
_FEATURE_NAMES = (
    "alpha_l", "alpha_t", "ca", "gamma", "beta",
    "alpha_ratio", "gamma_over_ca", "orientation_vertical",
    "n_elements",
    "x_min_src", "x_max_src", "x_span_src", "x_centroid",
    "y_min_src", "y_max_src", "y_span_src",
    "ws", "total_area", "max_r", "mean_r",
    "c_max", "c_min", "c_mean", "c_sum",
    "mass_proxy", "frac_donor", "has_acceptor", "c_span",
    "thickness_sq_over_at", "conc_ratio", "advection_scale",
    "log1p_thickness_sq_over_at", "log1p_total_area", "log1p_mass_proxy",
)


def feature_names():
    """Return the fixed, ordered tuple of feature keys."""
    return _FEATURE_NAMES


def config_hash_from_path(path: str) -> str:
    """
    Stable identity for a config file, used to dedupe dataset rows across
    machines/batches (two rows with the same hash are the same config, even
    if they live under different filenames or directories).
    """
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _element_area(elem):
    if elem.kind == ATElementType.Circle:
        return math.pi * elem.r ** 2
    if elem.kind == ATElementType.Ellipse:
        b = elem.b if elem.b is not None else elem.r
        return math.pi * elem.r * b
    if elem.kind == ATElementType.Line:
        # Treat a line source as a thin rectangle of width dom_inc.
        eps = 0.5  # documented stand-in for dom_inc when a config isn't available
        return 2.0 * elem.r * eps
    raise ValueError(f"Unknown element kind: {elem.kind}")


def extract_features(config) -> dict:
    """
    Compute the fixed-order feature dict for an ATConfiguration.

    Geometry is computed directly from config.elements (the un-mirrored
    source elements as loaded from JSON), since mirroring for vertical
    orientation is a deterministic solver-side step and would otherwise
    double-count source extent.
    """
    elements = config.elements
    if not elements:
        raise ValueError("Cannot extract features from a config with no elements.")

    alpha_l = config.alpha_l
    alpha_t = config.alpha_t
    ca = config.ca
    gamma = config.gamma
    beta = 1.0 / (2.0 * alpha_l)

    xs = [e.x for e in elements]
    ys = [e.y for e in elements]
    rs = [e.r for e in elements]
    cs = [e.c for e in elements]
    areas = [_element_area(e) for e in elements]

    x_min_src, x_max_src = min(xs), max(xs)
    y_min_src, y_max_src = min(ys), max(ys)
    x_span_src = x_max_src - x_min_src
    y_span_src = y_max_src - y_min_src

    c_max, c_min = max(cs), min(cs)
    c_sum = sum(cs)
    c_mean = c_sum / len(cs)
    total_area = sum(areas)
    mass_proxy = sum(c * a for c, a in zip(cs, areas))

    thickness_sq_over_at = (y_span_src ** 2) / alpha_t

    feats = {
        "alpha_l": alpha_l,
        "alpha_t": alpha_t,
        "ca": ca,
        "gamma": gamma,
        "beta": beta,
        "alpha_ratio": alpha_l / alpha_t,
        "gamma_over_ca": gamma / ca,
        "orientation_vertical": 1.0 if config.orientation == "vertical" else 0.0,
        "n_elements": float(len(elements)),
        "x_min_src": x_min_src,
        "x_max_src": x_max_src,
        "x_span_src": x_span_src,
        "x_centroid": sum(xs) / len(xs),
        "y_min_src": y_min_src,
        "y_max_src": y_max_src,
        "y_span_src": y_span_src,
        "ws": x_span_src,
        "total_area": total_area,
        "max_r": max(rs),
        "mean_r": sum(rs) / len(rs),
        "c_max": c_max,
        "c_min": c_min,
        "c_mean": c_mean,
        "c_sum": c_sum,
        "mass_proxy": mass_proxy,
        "frac_donor": sum(1 for c in cs if c > 0) / len(cs),
        "has_acceptor": 1.0 if c_min < 0 else 0.0,
        "c_span": c_max - c_min,
        "thickness_sq_over_at": thickness_sq_over_at,
        "conc_ratio": c_max / ca,
        "advection_scale": 2.0 * alpha_l,
        "log1p_thickness_sq_over_at": math.log1p(max(thickness_sq_over_at, 0.0)),
        "log1p_total_area": math.log1p(max(total_area, 0.0)),
        "log1p_mass_proxy": math.log1p(max(mass_proxy, 0.0)),
    }
    return {k: feats[k] for k in _FEATURE_NAMES}
