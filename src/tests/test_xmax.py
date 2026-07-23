"""
Unit tests for xmax_features.py and xmax_predictor.py — the learned
dom_xmax predictor used by run_tests.py's --use-predictor/--learn toggles.

Run with:  cd src && pytest tests/test_xmax.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from at_config import ATConfiguration
import xmax_features
import xmax_predictor


# ── Helpers ──────────────────────────────────────────────────────────────

def _base(orientation="horizontal", elements=None):
    return {
        "alpha_l": 2.0, "alpha_t": 0.2, "ca": 8.0, "gamma": 3.5,
        "dom_xmin": 0.0, "dom_xmax": 300.0, "dom_ymin": -20.0, "dom_ymax": 20.0,
        "dom_inc": 1.0, "num_cp": 40, "num_terms": 5,
        "orientation": orientation,
        "elements": elements or [],
    }


def _circle_config(orientation="horizontal"):
    return ATConfiguration.from_dict(_base(orientation, [
        {"kind": "circle", "x": 10.0, "y": 0.0, "c": 5.0, "r": 1.0},
        {"kind": "circle", "x": 12.0, "y": -1.0, "c": -3.0, "r": 0.5},
    ]))


def _line_config():
    return ATConfiguration.from_dict(_base("horizontal", [
        {"kind": "line", "x": 5.0, "y": 0.0, "c": 10.0, "l": 4.0, "theta": 90},
    ]))


def _ellipse_config(orientation="vertical"):
    return ATConfiguration.from_dict(_base(orientation, [
        {"kind": "ellipse", "x": 5.0, "y": -3.0, "c": 10.0, "a": 2.0, "b": 1.0, "theta": 30},
    ]))


# ── Feature extraction ───────────────────────────────────────────────────

def test_feature_names_stable_order():
    names = xmax_features.feature_names()
    assert len(names) == len(set(names))
    assert names[:4] == ("alpha_l", "alpha_t", "ca", "gamma")


def test_extract_features_circle_matches_feature_names():
    feats = xmax_features.extract_features(_circle_config())
    assert set(feats.keys()) == set(xmax_features.feature_names())
    assert feats["n_elements"] == 2
    assert feats["x_min_src"] == 10.0
    assert feats["x_max_src"] == 12.0
    assert feats["has_acceptor"] == 1.0  # one element has c < 0
    assert feats["frac_donor"] == 0.5


def test_extract_features_line():
    feats = xmax_features.extract_features(_line_config())
    assert feats["n_elements"] == 1
    assert feats["total_area"] > 0
    assert feats["c_max"] == feats["c_min"] == 10.0


def test_extract_features_ellipse():
    feats = xmax_features.extract_features(_ellipse_config())
    assert feats["n_elements"] == 1
    # area = pi * a * b
    assert feats["total_area"] == pytest.approx(2.0 * 1.0 * 3.141592653589793, rel=1e-6)


def test_orientation_flag():
    horiz = xmax_features.extract_features(_circle_config("horizontal"))
    vert = xmax_features.extract_features(_ellipse_config("vertical"))
    assert horiz["orientation_vertical"] == 0.0
    assert vert["orientation_vertical"] == 1.0


def test_extract_features_rejects_empty_config():
    empty = ATConfiguration.from_dict(_base("horizontal", []))
    with pytest.raises(ValueError):
        xmax_features.extract_features(empty)


# ── Predictor ────────────────────────────────────────────────────────────

def test_predictor_cold_start_falls_back_to_legacy_rule(tmp_path):
    predictor = xmax_predictor.XmaxPredictor(model_path=str(tmp_path / "model.pkl"))
    feats = xmax_features.extract_features(_circle_config())
    pred = predictor.predict(feats)
    assert pred["trained"] is False
    assert pred["dom_xmax"] == pytest.approx(feats["x_max_src"] + xmax_predictor.LEGACY_MARGIN)


def test_predictor_trains_after_min_rows(tmp_path):
    predictor = xmax_predictor.XmaxPredictor(
        model_path=str(tmp_path / "model.pkl"), min_train=5,
    )
    feats = xmax_features.extract_features(_circle_config())
    for i in range(5):
        row = dict(feats)
        row["x_max_src"] = feats["x_max_src"] + i  # vary inputs slightly
        predictor.update(row, L_max=100.0 + 10 * i)
    predictor.refit()

    pred = predictor.predict(feats)
    assert pred["trained"] is True
    assert pred["n_train"] == 5
    assert pred["dom_xmax"] == pytest.approx(pred["L_pred_q"] * predictor.margin)


def test_predictor_persists_and_reloads(tmp_path):
    path = str(tmp_path / "model.pkl")
    feats = xmax_features.extract_features(_circle_config())

    p1 = xmax_predictor.XmaxPredictor(model_path=path, min_train=3)
    for i in range(3):
        row = dict(feats)
        row["x_max_src"] = feats["x_max_src"] + i
        p1.update(row, L_max=90.0 + 5 * i)
    p1.refit()
    p1.save()

    p2 = xmax_predictor.XmaxPredictor(model_path=path, min_train=3)
    assert len(p2.y) == 3
    pred = p2.predict(feats)
    assert pred["trained"] is True


def test_predictor_out_of_box_falls_back(tmp_path):
    predictor = xmax_predictor.XmaxPredictor(
        model_path=str(tmp_path / "model.pkl"), min_train=3,
    )
    feats = xmax_features.extract_features(_circle_config())
    for i in range(3):
        row = dict(feats)
        row["x_max_src"] = feats["x_max_src"] + i
        predictor.update(row, L_max=90.0 + 5 * i)
    predictor.refit()

    far_out = dict(feats)
    far_out["x_max_src"] = feats["x_max_src"] + 10_000.0
    pred = predictor.predict(far_out)
    assert pred["trained"] is False
    assert pred["dom_xmax"] == pytest.approx(far_out["x_max_src"] + xmax_predictor.LEGACY_MARGIN)
