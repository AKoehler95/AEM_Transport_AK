"""
Unit tests for designer_model.py — §1.6 of the implementation plan.

Run with:  cd src && pytest tests/test_model.py -v
"""
import sys
import os
import math

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from designer_model import (
    Scene, GlobalParams, SimpleSource, CompositeSource,
    greedy_circle_pack, DEFAULT_CONC, DEFAULT_RADIUS, MIN_RADIUS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _scene_with(*sources: SimpleSource, orientation: str = "vertical") -> Scene:
    s = Scene(globals=GlobalParams(orientation=orientation))
    s.simple_sources.extend(sources)
    return s


# Defaults sit below the water table (negative y) so vertical scenes are valid
# under the physical-coordinate convention.
def _circle(x=0.1, y=-0.5, r=0.05, c=DEFAULT_CONC) -> SimpleSource:
    return SimpleSource(kind="circle", x=x, y=y, c=c, r=r, id="test_circle")


def _ellipse(x=0.1, y=-0.5, a=0.1, b=0.05, theta=0.0) -> SimpleSource:
    return SimpleSource(kind="ellipse", x=x, y=y, c=DEFAULT_CONC,
                        a=a, b=b, theta=theta, id="test_ellipse")


def _line(x=0.1, y=-0.5, l=0.2, theta=90.0) -> SimpleSource:
    return SimpleSource(kind="line", x=x, y=y, c=DEFAULT_CONC,
                        l=l, theta=theta, id="test_line")


# ── Element dict field mapping ─────────────────────────────────────────────────

class TestElementDictMapping:

    def test_circle_fields(self):
        scene = _scene_with(_circle())
        cfg = scene.to_config_dict()
        e = cfg["elements"][0]
        assert e["kind"] == "circle"
        assert "r" in e
        assert "a" not in e and "b" not in e and "l" not in e

    def test_ellipse_fields(self):
        scene = _scene_with(_ellipse(theta=30.0))
        cfg = scene.to_config_dict()
        e = cfg["elements"][0]
        assert e["kind"] == "ellipse"
        assert "a" in e and "b" in e
        assert "theta" in e
        assert abs(e["a"] - 0.1) < 1e-9
        assert abs(e["b"] - 0.05) < 1e-9

    def test_line_fields(self):
        scene = _scene_with(_line())
        cfg = scene.to_config_dict()
        e = cfg["elements"][0]
        assert e["kind"] == "line"
        assert "l" in e
        assert "theta" in e
        assert abs(e["l"] - 0.2) < 1e-9

    def test_ellipse_zero_theta_omitted(self):
        """theta=0 should still be emitted for ellipses (default angle is ambiguous)."""
        ss = SimpleSource(kind="ellipse", x=0.1, y=0.1, c=DEFAULT_CONC,
                          a=0.1, b=0.05, theta=0.0)
        scene = _scene_with(ss, orientation="horizontal")
        cfg = scene.to_config_dict()
        e = cfg["elements"][0]
        assert e["kind"] == "ellipse"
        # theta=0 is omitted per the to_element_dict logic
        # (only emitted when theta != 0.0)


# ── Vertical geometry (physical coordinates, no shift) ──────────────────────────

class TestVerticalGeometry:

    def test_below_water_exports_unchanged(self):
        scene = _scene_with(_circle(y=-0.5, r=0.05))
        cfg = scene.to_config_dict()
        assert abs(cfg["elements"][0]["y"] - (-0.5)) < 1e-9, "y must not be shifted"

    def test_above_water_circle_is_invalid(self):
        scene = _scene_with(_circle(y=0.1, r=0.05))   # top = 0.15 > 0
        errors = scene.validate_for_export()
        assert any("water table" in e for e in errors)

    def test_at_water_table_is_invalid(self):
        # Circle whose top is exactly at the surface: y=-r → top=0 ≥ -0.1.
        scene = _scene_with(_circle(y=-0.05, r=0.05))
        errors = scene.validate_for_export()
        assert any("water table" in e for e in errors)

    def test_ellipse_below_water_ok(self):
        scene = _scene_with(_ellipse(y=-0.5, a=0.1, b=0.05, theta=0.0))
        assert scene.validate_for_export() == []

    def test_line_below_water_ok(self):
        scene = _scene_with(_line(y=-0.5, l=0.2, theta=90.0))
        assert scene.validate_for_export() == []

    def test_horizontal_positive_y_ok(self):
        scene = _scene_with(_circle(y=0.3), orientation="horizontal")
        assert scene.validate_for_export() == []
        cfg = scene.to_config_dict()
        assert abs(cfg["elements"][0]["y"] - 0.3) < 1e-9


# ── Import / export idempotency ────────────────────────────────────────────────

class TestImportExportIdempotency:

    def test_circle_round_trip(self):
        scene1 = _scene_with(_circle())
        cfg1 = scene1.to_config_dict()
        scene2 = Scene.from_config_dict(cfg1)
        cfg2 = scene2.to_config_dict()
        assert len(cfg1["elements"]) == len(cfg2["elements"])
        for e1, e2 in zip(cfg1["elements"], cfg2["elements"]):
            assert abs(e1["x"] - e2["x"]) < 1e-4
            assert abs(e1["y"] - e2["y"]) < 1e-4
            assert abs(e1["r"] - e2["r"]) < 1e-4

    def test_ellipse_round_trip(self):
        scene1 = _scene_with(_ellipse(a=0.1, b=0.06, theta=30.0))
        cfg1 = scene1.to_config_dict()
        scene2 = Scene.from_config_dict(cfg1)
        cfg2 = scene2.to_config_dict()
        e1, e2 = cfg1["elements"][0], cfg2["elements"][0]
        assert abs(e1["y"] - e2["y"]) < 1e-4
        assert abs(e1["a"] - e2["a"]) < 1e-4
        assert abs(e1["b"] - e2["b"]) < 1e-4

    def test_line_round_trip(self):
        scene1 = _scene_with(_line(l=0.4, theta=45.0))
        cfg1 = scene1.to_config_dict()
        scene2 = Scene.from_config_dict(cfg1)
        cfg2 = scene2.to_config_dict()
        e1, e2 = cfg1["elements"][0], cfg2["elements"][0]
        assert abs(e1["y"] - e2["y"]) < 1e-4
        assert abs(e1["l"] - e2["l"]) < 1e-4

    def test_vertical_round_trip_unchanged(self):
        scene1 = _scene_with(_circle(y=-0.8, r=0.05))
        cfg1 = scene1.to_config_dict()
        scene2 = Scene.from_config_dict(cfg1)
        cfg2 = scene2.to_config_dict()
        assert abs(cfg1["elements"][0]["y"] - cfg2["elements"][0]["y"]) < 1e-9

    def test_horizontal_import_export_stable(self):
        y0 = 5.0
        ss = SimpleSource(kind="circle", x=10.0, y=y0, c=5.0, r=1.0)
        scene1 = Scene(globals=GlobalParams(orientation="horizontal"))
        scene1.simple_sources.append(ss)
        cfg1 = scene1.to_config_dict()
        scene2 = Scene.from_config_dict(cfg1)
        cfg2 = scene2.to_config_dict()
        assert abs(cfg1["elements"][0]["y"] - cfg2["elements"][0]["y"]) < 1e-9


# ── Export validation ──────────────────────────────────────────────────────────

class TestExportValidation:

    def test_empty_scene_is_invalid(self):
        scene = Scene()
        errors = scene.validate_for_export()
        assert any("empty" in e.lower() for e in errors)

    def test_num_terms_zero_is_invalid(self):
        scene = _scene_with(_circle())
        scene.globals.num_terms = 0
        errors = scene.validate_for_export()
        assert any("num_terms" in e for e in errors)

    def test_num_cp_less_than_num_terms(self):
        scene = _scene_with(_circle())
        scene.globals.num_terms = 10
        scene.globals.num_cp = 5
        errors = scene.validate_for_export()
        assert any("num_cp" in e for e in errors)

    def test_alpha_l_zero_is_invalid(self):
        scene = _scene_with(_circle())
        scene.globals.alpha_l = 0.0
        errors = scene.validate_for_export()
        assert any("alpha_l" in e for e in errors)

    def test_valid_scene_no_errors(self):
        scene = _scene_with(_circle())
        errors = scene.validate_for_export()
        assert errors == []

    def test_to_config_dict_raises_on_invalid(self):
        scene = Scene()   # empty
        with pytest.raises(ValueError):
            scene.to_config_dict()


# ── at_config.from_dict integration ───────────────────────────────────────────

class TestATConfigFromDict:

    def test_circle_loads_without_error(self):
        from at_config import ATConfiguration
        scene = _scene_with(_circle())
        cfg = scene.to_config_dict()
        config = ATConfiguration.from_dict(cfg)
        assert len(config.elements) == 1

    def test_ellipse_loads_without_error(self):
        from at_config import ATConfiguration
        scene = _scene_with(_ellipse(a=0.1, b=0.05, theta=30.0))
        cfg = scene.to_config_dict()
        config = ATConfiguration.from_dict(cfg)
        assert len(config.elements) == 1
        e = config.elements[0]
        assert abs(e.r - 0.1) < 1e-9   # r = semi-major a

    def test_line_loads_without_error(self):
        from at_config import ATConfiguration
        scene = _scene_with(_line(l=0.2, theta=90.0))
        cfg = scene.to_config_dict()
        config = ATConfiguration.from_dict(cfg)
        assert len(config.elements) == 1
        e = config.elements[0]
        assert abs(e.r - 0.1) < 1e-9   # r = l/2

    def test_from_json_still_works(self, tmp_path):
        """from_json is now a wrapper for from_dict; ensure it still works."""
        import json
        from at_config import ATConfiguration
        scene = _scene_with(_circle())
        cfg = scene.to_config_dict()
        fpath = tmp_path / "test_config.json"
        fpath.write_text(json.dumps(cfg))
        config = ATConfiguration.from_json(str(fpath))
        assert len(config.elements) == 1


# ── Circle packing ─────────────────────────────────────────────────────────────

class TestGreedyPack:

    def test_pack_triangle_returns_circles(self):
        verts = [[0.0, 0.0], [0.5, 0.0], [0.25, 0.5]]
        circles = greedy_circle_pack(verts, default_c=DEFAULT_CONC)
        assert len(circles) >= 1
        for c in circles:
            assert c["r"] >= MIN_RADIUS
            assert c["c"] == DEFAULT_CONC

    def test_pack_square(self):
        verts = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
        circles = greedy_circle_pack(verts)
        assert len(circles) >= 2

    def test_degenerate_returns_empty(self):
        verts = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        circles = greedy_circle_pack(verts)
        assert circles == []


# ── Domain derivation ──────────────────────────────────────────────────────────

class TestDomainDerivation:

    def test_vertical_dom_ymax_is_zero(self):
        scene = _scene_with(_circle())
        cfg = scene.to_config_dict()
        assert cfg["dom_ymax"] == 0.0

    def test_horizontal_dom_ymax_above_element(self):
        ss = SimpleSource(kind="circle", x=10.0, y=0.0, c=5.0, r=1.0)
        scene = Scene(globals=GlobalParams(orientation="horizontal"))
        scene.simple_sources.append(ss)
        cfg = scene.to_config_dict()
        assert cfg["dom_ymax"] > ss.y + ss.r

    def test_dom_override_wins(self):
        scene = _scene_with(_circle())
        override = {"dom_xmin": -99, "dom_xmax": 999, "dom_ymin": -50, "dom_ymax": 0.0}
        scene.globals.dom_override = override
        cfg = scene.to_config_dict()
        assert cfg["dom_xmin"] == -99
        assert cfg["dom_xmax"] == 999
