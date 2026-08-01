# Written by Alvin Yadav

"""
AEM Source Designer

Holds all data classes, packing logic, JSON export/import, and validation.
Scene is the single source of truth; GlobalParams replaces module constants.
"""
from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, Point

# ── Shared constants (visual/interaction constants stay in source_designer.py) ─
MIN_RADIUS     = 0.003
DEFAULT_RADIUS = 0.02
DEFAULT_CONC   = 10.0
CONC_MAX       = 50.0
GRID_SPACING   = 0.05
RADIUS_STEP    = 0.002
CONC_STEP      = 2.0


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SimpleSource:
    """A standalone source element: circle, ellipse, or line."""
    kind: str           # "circle" | "ellipse" | "line"
    x: float
    y: float
    c: float
    r: float | None = None      # circle radius
    a: float | None = None      # ellipse semi-major
    b: float | None = None      # ellipse semi-minor
    l: float | None = None      # line full length
    theta: float = 0.0          # degrees (0 = horizontal)
    id: str | None = None       # semantic/role label (donor, acceptor, …)
    index: int | None = None    # designer identity index (NOT the role id)

    def half_height(self) -> float:
        """Vertical half-extent (for vertical-shift computation)."""
        if self.kind == "circle":
            return self.r or 0.0
        if self.kind == "ellipse":
            a = self.a or 0.0
            b = self.b or 0.0
            t = math.radians(self.theta)
            return math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
        if self.kind == "line":
            t = math.radians(self.theta)
            return (self.l or 0.0) / 2.0 * abs(math.sin(t))
        return 0.0

    def half_width(self) -> float:
        """Horizontal half-extent."""
        if self.kind == "circle":
            return self.r or 0.0
        if self.kind == "ellipse":
            a = self.a or 0.0
            b = self.b or 0.0
            t = math.radians(self.theta)
            return math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
        if self.kind == "line":
            t = math.radians(self.theta)
            return (self.l or 0.0) / 2.0 * abs(math.cos(t))
        return 0.0

    def to_element_dict(self, elem_id: str) -> dict:
        """Emit the solver-format element dict for this source."""
        d: dict[str, Any] = {
            "kind": self.kind, "x": self.x, "y": self.y,
            "c": self.c, "id": elem_id,
        }
        if self.index is not None:
            d["index"] = self.index
        if self.kind == "circle":
            d["r"] = self.r
        elif self.kind == "ellipse":
            d["a"] = self.a
            d["b"] = self.b
            if self.theta != 0.0:
                d["theta"] = self.theta
        elif self.kind == "line":
            d["l"] = self.l
            d["theta"] = self.theta   # always emit; default 90° is meaningful
        return d

    def contains_point(self, wx: float, wy: float) -> bool:
        """Return True if (wx, wy) is inside (or very near) this source."""
        dx, dy = wx - self.x, wy - self.y
        if self.kind == "circle":
            r = self.r or 0.0
            return dx * dx + dy * dy <= (r * 1.5) ** 2
        if self.kind == "ellipse":
            a = self.a or DEFAULT_RADIUS
            b = self.b or DEFAULT_RADIUS
            t = math.radians(self.theta)
            ct, st = math.cos(t), math.sin(t)
            xp =  ct * dx + st * dy
            yp = -st * dx + ct * dy
            return (xp / a) ** 2 + (yp / b) ** 2 <= 1.5 ** 2
        if self.kind == "line":
            # Distance from point to line segment
            l = self.l or 0.0
            t = math.radians(self.theta)
            ux, uy = math.cos(t), math.sin(t)
            proj = dx * ux + dy * uy
            proj = max(-l / 2, min(l / 2, proj))
            cx = proj * ux
            cy = proj * uy
            dist = math.hypot(dx - cx, dy - cy)
            return dist <= max(DEFAULT_RADIUS * 3, 0.01)
        return False


@dataclass
class CompositeSource:
    """A polygon filled with packed circles."""
    vertices: list[list[float]]
    circles: list[dict]       # each: {"x", "y", "r", "c"}
    base_c: float = DEFAULT_CONC
    id: str | None = None


@dataclass
class GlobalParams:
    """All solver and designer-global parameters."""
    alpha_l: float = 2.0
    alpha_t: float = 0.2
    ca: float = 8.0
    gamma: float = 3.5
    dom_inc: float = 1.0
    num_cp: int = 40
    num_terms: int = 5
    orientation: str = "vertical"
    plot_aspect: str = ""
    ws: float = 0.5              # designer-only source-zone width guide
    dom_override: dict | None = None


@dataclass
class Scene:
    """Single source of truth for the designer state."""
    globals: GlobalParams = field(default_factory=GlobalParams)
    simple_sources: list[SimpleSource] = field(default_factory=list)
    composites: list[CompositeSource] = field(default_factory=list)

    # ── Export ──────────────────────────────────────────────────────────────

    def _flat_element_dicts(self) -> list[dict]:
        """Return all elements as flat list of export-ready dicts (no shift)."""
        elems: list[dict] = []
        for comp in self.composites:
            for c in comp.circles:
                eid = c.get("id", f"src_{len(elems)}")
                d = {
                    "kind": "circle",
                    "x": c["x"], "y": c["y"],
                    "r": c["r"], "c": c["c"],
                    "id": eid,
                }
                if c.get("index") is not None:
                    d["index"] = c["index"]
                elems.append(d)
        for ss in self.simple_sources:
            eid = ss.id or f"src_{len(elems)}"
            elems.append(ss.to_element_dict(eid))
        return elems

    # ── Indexing (designer identity, NOT the role `id`) ───────────────────────

    def _index_records(self) -> list[tuple]:
        """All indexable user elements as (y, x, kind, obj) records."""
        recs = []
        for comp in self.composites:
            for c in comp.circles:
                recs.append((c["y"], c["x"], "dict", c))
        for ss in self.simple_sources:
            recs.append((ss.y, ss.x, "ss", ss))
        return recs

    @staticmethod
    def _get_index(rec) -> int | None:
        _, _, kind, obj = rec
        return obj.get("index") if kind == "dict" else obj.index

    @staticmethod
    def _set_index(rec, value):
        _, _, kind, obj = rec
        if kind == "dict":
            obj["index"] = value
        else:
            obj.index = value

    def next_index(self) -> int:
        """Smallest non-negative integer not currently used as an index."""
        used = {self._get_index(r) for r in self._index_records()}
        used.discard(None)
        i = 0
        while i in used:
            i += 1
        return i

    def reindex_spatial(self) -> bool:
        """
        Assign contiguous 0-based indices in spatial order (top-to-bottom,
        then left-to-right: sort by (-y, x)). Returns True if anything changed.
        """
        recs = self._index_records()
        recs.sort(key=lambda r: (-r[0], r[1]))
        changed = False
        for i, rec in enumerate(recs):
            if self._get_index(rec) != i:
                changed = True
            self._set_index(rec, i)
        return changed

    def indices_are_clean(self) -> bool:
        """True if current indices are exactly the contiguous set 0..N-1."""
        idx = [self._get_index(r) for r in self._index_records()]
        if any(i is None for i in idx):
            return False
        return sorted(idx) == list(range(len(idx)))

    def ensure_display_indices(self):
        """Give every element some unique index for display (preserve existing)."""
        used = set()
        missing = []
        for rec in self._index_records():
            i = self._get_index(rec)
            if i is None or i in used:
                missing.append(rec)
            else:
                used.add(i)
        nxt = 0
        for rec in missing:
            while nxt in used:
                nxt += 1
            self._set_index(rec, nxt)
            used.add(nxt)

    @staticmethod
    def _elem_half_height(e: dict) -> float:
        kind = e["kind"]
        if kind == "circle":
            return e.get("r", 0.0) or 0.0
        if kind == "ellipse":
            a = e.get("a", 0.0) or 0.0
            b = e.get("b", 0.0) or 0.0
            t = math.radians(e.get("theta", 0.0) or 0.0)
            return math.sqrt((a * math.sin(t)) ** 2 + (b * math.cos(t)) ** 2)
        if kind == "line":
            t = math.radians(e.get("theta", 0.0) or 0.0)
            return (e.get("l", 0.0) or 0.0) / 2.0 * abs(math.sin(t))
        return 0.0

    @staticmethod
    def _elem_half_width(e: dict) -> float:
        kind = e["kind"]
        if kind == "circle":
            return e.get("r", 0.0) or 0.0
        if kind == "ellipse":
            a = e.get("a", 0.0) or 0.0
            b = e.get("b", 0.0) or 0.0
            t = math.radians(e.get("theta", 0.0) or 0.0)
            return math.sqrt((a * math.cos(t)) ** 2 + (b * math.sin(t)) ** 2)
        if kind == "line":
            t = math.radians(e.get("theta", 0.0) or 0.0)
            return (e.get("l", 0.0) or 0.0) / 2.0 * abs(math.cos(t))
        return 0.0

    def to_config_dict(self) -> dict:
        """
        Export the scene to a solver-compatible config dict.

        Coordinates are PHYSICAL and authoritative — no shifting. In vertical
        orientation the water table is at y=0 and all sources must already lie
        below it (validate_for_export enforces this); in horizontal orientation
        y is the transverse axis. Geometry is exported exactly as authored.
        """
        errors = self.validate_for_export()
        if errors:
            raise ValueError("\n".join(errors))

        # Renumber to clean contiguous 0-based spatial indices so the exported
        # JSON is always tidy and matches the on-canvas labels.
        self.reindex_spatial()

        g = self.globals
        elements = self._flat_element_dicts()

        # Domain derivation (override wins)
        if g.dom_override:
            dom = dict(g.dom_override)
        else:
            all_x = [e["x"] for e in elements]
            all_y = [e["y"] for e in elements]
            max_hw = max(self._elem_half_width(e) for e in elements)
            max_hh = max(self._elem_half_height(e) for e in elements)

            # Scale-aware x-padding. The original tool used a fixed +150 m,
            # which is appropriate for metre-scale sources whose plumes travel
            # far, but pathological for sub-metre sources: it produces a huge,
            # mostly-empty grid and lets exp(beta*x) blow up far downstream
            # (slow + numerically extreme). Scale the padding to the source
            # size and cap it at the original 150 m. The solver's dynamic
            # domain extension grows it further if the plume actually needs it.
            x_span = max(all_x) - min(all_x)
            char = max(2.0 * max_hw, 2.0 * max_hh, x_span, 1.0)
            x_pad = min(150.0, max(20.0, 25.0 * char))

            if g.orientation == "vertical":
                dom_ymax = 0.0
            else:
                dom_ymax = round(max(all_y) + max_hh + 5.0, 3)
            dom = {
                "dom_xmin": round(min(all_x) - max_hw - 2.0, 3),
                "dom_xmax": round(max(all_x) + x_pad, 1),
                "dom_ymin": round(min(all_y) - max_hh - 5.0, 3),
                "dom_ymax": dom_ymax,
            }

        return {
            "alpha_l": g.alpha_l, "alpha_t": g.alpha_t,
            "ca": g.ca, "gamma": g.gamma,
            **dom,
            "dom_inc": g.dom_inc,
            "num_cp": g.num_cp,
            "num_terms": g.num_terms,
            "orientation": g.orientation,
            "plot_aspect": g.plot_aspect,
            "elements": elements,
        }

    # ── Import ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config_dict(cls, data: dict) -> "Scene":
        """
        Parse a solver-format config dict back into a Scene.

        Every element becomes an independent SimpleSource (flat import, option a
        from the plan). Composite reconstruction is lossy — the polygon is gone.
        Geometry is imported as-is (physical coordinates), so import→export is
        idempotent.
        """
        g = GlobalParams(
            alpha_l=float(data.get("alpha_l", 2.0)),
            alpha_t=float(data.get("alpha_t", 0.2)),
            ca=float(data.get("ca", 8.0)),
            gamma=float(data.get("gamma", 3.5)),
            dom_inc=float(data.get("dom_inc", 1.0)),
            num_cp=int(data.get("num_cp", 40)),
            num_terms=int(data.get("num_terms", 5)),
            orientation=data.get("orientation", "vertical"),
            plot_aspect=data.get("plot_aspect", ""),
        )
        scene = cls(globals=g)
        for i, e in enumerate(data.get("elements", [])):
            kind = e["kind"].lower()
            idx = e.get("index", None)
            ss = SimpleSource(
                kind=kind,
                x=float(e["x"]),
                y=float(e["y"]),
                c=float(e["c"]),
                id=e.get("id", f"src_{i}"),
                index=int(idx) if idx is not None else None,
            )
            if kind == "circle":
                ss.r = float(e["r"])
            elif kind == "ellipse":
                ss.a = float(e["a"])
                ss.b = float(e["b"])
                ss.theta = float(e.get("theta", 0.0))
            elif kind == "line":
                ss.l = float(e["l"])
                ss.theta = float(e.get("theta", 90.0))
            else:
                raise ValueError(f"Unknown element kind in import: {e['kind']!r}")
            scene.simple_sources.append(ss)
        # Preserve provided indices; fill any missing/duplicate ones for display.
        scene.ensure_display_indices()
        return scene

    # ── Validation ──────────────────────────────────────────────────────────

    @staticmethod
    def _elem_solver_top(e: dict) -> float:
        """
        Highest y the solver sees for this element in vertical orientation,
        matching at_simulation's check  elem.y + r*sin(theta).
        (circle: theta=90°→r;  ellipse: a*|sinθ|;  line: (l/2)*|sinθ|.)
        """
        kind = e["kind"]
        if kind == "circle":
            return e["y"] + (e.get("r", 0.0) or 0.0)
        th = math.radians(e.get("theta", 90.0) or 0.0)
        if kind == "ellipse":
            return e["y"] + (e.get("a", 0.0) or 0.0) * abs(math.sin(th))
        if kind == "line":
            return e["y"] + (e.get("l", 0.0) or 0.0) / 2.0 * abs(math.sin(th))
        return e["y"]

    def validate_for_export(self) -> list[str]:
        """Return a list of error strings. Empty list = scene is valid."""
        errors: list[str] = []
        g = self.globals
        if not self.composites and not self.simple_sources:
            errors.append("Scene is empty — add at least one source element.")
        if g.num_terms < 1:
            errors.append("num_terms must be ≥ 1.")
        if g.num_cp < g.num_terms:
            errors.append(f"num_cp ({g.num_cp}) must be ≥ num_terms ({g.num_terms}).")
        if g.alpha_l <= 0:
            errors.append("alpha_l must be > 0.")
        if g.alpha_t <= 0:
            errors.append("alpha_t must be > 0.")
        if g.gamma <= 0:
            errors.append("gamma must be > 0.")

        # Vertical orientation: every source must sit fully below the water
        # table (y=0). The solver requires elem.y + r*sinθ < -0.1.
        if g.orientation == "vertical" and self.has_elements():
            offenders = []
            for e in self._flat_element_dicts():
                if self._elem_solver_top(e) >= -0.1:
                    offenders.append(e.get("id", "?"))
            if offenders:
                shown = ", ".join(offenders[:3]) + ("…" if len(offenders) > 3 else "")
                errors.append(
                    f"{len(offenders)} element(s) above the water table ({shown}). "
                    "In vertical orientation every source must lie below y=0.")
        return errors

    def warnings_for_export(self) -> list[str]:
        """Non-blocking warnings (return even when export is valid)."""
        warnings: list[str] = []
        g = self.globals
        if g.alpha_t > 0:
            ratio = g.alpha_l / g.alpha_t
            if ratio > 100 or ratio < 1:
                warnings.append(
                    f"alpha_l/alpha_t = {ratio:.1f} is extreme — "
                    "thin/wide plumes can be slow to solve."
                )
        return warnings

    def has_elements(self) -> bool:
        return bool(self.composites or self.simple_sources)


# ── Circle packing ─────────────────────────────────────────────────────────────

def _distance_to_boundary(x: float, y: float, poly: ShapelyPolygon) -> float:
    """Signed distance to polygon boundary (positive = inside)."""
    p = Point(x, y)
    d = poly.exterior.distance(p)
    return d if poly.contains(p) else -d


def _max_radius_at(x: float, y: float,
                   placed: list[tuple[float, float, float]],
                   poly: ShapelyPolygon,
                   abs_max: float = 0.06) -> float:
    """Largest non-overlapping circle radius at (x, y)."""
    r = _distance_to_boundary(x, y, poly)
    if r <= 0:
        return 0.0
    for cx, cy, cr in placed:
        r = min(r, math.hypot(x - cx, y - cy) - cr)
        if r <= MIN_RADIUS:
            return 0.0
    return min(r, abs_max)


def greedy_circle_pack(vertices: list[list[float]],
                       default_c: float = DEFAULT_CONC,
                       max_circles: int = 80) -> list[dict]:
    """
    Greedy largest-gap-first circle packing for a polygon.
    Returns a list of circle dicts: {x, y, r, c}.
    """
    poly = ShapelyPolygon(vertices)
    if not poly.is_valid or poly.area == 0:
        return []

    minx, miny, maxx, maxy = poly.bounds
    abs_max_r = min(maxx - minx, maxy - miny) * 0.25

    placed: list[tuple[float, float, float]] = []
    circles: list[dict] = []

    step = MIN_RADIUS * 2
    cand_xs = np.arange(minx + MIN_RADIUS, maxx - MIN_RADIUS + step, step)
    cand_ys = np.arange(miny + MIN_RADIUS, maxy - MIN_RADIUS + step, step)
    candidates = [
        (x, y)
        for x in cand_xs for y in cand_ys
        if poly.contains(Point(x, y))
    ]

    for _ in range(max_circles):
        best_r, best_x, best_y = 0.0, 0.0, 0.0
        surviving = []
        for (x, y) in candidates:
            r = _max_radius_at(x, y, placed, poly, abs_max_r)
            if r > MIN_RADIUS:
                surviving.append((x, y))
                if r > best_r:
                    best_r, best_x, best_y = r, x, y
        candidates = surviving
        if best_r <= MIN_RADIUS:
            break
        placed.append((best_x, best_y, best_r))
        circles.append({
            "x": round(best_x, 5), "y": round(best_y, 5),
            "r": round(best_r, 5), "c": default_c,
        })

    return circles


def repack_after_resize(circles: list[dict],
                        changed_idx: int,
                        polygon_vertices: list[list[float]]) -> list[dict]:
    """
    After one circle is resized, push overlapping circles apart then grow
    neighbours into freed space. The resized circle stays fixed.
    """
    poly = ShapelyPolygon(polygon_vertices)
    if not poly.is_valid:
        return circles

    # Phase 1 — resolve overlaps
    for _it in range(80):
        moved = False
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                ci, cj = circles[i], circles[j]
                dx = ci["x"] - cj["x"]
                dy = ci["y"] - cj["y"]
                dist = math.hypot(dx, dy)
                need = ci["r"] + cj["r"]
                if dist < need and dist > 1e-9:
                    overlap = need - dist
                    nx, ny = dx / dist, dy / dist
                    if i == changed_idx:
                        cj["x"] = round(cj["x"] - nx * overlap, 5)
                        cj["y"] = round(cj["y"] - ny * overlap, 5)
                    elif j == changed_idx:
                        ci["x"] = round(ci["x"] + nx * overlap, 5)
                        ci["y"] = round(ci["y"] + ny * overlap, 5)
                    else:
                        ci["x"] = round(ci["x"] + nx * overlap * 0.5, 5)
                        ci["y"] = round(ci["y"] + ny * overlap * 0.5, 5)
                        cj["x"] = round(cj["x"] - nx * overlap * 0.5, 5)
                        cj["y"] = round(cj["y"] - ny * overlap * 0.5, 5)
                    moved = True
        for i, ci in enumerate(circles):
            if i == changed_idx:
                continue
            p = Point(ci["x"], ci["y"])
            if not poly.contains(p):
                nearest = poly.exterior.interpolate(poly.exterior.project(p))
                ci["x"] = round((ci["x"] + nearest.x) / 2, 5)
                ci["y"] = round((ci["y"] + nearest.y) / 2, 5)
                moved = True
            while (ci["r"] > MIN_RADIUS and
                   not poly.contains(Point(ci["x"], ci["y"]).buffer(ci["r"] * 0.8))):
                ci["r"] = round(ci["r"] * 0.9, 5)
                moved = True
        if not moved:
            break

    changed_obj = circles[changed_idx]
    circles = [c for i, c in enumerate(circles)
               if c["r"] >= MIN_RADIUS and (
                   c is changed_obj or poly.contains(Point(c["x"], c["y"])))]
    changed_idx = next(i for i, c in enumerate(circles) if c is changed_obj)

    # Phase 2 — grow neighbours into freed space
    ch = circles[changed_idx]
    for _round in range(40):
        grew = False
        for i, ci in enumerate(circles):
            if i == changed_idx:
                continue
            max_r = _distance_to_boundary(ci["x"], ci["y"], poly)
            for j, cj in enumerate(circles):
                if j == i:
                    continue
                gap = math.hypot(ci["x"] - cj["x"], ci["y"] - cj["y"]) - cj["r"]
                max_r = min(max_r, gap)
            target = max_r * 0.95
            if target > ci["r"] + 0.0002:
                ci["r"] = round(ci["r"] + (target - ci["r"]) * 0.6, 5)
                grew = True

            dx = ch["x"] - ci["x"]
            dy = ch["y"] - ci["y"]
            dist_to_ch = math.hypot(dx, dy)
            if dist_to_ch < 1e-9:
                continue
            edge_gap = dist_to_ch - ci["r"] - ch["r"]
            if edge_gap > 0.001:
                step = min(edge_gap * 0.3, 0.002)
                nx, ny = dx / dist_to_ch, dy / dist_to_ch
                new_x = ci["x"] + nx * step
                new_y = ci["y"] + ny * step
                if poly.contains(Point(new_x, new_y).buffer(ci["r"] * 0.8)):
                    ok = True
                    for j, cj in enumerate(circles):
                        if j == i or j == changed_idx:
                            continue
                        if math.hypot(new_x - cj["x"], new_y - cj["y"]) < ci["r"] + cj["r"]:
                            ok = False
                            break
                    if ok:
                        ci["x"] = round(new_x, 5)
                        ci["y"] = round(new_y, 5)
                        grew = True
        if not grew:
            break

    return circles
