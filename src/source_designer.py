# Written by Alvin Yadav

"""
AEM Source Designer
Interactive matplotlib tool for drawing and editing source zones, then solving
the AEM transport model inline.

Tool palette (top row of buttons, or number keys 1–5):
  1 Polygon  — draw polygon vertices, right-click to close → greedy pack
  2 Circle   — click-drag to place a circle
  3 Ellipse  — click-drag bounding box to place an axis-aligned ellipse
  4 Line     — click start, click end
  5 Select   — click to select; drag to move; keyboard or fields to edit

Global keyboard shortcuts:
  1–5  switch tool   d Polygon   e Select   v View   s Export   o Load   q Quit

Select-mode key actions (when an element is selected):
  ↑/↓ size      8/9 ellipse b     [ ] , . rotate     +/- conc
  Del delete    x delete polygon

Both modes: scroll = zoom, middle/Shift+drag = pan.

Settings are an in-window panel (the ⚙ button), not a separate dialog.
"""

import json
import math
import queue
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
# Backend: use the native "macosx" backend on macOS for crisp Retina rendering;
# TkAgg elsewhere. Settings are an in-figure panel (matplotlib widgets), so the
# only Tk dependency is the file open/save dialogs, which are created transiently.
if sys.platform == "darwin":
    matplotlib.use("macosx")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, TextBox
from matplotlib.lines import Line2D

from designer_model import (
    Scene, GlobalParams, SimpleSource, CompositeSource,
    greedy_circle_pack, repack_after_resize,
    DEFAULT_CONC, DEFAULT_RADIUS, MIN_RADIUS,
    CONC_MAX, GRID_SPACING, RADIUS_STEP, CONC_STEP,
)

# ── Visual constants (view-only, not exported) ─────────────────────────────────
GRID_HEIGHT  = 15.0
DONOR_CMAP   = plt.cm.Reds
DONOR_NORM   = mcolors.Normalize(vmin=0, vmax=CONC_MAX)
SNAP_TO_GRID = True

# ── Theme (Option A restyle: indigo accent on warm-grey light) ─────────────────
SEL_COLOR  = "#4F46E5"         # indigo accent: selection / headers / hover
SEL_FILL   = "#EEF0FF"         # pale indigo for hovered buttons
WS_COLOR   = "#607d8b"         # muted slate for the Ws guide lines
PANEL_BG   = "#F4F3F1"         # warm-grey settings card
FIELD_BG   = "#FFFFFF"
BTN_BG     = "#ECEAE6"         # warm-grey buttons
BTN_HOVER  = SEL_FILL
BTN_ACTION = "#EAE7F6"         # action-button hover tint
EDGE_GREY  = "#37474f"         # element outline (softer than pure black)
TEXT_DARK  = "#263238"
TEXT_MUTE  = "#7a7a7a"
CB_LABEL   = "Contaminant concentration [mg/L]"   # designer-GUI colorbar label

EXPORT_DIR = Path(__file__).resolve().parent / "designer_exports"
EXPORT_DIR.mkdir(exist_ok=True)


# ── Transient Tk file dialogs ──────────────────────────────────────────────────
# A fresh hidden root is created per call and destroyed afterwards. This works
# under both the macosx and TkAgg backends. Returns "" on cancel, None on error.

def _ask_open_file(title, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return path
    except Exception:
        return None


def _ask_save_file(title, defaultextension, initialfile, filetypes):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        root.update()
        path = filedialog.asksaveasfilename(
            title=title, defaultextension=defaultextension,
            initialfile=initialfile, filetypes=filetypes)
        root.destroy()
        return path
    except Exception:
        return None


def _hide_cursor(tb):
    """Hide a TextBox's blinking caret until the user actually clicks it.

    matplotlib renders the caret as soon as a value is set, which leaves every
    field showing a stray '|'. We hide it after each programmatic update; the
    widget re-shows it on focus.
    """
    cur = getattr(tb, "cursor", None)
    if cur is not None:
        try:
            cur.set_visible(False)
        except Exception:
            pass


# ── Selection model ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Selection:
    """Tagged reference to a selected element.

    kind "composite_circle": packed circle in a composite (source_idx, sub_idx).
    kind "composite":        the whole composite (source_idx).
    kind "simple":           a SimpleSource (source_idx).
    """
    kind: str
    source_idx: int
    sub_idx: int | None = None

    def key(self):
        return (self.kind, self.source_idx, self.sub_idx)


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_elements_on_ax(ax, scene: Scene, selected_keys=frozenset(),
                        show_indices: bool = False) -> list:
    """Render all scene elements onto ax. Returns list of artist handles.

    selected_keys: set of (kind, source_idx, sub_idx) tuples to highlight.
    """
    artists = []

    for pi, comp in enumerate(scene.composites):
        verts = comp.vertices
        comp_selected = ("composite", pi, None) in selected_keys
        if len(verts) >= 3:
            pg = plt.Polygon(verts, closed=True, facecolor="gold", alpha=0.08,
                             edgecolor=(SEL_COLOR if comp_selected else "goldenrod"),
                             linewidth=(2.0 if comp_selected else 1.0), zorder=1)
            ax.add_patch(pg)
            artists.append(pg)
        for ci, c in enumerate(comp.circles):
            sel = comp_selected or ("composite_circle", pi, ci) in selected_keys
            color = DONOR_CMAP(DONOR_NORM(c["c"]))
            ec = SEL_COLOR if sel else EDGE_GREY
            lw = 2.2 if sel else 0.5
            patch = mpatches.Circle((c["x"], c["y"]), c["r"], facecolor=color,
                                    edgecolor=ec, linewidth=lw, zorder=3)
            ax.add_patch(patch)
            artists.append(patch)
            if show_indices and c.get("index") is not None:
                artists.append(_index_label(ax, c["x"], c["y"], c["index"]))

    for si, ss in enumerate(scene.simple_sources):
        sel = ("simple", si, None) in selected_keys
        color = DONOR_CMAP(DONOR_NORM(ss.c))
        ec = SEL_COLOR if sel else "black"
        lw = 2.5 if sel else 0.8

        if ss.kind == "circle":
            r = ss.r or DEFAULT_RADIUS
            patch = mpatches.Circle((ss.x, ss.y), r, facecolor=color,
                                    edgecolor=ec, linewidth=lw, zorder=3)
            ax.add_patch(patch)
            artists.append(patch)
        elif ss.kind == "ellipse":
            a = ss.a or DEFAULT_RADIUS
            b = ss.b or DEFAULT_RADIUS
            patch = mpatches.Ellipse((ss.x, ss.y), 2 * a, 2 * b, angle=ss.theta,
                                     facecolor=color, edgecolor=ec, linewidth=lw,
                                     zorder=3)
            ax.add_patch(patch)
            artists.append(patch)
        elif ss.kind == "line":
            l = ss.l or DEFAULT_RADIUS * 4
            t = math.radians(ss.theta)
            dx = l / 2 * math.cos(t)
            dy = l / 2 * math.sin(t)
            line, = ax.plot([ss.x - dx, ss.x + dx], [ss.y - dy, ss.y + dy], "-",
                            color=list(color)[:3], linewidth=5 if sel else 4,
                            solid_capstyle="round", zorder=3)
            artists.append(line)
            dot, = ax.plot(ss.x, ss.y, "o", color=ec, markersize=4, zorder=4)
            artists.append(dot)

        if show_indices and ss.index is not None:
            artists.append(_index_label(ax, ss.x, ss.y, ss.index))

    return artists


def _index_label(ax, x, y, idx):
    """Small faint index number drawn at an element centroid (clipped to axes)."""
    return ax.text(x, y, str(idx), fontsize=7, color=TEXT_DARK, ha="center",
                   va="center", zorder=7, fontweight="bold", clip_on=True,
                   bbox=dict(facecolor="white", alpha=0.55, edgecolor="none",
                             boxstyle="circle,pad=0.12"))


# ── In-figure settings panel ────────────────────────────────────────────────────

class SettingsPanel:
    """
    In-figure settings panel (matplotlib widgets) for editing GlobalParams.

    Two modes:
      docked=True  — a permanent sidebar (used in the Draw window), always
                     visible, no Close button.
      docked=False — a toggleable modal overlay (used in the View window).

    Each field applies live on Enter; the orientation button toggles instantly.
    Hidden widgets (overlay mode) are deactivated so they never intercept clicks.
    """

    # (attr, mathtext label, type, validator, tooltip description)
    MODEL_FIELDS = [
        ("alpha_l", r"Long. dispersivity $\alpha_L$", float, lambda v: v > 0,
         "Longitudinal dispersivity — plume spreading along the flow direction [m]."),
        ("alpha_t", r"Trans. dispersivity $\alpha_T$", float, lambda v: v > 0,
         "Transverse dispersivity — plume spreading across the flow direction [m]."),
        ("ca",      r"Acceptor conc. $c_A$",  float, lambda v: True,
         "Background electron-acceptor concentration [mg/L]."),
        ("gamma",   r"Stoich. ratio $\gamma$", float, lambda v: v > 0,
         "Donor : acceptor stoichiometric mass ratio (dimensionless)."),
        ("ws",      r"Source width $W_s$",    float, lambda v: v > 0,
         "Source-zone width guide [m] — the x-extent shown by the gold lines."),
    ]
    COMP_FIELDS = [
        ("dom_inc",     r"Grid spacing $\Delta$", float, lambda v: v > 0,
         "Output grid spacing [m]. Smaller = finer field but slower solve."),
        ("num_cp",      r"Control points",     int,   lambda v: v >= 1,
         "Boundary control points per element used to fit the solution."),
        ("num_terms",   r"Expansion terms",    int,   lambda v: v >= 1,
         "Number of Mathieu expansion terms (higher = more accurate, slower)."),
        ("plot_aspect", r"Plot aspect",        str,   lambda v: True,
         "Result aspect: 'scaled' for equal x/y axes, blank for auto."),
    ]
    FIELDS = MODEL_FIELDS + COMP_FIELDS

    def __init__(self, fig, scene, on_change, rect, docked=False):
        self.fig = fig
        self.scene = scene
        self.on_change = on_change
        self.rect = rect              # (x0, y0, w, h) in figure coords
        self.docked = docked
        self.visible = False
        self._updating = False
        self.textboxes: dict = {}
        self._axes: list = []
        self._texts: list = []
        self._build()
        if docked:
            self.show()
        else:
            self.hide()

    def _fc(self, rx, ry, rw, rh):
        x0, y0, w, h = self.rect
        return [x0 + rx * w, y0 + ry * h, rw * w, rh * h]

    def _build(self):
        x0, y0, w, h = self.rect
        self.bg = self.fig.add_axes(self.rect, zorder=50)
        self.bg.set_facecolor(PANEL_BG)
        self.bg.set_xticks([]); self.bg.set_yticks([])
        for s in self.bg.spines.values():
            s.set_edgecolor("#9099a8"); s.set_linewidth(1.3)
        self._axes.append(self.bg)

        self.title = self.fig.text(x0 + w / 2, y0 + 0.965 * h, "Settings",
                                   ha="center", va="top", fontsize=13,
                                   fontweight="bold", color="#263238", zorder=52)
        self.hint = self.fig.text(x0 + w / 2, y0 + 0.928 * h,
                                  "Type a value, press Enter to apply",
                                  ha="center", va="top", fontsize=8,
                                  color="#888", zorder=52)
        self._texts += [self.title, self.hint]

        lx = 0.06               # left margin for headers / separator
        box_x, box_w, box_h = 0.55, 0.30, 0.046
        icon_x = 0.93           # info icon centre (panel-relative)
        self._icons = []        # (fig_x, fig_y, description)

        def header(text, rel_y):
            t = self.fig.text(x0 + lx * w, y0 + rel_y * h, text, ha="left",
                              va="center", fontsize=10, fontweight="bold",
                              color=SEL_COLOR, zorder=52)
            self._texts.append(t)

        def info_icon(rel_y, desc):
            fx, fy = x0 + icon_x * w, y0 + (rel_y + box_h / 2) * h
            t = self.fig.text(fx, fy, "i", fontsize=7.5, ha="center", va="center",
                              color="white", fontweight="bold", zorder=53,
                              bbox=dict(boxstyle="circle,pad=0.22",
                                        facecolor=SEL_COLOR, edgecolor="none"))
            self._texts.append(t)
            self._icons.append((fx, fy, desc))

        def field_row(spec, rel_y):
            name, label, typ, _val, desc = spec
            ax = self.fig.add_axes(self._fc(box_x, rel_y, box_w, box_h), zorder=52)
            ax.set_facecolor(FIELD_BG)
            tb = TextBox(ax, label, initial="", textalignment="left")
            tb.label.set_fontsize(9.5)
            tb.label.set_color(TEXT_DARK)
            tb.on_submit(lambda text, n=name: self._submit(n, text))
            _hide_cursor(tb)
            self.textboxes[name] = tb
            self._axes.append(ax)
            info_icon(rel_y, desc)

        # ── Model parameters ──────────────────────────────────────────────
        header("Model parameters", 0.885)
        ry = 0.825
        for spec in self.MODEL_FIELDS:
            field_row(spec, ry)
            ry -= 0.066
        # Orientation — aligned to the same field column, with a left label.
        ry -= 0.004
        olabel = self.fig.text(x0 + (box_x - 0.02) * w, y0 + (ry + box_h / 2) * h,
                               "Orientation", ha="right", va="center",
                               fontsize=9.5, color=TEXT_DARK, zorder=52)
        self._texts.append(olabel)
        oax = self.fig.add_axes(self._fc(box_x, ry, box_w, box_h), zorder=52)
        self.btn_orient = Button(oax, "vertical", color=BTN_BG, hovercolor=BTN_HOVER)
        self.btn_orient.on_clicked(lambda _: self._toggle_orientation())
        self._axes.append(oax)
        ry -= 0.072

        # Separator
        sep = Line2D([x0 + lx * w, x0 + (1 - lx) * w], [y0 + ry * h] * 2,
                     color="#d7dade", lw=1.0, zorder=52,
                     transform=self.fig.transFigure)
        self.fig.add_artist(sep)
        self._extras = [sep]
        ry -= 0.035

        # ── Computational parameters ──────────────────────────────────────
        header("Computational parameters", ry)
        ry -= 0.058
        for spec in self.COMP_FIELDS:
            field_row(spec, ry)
            ry -= 0.066

        self.msg = self.fig.text(x0 + w / 2, y0 + 0.055 * h, "", ha="center",
                                 va="center", fontsize=8.5, color="#b00", zorder=52)
        self._texts.append(self.msg)

        # Hover tooltip (one reusable artist)
        self._tip = self.fig.text(0, 0, "", fontsize=8.5, ha="right", va="center",
                                  color=TEXT_DARK, zorder=60, visible=False,
                                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFDE7",
                                            edgecolor=SEL_COLOR, linewidth=0.8))
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

        self._widgets = [*self.textboxes.values(), self.btn_orient]
        if not self.docked:
            cax = self.fig.add_axes(self._fc(0.34, 0.008, 0.32, 0.05), zorder=52)
            self.btn_close = Button(cax, "Close ✕",
                                    color="#f3d6d6", hovercolor="#eebcbc")
            self.btn_close.on_clicked(lambda _: self.hide())
            self._axes.append(cax)
            self._widgets.append(self.btn_close)

    def _on_hover(self, ev):
        if not self.visible or ev.x is None:
            return
        fw = self.fig.get_size_inches()[0] * self.fig.dpi
        fh = self.fig.get_size_inches()[1] * self.fig.dpi
        fx, fy = ev.x / fw, ev.y / fh
        for ix, iy, desc in self._icons:
            if abs(fx - ix) < 0.010 and abs(fy - iy) < 0.012:
                import textwrap
                self._tip.set_text("\n".join(textwrap.wrap(desc, 34)))
                self._tip.set_position((ix - 0.02, iy))
                if not self._tip.get_visible():
                    self._tip.set_visible(True)
                    self.fig.canvas.draw_idle()
                return
        if self._tip.get_visible():
            self._tip.set_visible(False)
            self.fig.canvas.draw_idle()

    def _set_tb(self, name, value):
        tb = self.textboxes[name]
        self._updating = True
        try:
            tb.set_val(f"{value:g}" if isinstance(value, float) else str(value))
        finally:
            self._updating = False
        _hide_cursor(tb)

    def refresh_values(self):
        g = self.scene.globals
        for name, *_ in self.FIELDS:
            self._set_tb(name, getattr(g, name))
        self.btn_orient.label.set_text(g.orientation)

    def show(self):
        self.visible = True
        self.refresh_values()
        self.msg.set_text("")
        for ax in self._axes:
            ax.set_visible(True)
        for art in self._texts + self._extras:
            art.set_visible(True)
        for wdg in self._widgets:
            wdg.active = True
        self.fig.canvas.draw_idle()

    def hide(self):
        if self.docked:
            return
        self.visible = False
        for ax in self._axes:
            ax.set_visible(False)
        for art in self._texts + self._extras:
            art.set_visible(False)
        self._tip.set_visible(False)
        for wdg in self._widgets:
            wdg.active = False
        self.fig.canvas.draw_idle()

    def toggle(self):
        if self.docked:
            return
        self.hide() if self.visible else self.show()

    def _submit(self, name, text):
        if self._updating:
            return
        spec = next(f for f in self.FIELDS if f[0] == name)
        _, label, typ, validator, _desc = spec
        try:
            value = typ(text)
        except (ValueError, TypeError):
            self.msg.set_text(f"{name}: must be {typ.__name__}")
            self.msg.set_color("#b00")
            self.fig.canvas.draw_idle()
            return
        if not validator(value):
            self.msg.set_text(f"{name}: invalid value")
            self.msg.set_color("#b00")
            self.fig.canvas.draw_idle()
            return
        setattr(self.scene.globals, name, value)
        self.msg.set_text(f"{name} = {value}  ✓")
        self.msg.set_color("#2e7d32")
        self.on_change()
        self.fig.canvas.draw_idle()

    def _toggle_orientation(self):
        g = self.scene.globals
        g.orientation = "horizontal" if g.orientation == "vertical" else "vertical"
        self.btn_orient.label.set_text(g.orientation)
        self.msg.set_text(f"orientation = {g.orientation}  ✓")
        self.msg.set_color("#2e7d32")
        self.on_change()
        self.fig.canvas.draw_idle()


# ── DRAW / EDIT window ─────────────────────────────────────────────────────────

class DrawWindow:

    TOOLS = ["polygon", "circle", "ellipse", "line", "select"]
    TOOL_LABELS = ["Polygon (1)", "Circle (2)", "Ellipse (3)", "Line (4)", "Select (5)"]

    def __init__(self, designer: "SourceDesigner"):
        self.d = designer
        self.active_tool = "polygon"
        self._title = "Source zone — Draw (Polygon)"
        self.dragging = False
        self.drag_offset = (0.0, 0.0)
        self._pan = False
        self._pan0 = None
        self._pan_xl = None
        self._pan_yl = None
        self._pan_moved = False
        self._dyn: list = []
        self._preview: list = []

        self._press_pos = None
        self._mouse_down = False
        self._line_start = None

        # Multi-select / rubber-band state
        self.show_indices = True
        self._band = None              # rubber-band rectangle artist
        self._band_start = None        # (x, y) where the band drag began
        self._band_add = False         # Shift held → add to existing selection
        self._moved_during_press = False

        self._tb_updating = False
        self._textboxes: dict = {}
        self._tb_param: dict = {}

        self.fig = plt.figure("Source Designer — Draw", figsize=(14, 9.0), dpi=100)
        self.fig.patch.set_facecolor("white")

        # Wide plot on the left; a narrow settings sidebar spans the full height
        # on the right; the buttons sit directly under the plot.
        self.ax = self.fig.add_axes([0.05, 0.47, 0.60, 0.50])
        self.ax.set_facecolor("white")
        self.ax.set_aspect("equal")

        self._update_axis_labels()
        self.ax.set_title(self._title, fontsize=13, pad=6)
        self.ax.tick_params(labelsize=10)
        self._reframe_for_orientation()

        self._draw_static()
        self._setup_bottom()

        # Settings docked as a right-hand sidebar (equal top/bottom margins).
        self.settings = SettingsPanel(
            self.fig, self.d.scene, self.d.on_globals_changed,
            rect=(0.695, 0.085, 0.29, 0.83), docked=True)

        self._connect()
        self.redraw()

    # ── Static decorations ──────────────────────────────────────────────────

    def _draw_static(self):
        ax = self.ax
        g = self.d.scene.globals
        for x in np.arange(0, g.ws + GRID_SPACING, GRID_SPACING):
            ax.axvline(x, color="#eeeeee", linewidth=0.3, zorder=0)
        for y in np.arange(-GRID_HEIGHT, GRID_HEIGHT + GRID_SPACING, GRID_SPACING):
            ax.axhline(y, color="#eeeeee", linewidth=0.3, zorder=0)
        # Ws guides: muted slate (was hard-to-read yellow).
        ax.axvline(0.0, color=WS_COLOR, linewidth=1.3, linestyle=(0, (6, 4)), zorder=4)
        ax.axvline(g.ws, color=WS_COLOR, linewidth=1.3, linestyle=(0, (6, 4)), zorder=4)
        ax.text(0.99, 0.99, f"$W_s$ = {g.ws:g} m", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=WS_COLOR,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor=WS_COLOR,
                          boxstyle="round,pad=0.2"))

        if g.orientation == "vertical":
            # Water table at y=0 — strong solid blue, distinct from the Ws guides.
            ax.axhline(0.0, color=SEL_COLOR, linewidth=2.2, zorder=5)
            ax.text(0.01, 0.99, "water table (y = 0) — draw below",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                    color=SEL_COLOR,
                    bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.2"))

    def _reframe_for_orientation(self):
        """
        Set the y-range to match the orientation convention:
          vertical   → aquifer below the water table, y ∈ [-(Ws+m), +small]
          horizontal → transverse band, y ∈ [-small, Ws+m]
        x always spans the source-zone width [0, Ws].
        """
        g = self.d.scene.globals
        m = max(g.ws * 0.10, 0.05)
        self.ax.set_xlim(-m, g.ws + m)
        if g.orientation == "vertical":
            self.ax.set_ylim(-(g.ws + m), m)
        else:
            self.ax.set_ylim(-m, g.ws + m)

    def _clamp_to_region(self, wx, wy):
        """Clamp a clicked point into the valid drawing region for the
        current orientation (x ≥ 0; vertical → y ≤ 0, horizontal → y ≥ 0)."""
        g = self.d.scene.globals
        wx = max(0.0, wx)
        if g.orientation == "vertical":
            wy = min(0.0, wy)
        else:
            wy = max(0.0, wy)
        return wx, wy

    def _update_axis_labels(self):
        g = self.d.scene.globals
        ylabel = "$z$ (m)" if g.orientation == "vertical" else "$y$ (m)"
        self.ax.set_xlabel("$x$ (m)", fontsize=12)
        self.ax.set_ylabel(ylabel, fontsize=12)

    # ── Bottom panel ────────────────────────────────────────────────────────

    def _setup_bottom(self):
        # Every widget kept on self — a collected Button stops responding.
        self._buttons: list[Button] = []

        # Vertical colorbar immediately right of the plot.
        sm = plt.cm.ScalarMappable(cmap="Reds", norm=mcolors.Normalize(0, CONC_MAX))
        sm.set_array([])
        cax = self.fig.add_axes([0.665, 0.47, 0.012, 0.50])
        cb = self.fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label(CB_LABEL, fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Short selection-summary label (values live in the fields, not here).
        self.info_text = self.fig.text(0.05, 0.420, "", fontsize=10,
                                       color=SEL_COLOR, va="top")

        # Editable parameter fields (under the plot, left of the sidebar)
        self._setup_param_textboxes()

        # Status line
        self.status_text = self.fig.text(0.05, 0.300, "", fontsize=9.5,
                                         color=TEXT_MUTE, va="top")

        # Controls under the plot (smaller buttons; sidebar is on the right)
        bh, gap = 0.044, 0.010
        left, right = 0.05, 0.655
        span = right - left
        tw = (span - 4 * gap) / 5
        ty = 0.205
        for i, (tool, label) in enumerate(zip(self.TOOLS, self.TOOL_LABELS)):
            bax = self.fig.add_axes([left + i * (tw + gap), ty, tw, bh])
            btn = Button(bax, label, color=BTN_BG, hovercolor=BTN_HOVER)
            btn.on_clicked(lambda _, t=tool: self._set_active_tool(t))
            self._buttons.append(btn)

        # Action row — 5 buttons incl. the index-label toggle.
        ay = 0.135
        aw = (span - 4 * gap) / 5
        self._idx_btn_ax = self.fig.add_axes([left + 4 * (aw + gap), ay, aw, bh])
        action_defs = [
            ("View (v)",      lambda _: self.d.show_view()),
            ("Export (s)",    lambda _: self.d.export_json()),
            ("Load JSON (o)", lambda _: self.d.import_json()),
            ("Clear all",     lambda _: self.d.clear_all()),
        ]
        for i, (label, cb) in enumerate(action_defs):
            bax = self.fig.add_axes([left + i * (aw + gap), ay, aw, bh])
            btn = Button(bax, label, color=BTN_BG, hovercolor=BTN_ACTION)
            btn.on_clicked(cb)
            self._buttons.append(btn)
        self.btn_idx = Button(self._idx_btn_ax, "Index # (i)",
                              color=SEL_FILL, hovercolor=BTN_ACTION)
        self.btn_idx.on_clicked(lambda _: self._toggle_indices())
        self._buttons.append(self.btn_idx)

        self._status("DRAW — pick a tool and click on the canvas to place sources")

    def _toggle_indices(self):
        self.show_indices = not self.show_indices
        self.btn_idx.color = SEL_FILL if self.show_indices else BTN_BG
        self.btn_idx.ax.set_facecolor(self.btn_idx.color)
        self._status(f"Index labels {'on' if self.show_indices else 'off'}")
        self.redraw()

    def _setup_param_textboxes(self):
        """Six editable fields (relabelled per element kind), hidden until select."""
        # Spread across the left region (under the plot).
        slots = [("x", 0.075), ("y", 0.175), ("c", 0.275),
                 ("p1", 0.375), ("p2", 0.475), ("p3", 0.575)]
        y, w, h = 0.360, 0.058, 0.034
        for name, x in slots:
            ax = self.fig.add_axes([x, y, w, h])
            tb = TextBox(ax, "", initial="", textalignment="left")
            tb.label.set_fontsize(11)
            tb.on_submit(lambda text, n=name: self._on_tb_submit(n, text))
            _hide_cursor(tb)
            self._textboxes[name] = tb
            ax.set_visible(False)
            tb.active = False
        self._populate_textboxes()

    # ── Status / info ───────────────────────────────────────────────────────

    def _status(self, msg: str):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _info(self, msg: str):
        self.info_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _set_active_tool(self, tool: str):
        self.active_tool = tool
        self.d.selection = []
        self._line_start = None
        self._press_pos = None
        self._mouse_down = False
        self._clear_band()
        self._clear_preview()
        hints = {
            "polygon": ("Source zone — Draw (Polygon)",
                        "POLYGON — left-click vertices, right-click to close & pack"),
            "circle":  ("Source zone — Draw (Circle)",
                        "CIRCLE — click-drag to place (drag distance sets radius)"),
            "ellipse": ("Source zone — Draw (Ellipse)",
                        "ELLIPSE — click-drag a bounding box (rotate later in Select)"),
            "line":    ("Source zone — Draw (Line)",
                        "LINE — click the start point, then click the end point"),
            "select":  ("Source zone — Select/Edit",
                        "SELECT — click an element to edit it"),
        }
        self._title, hint = hints[tool]
        self.ax.set_title(self._title, fontsize=13, pad=6)
        self._info("")
        if self._textboxes:
            self._populate_textboxes()
        self._status(hint)
        self.redraw()

    def _set_interaction(self, mode: str):           # legacy
        self._set_active_tool("polygon" if mode == "draw" else "select")

    # ── Event connection ────────────────────────────────────────────────────

    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("button_press_event",   self._press)
        c.mpl_connect("button_release_event", self._release)
        c.mpl_connect("motion_notify_event",  self._motion)
        c.mpl_connect("scroll_event",         self._scroll)
        c.mpl_connect("key_press_event",      self._key)

    # ── Mouse events ────────────────────────────────────────────────────────

    @staticmethod
    def _is_shift(ev):
        return bool(ev.key) and "shift" in ev.key

    @staticmethod
    def _is_cmd(ev):
        return bool(ev.key) and any(m in ev.key for m in ("cmd", "ctrl", "control", "super"))

    def _press(self, ev):
        if self.settings.visible and not self.settings.docked:   # modal popup only
            return
        if ev.inaxes != self.ax:
            return
        # Pan with middle OR right button drag (works on trackpads via right-click
        # = two-finger / control-click). Right-click without dragging still
        # closes a polygon (handled on release).
        if ev.button in (2, 3):
            self._pan = True
            self._pan_moved = False
            self._pan0 = (ev.x, ev.y)
            self._pan_xl = self.ax.get_xlim()
            self._pan_yl = self.ax.get_ylim()
            return

        wx, wy = ev.xdata, ev.ydata
        tool = self.active_tool

        if tool == "polygon":
            if ev.button == 1:
                if SNAP_TO_GRID:
                    wx = round(wx / GRID_SPACING) * GRID_SPACING
                    wy = round(wy / GRID_SPACING) * GRID_SPACING
                wx, wy = self._clamp_to_region(wx, wy)
                self.d.current_verts.append([wx, wy])
                n = len(self.d.current_verts)
                self._status(f"Vertex {n} at ({wx:.3f}, {wy:.3f}) — right-click to close")
                self.redraw()
        elif tool in ("circle", "ellipse"):
            if ev.button == 1:
                self._press_pos = self._clamp_to_region(wx, wy)
                self._mouse_down = True
        elif tool == "line":
            if ev.button == 1:
                if self._line_start is None:
                    self._line_start = self._clamp_to_region(wx, wy)
                    sx, sy = self._line_start
                    self._status(f"Line start ({sx:.3f}, {sy:.3f}) — click the end point")
                    self.redraw()
                else:
                    ex, ey = self._clamp_to_region(wx, wy)
                    self._commit_line(ex, ey)
        elif tool == "select":
            if ev.button == 1:
                hit = self._hit_test(wx, wy)
                if hit is None:
                    # Empty canvas → begin a rubber-band selection.
                    self._band_start = (wx, wy)
                    self._band_add = self._is_shift(ev)
                    if not self._band_add:
                        self.d.selection = []
                        self._sync_selection_ui()
                        self.redraw()
                    return
                if getattr(ev, "dblclick", False) and hit.kind == "composite_circle":
                    # Double-click a packed circle → select the whole polygon.
                    self._select_only(Selection("composite", hit.source_idx))
                    self._begin_drag(wx, wy)
                elif self._is_shift(ev) or self._is_cmd(ev):
                    self._toggle_selection(hit)
                else:
                    if hit.key() not in self._sel_keys():
                        self._select_only(hit)
                    self._begin_drag(wx, wy)

    def _release(self, ev):
        # A right-click that did NOT pan closes the polygon (button 3, no drag).
        was_pan, moved = self._pan, self._pan_moved
        self._pan = False
        self._pan_moved = False
        if was_pan:
            if (ev.button == 3 and not moved and self.active_tool == "polygon"):
                self._close_polygon()
            return

        # Finalise a rubber-band selection.
        if self._band_start is not None:
            x0, y0 = self._band_start
            wx = ev.xdata if (ev.inaxes == self.ax and ev.xdata is not None) else x0
            wy = ev.ydata if (ev.inaxes == self.ax and ev.ydata is not None) else y0
            self._clear_band()
            if abs(wx - x0) > 1e-6 or abs(wy - y0) > 1e-6:
                self._select_in_rect(x0, y0, wx, wy, add=self._band_add)
            self._band_add = False
            self.dragging = False
            self.redraw()
            return

        self.dragging = False

        if self._mouse_down:
            self._mouse_down = False
            tool = self.active_tool
            wx = ev.xdata if (ev.inaxes == self.ax and ev.xdata is not None) else None
            wy = ev.ydata if (ev.inaxes == self.ax and ev.ydata is not None) else None

            if tool == "circle" and self._press_pos is not None:
                px, py = self._press_pos
                r = (max(math.hypot(wx - px, wy - py), DEFAULT_RADIUS)
                     if wx is not None and wy is not None else DEFAULT_RADIUS)
                ss = SimpleSource(kind="circle", x=round(px, 5), y=round(py, 5),
                                  c=DEFAULT_CONC, r=round(r, 5),
                                  id=f"circle_{uuid.uuid4().hex[:6]}")
                self.d.scene.simple_sources.append(ss)
                self.d.scene.ensure_display_indices()
                self._clear_preview()
                self._status(f"Circle placed: r = {r:.4f} m  (switch to Select to edit)")
                self.redraw()
            elif tool == "ellipse" and self._press_pos is not None:
                px, py = self._press_pos
                if wx is not None and wy is not None and (abs(wx-px) > 1e-6 or abs(wy-py) > 1e-6):
                    a = max(abs(wx - px) / 2.0, DEFAULT_RADIUS)
                    b = max(abs(wy - py) / 2.0, DEFAULT_RADIUS)
                    cx, cy = (px + wx) / 2.0, (py + wy) / 2.0
                else:
                    a = b = DEFAULT_RADIUS
                    cx, cy = px, py
                ss = SimpleSource(kind="ellipse", x=round(cx, 5), y=round(cy, 5),
                                  c=DEFAULT_CONC, a=round(a, 5), b=round(b, 5),
                                  theta=0.0, id=f"ellipse_{uuid.uuid4().hex[:6]}")
                self.d.scene.simple_sources.append(ss)
                self.d.scene.ensure_display_indices()
                self._clear_preview()
                self._status(f"Ellipse placed: a={a:.4f} b={b:.4f}  (Select to rotate/edit)")
                self.redraw()
            self._press_pos = None
        else:
            self.dragging = False

    def _motion(self, ev):
        if self._pan and self._pan0 and ev.x is not None:
            dx = ev.x - self._pan0[0]
            dy = ev.y - self._pan0[1]
            if abs(dx) > 2 or abs(dy) > 2:
                self._pan_moved = True
            bbox = self.ax.get_position()
            fw = self.fig.get_size_inches()[0] * self.fig.dpi
            fh = self.fig.get_size_inches()[1] * self.fig.dpi
            xr, yr = self._pan_xl, self._pan_yl
            self.ax.set_xlim(xr[0] - (xr[1]-xr[0])*dx/(bbox.width*fw),
                             xr[1] - (xr[1]-xr[0])*dx/(bbox.width*fw))
            self.ax.set_ylim(yr[0] - (yr[1]-yr[0])*dy/(bbox.height*fh),
                             yr[1] - (yr[1]-yr[0])*dy/(bbox.height*fh))
            self.fig.canvas.draw_idle()
            return

        if ev.inaxes != self.ax or ev.xdata is None:
            return
        wx, wy = ev.xdata, ev.ydata

        if self._mouse_down and self._press_pos is not None:
            px, py = self._press_pos
            if self.active_tool == "circle":
                r = max(math.hypot(wx - px, wy - py), DEFAULT_RADIUS)
                self._show_preview_circle(px, py, r)
            elif self.active_tool == "ellipse":
                a = max(abs(wx - px) / 2.0, DEFAULT_RADIUS)
                b = max(abs(wy - py) / 2.0, DEFAULT_RADIUS)
                self._show_preview_ellipse((px + wx) / 2.0, (py + wy) / 2.0, a, b)
            return

        # Live rubber-band rectangle
        if self._band_start is not None:
            self._moved_during_press = True
            x0, y0 = self._band_start
            if self._band is not None:
                try:
                    self._band.remove()
                except Exception:
                    pass
            self._band = mpatches.Rectangle(
                (min(x0, wx), min(y0, wy)), abs(wx - x0), abs(wy - y0),
                facecolor=SEL_COLOR, alpha=0.12, edgecolor=SEL_COLOR,
                linewidth=1.0, linestyle="--", zorder=8)
            self.ax.add_patch(self._band)
            self.fig.canvas.draw_idle()
            return

        # Drag the whole selection together
        if self.dragging and self.d.selection:
            self._apply_drag(wx, wy)
            self._update_info()
            self._populate_textboxes()
            self.redraw()

    def _scroll(self, ev):
        if ev.inaxes != self.ax:
            return
        factor = 0.85 if ev.button == "up" else 1.18
        xc, yc = ev.xdata, ev.ydata
        xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim([xc + (v - xc) * factor for v in xl])
        self.ax.set_ylim([yc + (v - yc) * factor for v in yl])
        self.fig.canvas.draw_idle()

    def _key(self, ev):
        k = ev.key
        if self._textbox_capturing():
            return
        if k == "1":
            self._set_active_tool("polygon")
        elif k == "2":
            self._set_active_tool("circle")
        elif k == "3":
            self._set_active_tool("ellipse")
        elif k in ("5", "e"):
            self._set_active_tool("select")
        elif k == "4":
            self._set_active_tool("line")
        elif k == "d":
            self._set_active_tool("polygon")
        elif k == "v":
            self.d.show_view()
        elif k == "s":
            self.d.export_json()
        elif k == "o":
            self.d.import_json()
        elif k == "i":
            self._toggle_indices()
        elif k in ("ctrl+a", "cmd+a"):
            self.select_all()
        elif k == "escape":
            if self.settings.visible and not self.settings.docked:
                self.settings.hide()
            elif self.d.selection:
                self.d.selection = []
                self._clear_band()
                self._sync_selection_ui()
                self.redraw()
            else:
                self.d.current_verts = []
                self._line_start = None
                self._clear_preview()
                self._status("Cancelled")
                self.redraw()
        elif k == "q":
            plt.close("all")
        elif k == "n" and self.active_tool == "polygon":
            self.d.current_verts = []
            self._status("New polygon — click to place vertices")
            self.redraw()
        elif k in ("delete", "backspace"):
            self._delete_selected()
        elif k == "x":
            self._delete_polygon_or_composite()
        elif k == "up":
            self._adjust_size(+RADIUS_STEP)
        elif k == "down":
            self._adjust_size(-RADIUS_STEP)
        elif k in ("shift+up", "9"):
            self._adjust_b(+RADIUS_STEP)
        elif k in ("shift+down", "8"):
            self._adjust_b(-RADIUS_STEP)
        elif k in ("]", ".", ">"):
            self._adjust_angle(+5.0)
        elif k in ("[", ",", "<"):
            self._adjust_angle(-5.0)
        elif k in ("+", "="):
            self._adjust_conc(+CONC_STEP)
        elif k in ("-", "_"):
            self._adjust_conc(-CONC_STEP)

    # ── Polygon / line tools ──────────────────────────────────────────────────

    def _close_polygon(self):
        if len(self.d.current_verts) < 3:
            self._status("Need ≥ 3 vertices")
            return
        verts = list(self.d.current_verts)
        self._status("Packing circles (greedy)…")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        circles = greedy_circle_pack(verts, default_c=DEFAULT_CONC)
        comp = CompositeSource(vertices=verts, circles=circles, base_c=DEFAULT_CONC)
        self.d.scene.composites.append(comp)
        self.d.scene.ensure_display_indices()
        self.d.current_verts = []
        self._status(f"Polygon {len(self.d.scene.composites)} — {len(circles)} circles packed")
        self._set_active_tool("select")

    def _commit_line(self, x2, y2):
        x1, y1 = self._line_start
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        l = math.hypot(x2 - x1, y2 - y1)
        if l < 1e-6:
            self._status("Line too short — click further away")
            return
        theta = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ss = SimpleSource(kind="line", x=round(cx, 5), y=round(cy, 5),
                          c=DEFAULT_CONC, l=round(l, 5), theta=round(theta, 2),
                          id=f"line_{uuid.uuid4().hex[:6]}")
        self.d.scene.simple_sources.append(ss)
        self.d.scene.ensure_display_indices()
        self._line_start = None
        self._status(f"Line placed: l = {l:.4f} m, θ = {theta:.1f}°")
        self.redraw()

    # ── Preview patches ─────────────────────────────────────────────────────

    def _clear_preview(self):
        for a in self._preview:
            try:
                a.remove()
            except Exception:
                pass
        self._preview = []
        self.fig.canvas.draw_idle()

    def _show_preview_circle(self, cx, cy, r):
        self._clear_preview()
        p = mpatches.Circle((cx, cy), r, facecolor="steelblue", alpha=0.3,
                            edgecolor="steelblue", linewidth=1.5, zorder=5)
        self.ax.add_patch(p)
        self._preview.append(p)
        self.fig.canvas.draw_idle()

    def _show_preview_ellipse(self, cx, cy, a, b):
        self._clear_preview()
        p = mpatches.Ellipse((cx, cy), 2*a, 2*b, angle=0.0, facecolor="steelblue",
                            alpha=0.3, edgecolor="steelblue", linewidth=1.5, zorder=5)
        self.ax.add_patch(p)
        self._preview.append(p)
        self.fig.canvas.draw_idle()

    # ── Selection ───────────────────────────────────────────────────────────

    # ── Selection set management ──────────────────────────────────────────────

    def _hit_test(self, wx, wy):
        """Return the closest element Selection under the cursor, or None."""
        best_dist = float("inf")
        best_sel = None
        for pi, comp in enumerate(self.d.scene.composites):
            for ci, c in enumerate(comp.circles):
                dist = math.hypot(wx - c["x"], wy - c["y"])
                if dist <= c["r"] * 1.5 and dist < best_dist:
                    best_dist, best_sel = dist, Selection("composite_circle", pi, ci)
        for si, ss in enumerate(self.d.scene.simple_sources):
            if ss.contains_point(wx, wy):
                dist = math.hypot(wx - ss.x, wy - ss.y)
                if dist < best_dist:
                    best_dist, best_sel = dist, Selection("simple", si)
        return best_sel

    def _sel_keys(self):
        return {s.key() for s in self.d.selection}

    def _select_only(self, sel):
        self.d.selection = [sel]
        self._sync_selection_ui()

    def _toggle_selection(self, sel):
        keys = self._sel_keys()
        if sel.key() in keys:
            self.d.selection = [s for s in self.d.selection if s.key() != sel.key()]
        else:
            self.d.selection = self.d.selection + [sel]
        self._sync_selection_ui()

    def _sync_selection_ui(self):
        # Redraw the canvas immediately so the highlight tracks the selection
        # (previously the highlight only refreshed on the next drag/motion).
        self._update_info()
        self._populate_textboxes()
        if self.d.selection:
            self._status("Edit fields · ↑↓ size · +/- conc · [ ] rotate · drag · Del · "
                         "Shift-click multi · dbl-click polygon · Ctrl+A all")
        else:
            self._status("SELECT — click an element; drag on empty space to rubber-band")
        self.redraw()

    # Resolve selections to live objects (call before mutating; act by identity)

    def _resolve(self, sel):
        """Return ('cc', comp, circle) | ('comp', comp) | ('ss', ss) | None."""
        try:
            if sel.kind == "composite_circle":
                comp = self.d.scene.composites[sel.source_idx]
                return ("cc", comp, comp.circles[sel.sub_idx])
            if sel.kind == "composite":
                return ("comp", self.d.scene.composites[sel.source_idx])
            if sel.kind == "simple":
                return ("ss", self.d.scene.simple_sources[sel.source_idx])
        except (IndexError, KeyError):
            return None
        return None

    def _selected_resolved(self):
        out = []
        for s in self.d.selection:
            r = self._resolve(s)
            if r is not None:
                out.append(r)
        return out

    # ── Drag of the whole selection ──────────────────────────────────────────

    def _begin_drag(self, wx, wy):
        """Snapshot start positions of every selected element for translation."""
        self.dragging = True
        self._drag_origin = (wx, wy)
        self._drag_snap = []   # (mover_fn,) closures capturing start positions
        for r in self._selected_resolved():
            if r[0] == "cc":
                c = r[2]
                self._drag_snap.append(("pt", c, c["x"], c["y"]))
            elif r[0] == "ss":
                ss = r[1]
                self._drag_snap.append(("ss", ss, ss.x, ss.y))
            elif r[0] == "comp":
                comp = r[1]
                starts = [(c, c["x"], c["y"]) for c in comp.circles]
                vstarts = [(v, v[0], v[1]) for v in comp.vertices]
                self._drag_snap.append(("comp", comp, starts, vstarts))

    def _apply_drag(self, wx, wy):
        dx = wx - self._drag_origin[0]
        dy = wy - self._drag_origin[1]
        for snap in self._drag_snap:
            if snap[0] == "pt":
                _, c, x0, y0 = snap
                c["x"] = round(x0 + dx, 5); c["y"] = round(y0 + dy, 5)
            elif snap[0] == "ss":
                _, ss, x0, y0 = snap
                ss.x = round(x0 + dx, 5); ss.y = round(y0 + dy, 5)
            elif snap[0] == "comp":
                _, comp, starts, vstarts = snap
                for c, x0, y0 in starts:
                    c["x"] = round(x0 + dx, 5); c["y"] = round(y0 + dy, 5)
                for v, x0, y0 in vstarts:
                    v[0] = round(x0 + dx, 5); v[1] = round(y0 + dy, 5)

    def _clear_band(self):
        if self._band is not None:
            try:
                self._band.remove()
            except Exception:
                pass
        self._band = None
        self._band_start = None

    def _select_in_rect(self, x0, y0, x1, y1, add=False):
        """Select every element whose centre lies inside the rectangle."""
        xlo, xhi = sorted((x0, x1))
        ylo, yhi = sorted((y0, y1))
        found = []
        for pi, comp in enumerate(self.d.scene.composites):
            for ci, c in enumerate(comp.circles):
                if xlo <= c["x"] <= xhi and ylo <= c["y"] <= yhi:
                    found.append(Selection("composite_circle", pi, ci))
        for si, ss in enumerate(self.d.scene.simple_sources):
            if xlo <= ss.x <= xhi and ylo <= ss.y <= yhi:
                found.append(Selection("simple", si))
        if add:
            have = self._sel_keys()
            self.d.selection = self.d.selection + [s for s in found if s.key() not in have]
        else:
            self.d.selection = found
        self._sync_selection_ui()

    # ── Editable parameter fields ─────────────────────────────────────────────

    def _tb_set(self, name, value, label):
        tb = self._textboxes[name]
        tb.ax.set_visible(True)
        tb.active = True
        tb.label.set_text(label)
        self._tb_updating = True
        try:
            # Display rounded to 2 dp; the model keeps full precision.
            tb.set_val(f"{value:.2f}" if isinstance(value, float) else str(value))
        finally:
            self._tb_updating = False
        _hide_cursor(tb)

    def _tb_set_text(self, name, text, label):
        """Set a literal string (used for the '(mixed)' placeholder)."""
        tb = self._textboxes[name]
        tb.ax.set_visible(True)
        tb.active = True
        tb.label.set_text(label)
        self._tb_updating = True
        try:
            tb.set_val(text)
        finally:
            self._tb_updating = False
        _hide_cursor(tb)

    def _tb_hide(self, *names):
        for n in names:
            self._textboxes[n].ax.set_visible(False)
            self._textboxes[n].active = False

    MIXED = "—"

    def _tb_value_or_mixed(self, name, values, label):
        vals = [round(v, 6) for v in values]
        if all(v == vals[0] for v in vals):
            self._tb_set(name, float(vals[0]), label)
        else:
            self._tb_set_text(name, self.MIXED, label)

    def _leaf_values(self):
        """(xs, ys, cs, kinds) over all selected leaf elements (composite
        expands to its circles)."""
        xs, ys, cs, kinds = [], [], [], set()
        for r in self._selected_resolved():
            if r[0] == "cc":
                c = r[2]; xs.append(c["x"]); ys.append(c["y"]); cs.append(c["c"]); kinds.add("circle")
            elif r[0] == "ss":
                ss = r[1]; xs.append(ss.x); ys.append(ss.y); cs.append(ss.c); kinds.add(ss.kind)
            elif r[0] == "comp":
                for c in r[1].circles:
                    xs.append(c["x"]); ys.append(c["y"]); cs.append(c["c"])
                kinds.add("composite")
        return xs, ys, cs, kinds

    def _centroid(self):
        xs, ys, _, _ = self._leaf_values()
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    def _populate_textboxes(self):
        self._tb_param = {}
        res = self._selected_resolved()
        if not res:
            self._tb_hide("x", "y", "c", "p1", "p2", "p3")
            self.fig.canvas.draw_idle()
            return

        # Single simple/packed element → full per-kind fields (exact values).
        if len(res) == 1 and res[0][0] in ("cc", "ss"):
            self._populate_single(res[0])
            self.fig.canvas.draw_idle()
            return

        # Multi-selection (or a composite unit): x,y = centroid, c set/mixed.
        xs, ys, cs, _ = self._leaf_values()
        cx, cy = (sum(xs) / len(xs), sum(ys) / len(ys))
        self._tb_set("x", cx, "$x$ ")
        self._tb_set("y", cy, "$y$ ")
        self._tb_value_or_mixed("c", cs, "$c$ ")
        self._tb_hide("p1", "p2", "p3")
        self._tb_param = {"x": "x", "y": "y", "c": "c"}
        self.fig.canvas.draw_idle()

    def _populate_single(self, r):
        if r[0] == "cc":
            c = r[2]
            self._tb_set("x", c["x"], "$x$ ")
            self._tb_set("y", c["y"], "$y$ ")
            self._tb_set("c", c["c"], "$c$ ")
            self._tb_set("p1", c["r"], "$r$ ")
            self._tb_hide("p2", "p3")
            self._tb_param = {"x": "x", "y": "y", "c": "c", "p1": "r"}
            return
        ss = r[1]
        if True:
            self._tb_set("x", ss.x, "$x$ ")
            self._tb_set("y", ss.y, "$y$ ")
            self._tb_set("c", ss.c, "$c$ ")
            if ss.kind == "circle":
                self._tb_set("p1", ss.r or DEFAULT_RADIUS, "$r$ ")
                self._tb_hide("p2", "p3")
                self._tb_param = {"x": "x", "y": "y", "c": "c", "p1": "r"}
            elif ss.kind == "ellipse":
                self._tb_set("p1", ss.a or DEFAULT_RADIUS, "$a$ ")
                self._tb_set("p2", ss.b or DEFAULT_RADIUS, "$b$ ")
                self._tb_set("p3", ss.theta, r"$\theta$ ")
                self._tb_param = {"x": "x", "y": "y", "c": "c",
                                  "p1": "a", "p2": "b", "p3": "theta"}
            elif ss.kind == "line":
                self._tb_set("p1", ss.l or DEFAULT_RADIUS * 4, "$l$ ")
                self._tb_set("p3", ss.theta, r"$\theta$ ")
                self._tb_hide("p2")
                self._tb_param = {"x": "x", "y": "y", "c": "c",
                                  "p1": "l", "p3": "theta"}
        else:
            self._tb_hide("x", "y", "c", "p1", "p2", "p3")
        self.fig.canvas.draw_idle()

    def _on_tb_submit(self, slot, text):
        if self._tb_updating:
            return
        param = self._tb_param.get(slot)
        if param is None:
            return
        if text.strip() in ("", self.MIXED):     # untouched mixed placeholder
            return
        try:
            value = float(text)
        except (ValueError, TypeError):
            self._status(f"Invalid value: {text!r}")
            self._populate_textboxes()
            return
        self._apply_param(param, value)

    # ── Collective mutators ────────────────────────────────────────────────────

    def _set_conc_all(self, value):
        value = round(max(0.1, value), 1)
        for r in self._selected_resolved():
            if r[0] == "cc":
                r[2]["c"] = value
            elif r[0] == "ss":
                r[1].c = value
            elif r[0] == "comp":
                for c in r[1].circles:
                    c["c"] = value

    def _translate_all(self, dx, dy):
        for r in self._selected_resolved():
            if r[0] == "cc":
                r[2]["x"] = round(r[2]["x"] + dx, 5); r[2]["y"] = round(r[2]["y"] + dy, 5)
            elif r[0] == "ss":
                r[1].x = round(r[1].x + dx, 5); r[1].y = round(r[1].y + dy, 5)
            elif r[0] == "comp":
                for c in r[1].circles:
                    c["x"] = round(c["x"] + dx, 5); c["y"] = round(c["y"] + dy, 5)
                for v in r[1].vertices:
                    v[0] = round(v[0] + dx, 5); v[1] = round(v[1] + dy, 5)

    def _apply_param(self, param, value):
        res = self._selected_resolved()
        if not res:
            return
        if param == "c":
            self._set_conc_all(value)
        elif param in ("x", "y"):
            cx, cy = self._centroid()
            self._translate_all(value - cx if param == "x" else 0.0,
                                value - cy if param == "y" else 0.0)
        else:
            # Size/angle fields appear only for a single element.
            self._apply_size_param(res[0], param, value)
        self._update_info()
        self._populate_textboxes()
        self.redraw()

    def _apply_size_param(self, r, param, value):
        if r[0] == "cc":
            comp, c = r[1], r[2]
            if param == "r":
                c["r"] = round(max(MIN_RADIUS, value), 5)
                idx = comp.circles.index(c)
                comp.circles = repack_after_resize(comp.circles, idx, comp.vertices)
            return
        ss = r[1]
        if param == "r":
            ss.r = round(max(MIN_RADIUS, value), 5)
        elif param == "a":
            ss.a = round(max(MIN_RADIUS, value), 5)
        elif param == "b":
            ss.b = round(max(MIN_RADIUS, value), 5)
        elif param == "l":
            ss.l = round(max(MIN_RADIUS * 2, value), 5)
        elif param == "theta":
            ss.theta = round(value % 360.0, 2)

    def _textbox_capturing(self):
        boxes = list(self._textboxes.values())
        if getattr(self, "settings", None) is not None:
            boxes += list(self.settings.textboxes.values())
        for tb in boxes:
            if getattr(tb, "capturekeystrokes", False):
                return True
        return False

    # ── Info display ────────────────────────────────────────────────────────

    def _update_info(self):
        sel = self.d.selection
        if not sel:
            self._info("")
            return
        if len(sel) == 1:
            r = self._resolve(sel[0])
            if r is None:
                self._info("")
                return
            if r[0] == "cc":
                idx = r[2].get("index")
                self._info(f"Selected:  #{idx} Packed circle")
            elif r[0] == "comp":
                n = len(r[1].circles)
                self._info(f"Selected:  Composite polygon ({n} circles)")
            else:
                ss = r[1]
                self._info(f"Selected:  #{ss.index} {ss.kind.capitalize()}")
        else:
            # Count leaf elements for a clear summary.
            xs, _, _, _ = self._leaf_values()
            self._info(f"Selected:  {len(xs)} elements")

    # ── Edit handle actions ─────────────────────────────────────────────────

    def _adjust_size(self, delta):
        res = self._selected_resolved()
        if not res:
            return
        single_cc = len(res) == 1 and res[0][0] == "cc"
        for r in res:
            if r[0] == "cc":
                comp, c = r[1], r[2]
                c["r"] = round(max(MIN_RADIUS, c["r"] + delta), 5)
                if single_cc:   # repack only for a lone packed circle
                    idx = comp.circles.index(c)
                    comp.circles = repack_after_resize(comp.circles, idx, comp.vertices)
            elif r[0] == "comp":
                for c in r[1].circles:
                    c["r"] = round(max(MIN_RADIUS, c["r"] + delta), 5)
            elif r[0] == "ss":
                ss = r[1]
                if ss.kind == "circle":
                    ss.r = round(max(MIN_RADIUS, (ss.r or DEFAULT_RADIUS) + delta), 5)
                elif ss.kind == "ellipse":
                    a0 = ss.a or DEFAULT_RADIUS
                    b0 = ss.b or DEFAULT_RADIUS
                    new_a = max(MIN_RADIUS, a0 + delta)
                    factor = new_a / a0 if a0 > 1e-9 else 1.0
                    ss.a = round(new_a, 5)
                    ss.b = round(max(MIN_RADIUS, b0 * factor), 5)
                elif ss.kind == "line":
                    ss.l = round(max(MIN_RADIUS * 2, (ss.l or DEFAULT_RADIUS * 4) + delta * 5), 5)
        self._update_info()
        self._populate_textboxes()
        self.redraw()

    def _adjust_b(self, delta):
        for r in self._selected_resolved():
            if r[0] == "ss" and r[1].kind == "ellipse":
                r[1].b = round(max(MIN_RADIUS, (r[1].b or DEFAULT_RADIUS) + delta), 5)
        self._update_info()
        self._populate_textboxes()
        self.redraw()

    def _adjust_angle(self, deg):
        rotated = False
        for r in self._selected_resolved():
            if r[0] == "ss" and r[1].kind in ("ellipse", "line"):
                r[1].theta = round((r[1].theta + deg) % 360.0, 2)
                rotated = True
        self._status("Rotated" if rotated else "Rotation applies to ellipse/line only")
        self._update_info()
        self._populate_textboxes()
        self.redraw()

    def _adjust_conc(self, delta):
        for r in self._selected_resolved():
            if r[0] == "cc":
                r[2]["c"] = round(max(0.1, r[2]["c"] + delta), 1)
            elif r[0] == "ss":
                r[1].c = round(max(0.1, r[1].c + delta), 1)
            elif r[0] == "comp":
                for c in r[1].circles:
                    c["c"] = round(max(0.1, c["c"] + delta), 1)
        self._update_info()
        self._populate_textboxes()
        self.redraw()

    def _delete_selected(self):
        res = self._selected_resolved()
        if not res:
            return
        # Collect target objects, then remove by identity (indices shift).
        comp_circles = {}   # id(comp) -> (comp, set of circle ids)
        kill_comps = set()
        kill_ss = set()
        for r in res:
            if r[0] == "cc":
                comp_circles.setdefault(id(r[1]), (r[1], set()))[1].add(id(r[2]))
            elif r[0] == "comp":
                kill_comps.add(id(r[1]))
            elif r[0] == "ss":
                kill_ss.add(id(r[1]))
        for comp, circ_ids in comp_circles.values():
            if id(comp) in kill_comps:
                continue
            comp.circles = [c for c in comp.circles if id(c) not in circ_ids]
        self.d.scene.composites = [c for c in self.d.scene.composites
                                   if id(c) not in kill_comps]
        self.d.scene.simple_sources = [s for s in self.d.scene.simple_sources
                                       if id(s) not in kill_ss]
        self.d.selection = []
        self._sync_selection_ui()
        self._status("Deleted selection")
        self.redraw()

    def _delete_polygon_or_composite(self):
        # 'x' deletes the composite(s) any selected packed-circle belongs to.
        kill = set()
        for r in self._selected_resolved():
            if r[0] in ("cc", "comp"):
                kill.add(id(r[1]))
        if not kill:
            return
        self.d.scene.composites = [c for c in self.d.scene.composites
                                   if id(c) not in kill]
        self.d.selection = []
        self._sync_selection_ui()
        self._status("Polygon deleted")
        self.redraw()

    def select_all(self):
        sels = []
        for pi, comp in enumerate(self.d.scene.composites):
            for ci in range(len(comp.circles)):
                sels.append(Selection("composite_circle", pi, ci))
        for si in range(len(self.d.scene.simple_sources)):
            sels.append(Selection("simple", si))
        self.d.selection = sels
        self._sync_selection_ui()
        self.redraw()

    # ── Redraw ──────────────────────────────────────────────────────────────

    def redraw(self):
        ax = self.ax
        xl, yl = ax.get_xlim(), ax.get_ylim()
        for a in self._dyn:
            try:
                a.remove()
            except Exception:
                pass
        self._dyn = draw_elements_on_ax(ax, self.d.scene, self._sel_keys(),
                                        show_indices=self.show_indices)

        if self.d.current_verts:
            xs = [v[0] for v in self.d.current_verts]
            ys = [v[1] for v in self.d.current_verts]
            line, = ax.plot(xs, ys, "-o", color="black", markersize=5,
                            linewidth=1.5, zorder=5)
            self._dyn.append(line)
            for v in self.d.current_verts:
                dot, = ax.plot(v[0], v[1], "s", color="goldenrod", markersize=5, zorder=6)
                self._dyn.append(dot)

        if self._line_start is not None:
            lx, ly = self._line_start
            mk, = ax.plot(lx, ly, "^", color="steelblue", markersize=8, zorder=6)
            self._dyn.append(mk)

        ax.set_xlim(xl)
        ax.set_ylim(yl)
        self.fig.canvas.draw_idle()

    # ── Globals changed ───────────────────────────────────────────────────────

    def on_globals_changed(self):
        """Redraw static decorations and reframe for Ws / orientation."""
        self.ax.cla()
        self.ax.set_facecolor("white")
        self.ax.set_aspect("equal")
        self._draw_static()
        self._update_axis_labels()
        self.ax.set_title(self._title, fontsize=13, pad=6)
        self.ax.tick_params(labelsize=10)
        self._dyn = []
        self._preview = []
        self._reframe_for_orientation()
        self.redraw()


# ── VIEW window ────────────────────────────────────────────────────────────────

class ViewWindow:

    def __init__(self, designer: "SourceDesigner"):
        self.d = designer
        self._dyn: list = []          # ax_src input artists
        self._dom_input: list = []    # ax_dom input artists (hidden when result shown)
        self._result_artists: list = []
        self._pan = False
        self._pan0 = None
        self._pan_xl = None
        self._pan_yl = None
        self._pan_ax = None
        self._solve_queue = None
        self._poll_timer = None
        self._last_result = None

        self.fig = plt.figure("Source Designer — View", figsize=(15.5, 8.5), dpi=100)
        self.fig.patch.set_facecolor("#F5F5F5")   # soft neutral figure face

        gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 3.4], wspace=0.07,
                      left=0.045, right=0.885, bottom=0.27, top=0.93)
        self.ax_src = self.fig.add_subplot(gs[0])
        self.ax_dom = self.fig.add_subplot(gs[1])
        self._setup_axes()
        self._setup_bottom()

        self.settings = SettingsPanel(
            self.fig, self.d.scene, self.d.on_globals_changed,
            rect=(0.34, 0.14, 0.34, 0.78))

        self._connect()

    def _setup_axes(self):
        g = self.d.scene.globals
        ylabel = "$z$ (m)" if g.orientation == "vertical" else "$y$ (m)"
        self.ax_src.set_facecolor("white")
        self.ax_src.set_xlabel("$x$ (m)", fontsize=12)
        self.ax_src.set_ylabel(ylabel, fontsize=12)
        self.ax_src.set_title("Source zone", fontsize=13, pad=4)
        self.ax_src.tick_params(labelsize=9)
        self.ax_dom.set_facecolor("white")
        self.ax_dom.set_xlabel("$x$ (m)", fontsize=13)
        self.ax_dom.set_title("Full simulation domain", fontsize=14, fontweight="bold", pad=6)
        self.ax_dom.tick_params(labelsize=10)

    def _setup_bottom(self):
        self._buttons: list[Button] = []

        sm = plt.cm.ScalarMappable(cmap="Reds", norm=mcolors.Normalize(0, CONC_MAX))
        sm.set_array([])
        cax = self.fig.add_axes([0.90, 0.27, 0.013, 0.66])
        cb = self.fig.colorbar(sm, cax=cax, orientation="vertical")
        cb.set_label(CB_LABEL, fontsize=9)
        cb.ax.tick_params(labelsize=8)

        # Larger, comfortably-spaced status lines (per supervisor feedback).
        self.status = self.fig.text(0.045, 0.205, "", fontsize=12, color="#333", va="top")
        self.result_text = self.fig.text(0.045, 0.160, "", fontsize=11, color="#555",
                                         family="monospace", va="top")

        gap = 0.01
        bw = (0.83 - 6 * gap) / 7
        by, bh = 0.06, 0.055
        x = 0.05

        def _btn(label, cb, hover=BTN_HOVER):
            nonlocal x
            ax = self.fig.add_axes([x, by, bw, bh])
            btn = Button(ax, label, color=BTN_BG, hovercolor=hover)
            btn.on_clicked(cb)
            self._buttons.append(btn)
            x += bw + gap
            return btn

        _btn("Draw (d)",     lambda _: self.d.show_draw(), "#c8e6c9")
        _btn("Export (s)",   lambda _: self.d.export_json(), "#e1bee7")
        self.btn_solve = _btn("Solve ▶", lambda _: self._start_solve(), "#fff3b0")
        self.btn_dl    = _btn("Download ZIP", lambda _: self._download_results())
        self.btn_plot  = _btn("Save plot",    lambda _: self._save_plot())
        _btn("Clear",        lambda _: self.d.clear_all(), "#ffcdd2")
        _btn("Settings ⚙", lambda _: self.settings.toggle(), "#ffe0b2")

        self.btn_dl.ax.set_visible(False)
        self.btn_plot.ax.set_visible(False)
        self.btn_dl.active = False
        self.btn_plot.active = False

    def _connect(self):
        c = self.fig.canvas
        c.mpl_connect("scroll_event",         self._scroll)
        c.mpl_connect("key_press_event",      self._key)
        c.mpl_connect("button_press_event",   self._press)
        c.mpl_connect("button_release_event", self._release)
        c.mpl_connect("motion_notify_event",  self._motion)
        c.mpl_connect("close_event",          self._on_close)

    def _on_close(self, _ev):
        # Let the controller know so the next "View" click rebuilds the window.
        if getattr(self.d, "view_win", None) is self:
            self.d.view_win = None

    def _scroll(self, ev):
        ax = ev.inaxes
        if ax not in (self.ax_src, self.ax_dom):
            return
        factor = 0.85 if ev.button == "up" else 1.18
        xc, yc = ev.xdata, ev.ydata
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim([xc + (v - xc) * factor for v in xl])
        ax.set_ylim([yc + (v - yc) * factor for v in yl])
        self.fig.canvas.draw_idle()

    def _press(self, ev):
        if self.settings.visible:
            return
        ax = ev.inaxes
        if ax not in (self.ax_src, self.ax_dom):
            return
        if ev.button in (2, 3) or (ev.button == 1 and ev.key == "shift"):
            self._pan = True
            self._pan0 = (ev.x, ev.y)
            self._pan_xl = ax.get_xlim()
            self._pan_yl = ax.get_ylim()
            self._pan_ax = ax

    def _release(self, ev):
        self._pan = False

    def _motion(self, ev):
        if not self._pan or not self._pan_ax or ev.x is None:
            return
        ax = self._pan_ax
        dx = ev.x - self._pan0[0]
        dy = ev.y - self._pan0[1]
        bbox = ax.get_position()
        fw = self.fig.get_size_inches()[0] * self.fig.dpi
        fh = self.fig.get_size_inches()[1] * self.fig.dpi
        xr, yr = self._pan_xl, self._pan_yl
        ax.set_xlim(xr[0] - (xr[1]-xr[0])*dx/(bbox.width*fw),
                    xr[1] - (xr[1]-xr[0])*dx/(bbox.width*fw))
        ax.set_ylim(yr[0] - (yr[1]-yr[0])*dy/(bbox.height*fh),
                    yr[1] - (yr[1]-yr[0])*dy/(bbox.height*fh))
        self.fig.canvas.draw_idle()

    def _key(self, ev):
        if getattr(self.settings, "_updating", False):
            return
        for tb in self.settings.textboxes.values():
            if getattr(tb, "capturekeystrokes", False):
                return
        if ev.key == "d":
            self.d.show_draw()
        elif ev.key == "s":
            self.d.export_json()
        elif ev.key == "escape" and self.settings.visible:
            self.settings.hide()
        elif ev.key == "q":
            plt.close("all")

    # ── Solve ───────────────────────────────────────────────────────────────

    def _start_solve(self):
        errors = self.d.scene.validate_for_export()
        if errors:
            self.status.set_text("Cannot solve: " + errors[0])
            self.fig.canvas.draw_idle()
            return
        warnings = self.d.scene.warnings_for_export()
        self.result_text.set_text("⚠ " + warnings[0] if warnings else "")

        self._solve_queue = queue.Queue()
        self.status.set_text("Solving…  (window stays responsive)")
        self.btn_solve.label.set_text("Solving…")
        self.fig.canvas.draw_idle()

        scene_snapshot = self.d.scene

        def _worker():
            try:
                from designer_solver import solve_scene
                self._solve_queue.put(("ok", solve_scene(scene_snapshot)))
            except Exception as exc:
                self._solve_queue.put(("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()
        self._poll_timer = self.fig.canvas.new_timer(interval=250)
        self._poll_timer.add_callback(self._check_solve)
        self._poll_timer.start()

    def _check_solve(self):
        if self._solve_queue is None:
            return
        try:
            kind, data = self._solve_queue.get_nowait()
        except queue.Empty:
            return
        self._poll_timer.stop()
        self._poll_timer = None
        self.btn_solve.label.set_text("Solve ▶")
        if kind == "ok":
            self._on_solve_complete(data)
        else:
            self.status.set_text(f"Solve error: {data}")
            self.result_text.set_text("")
            self.fig.canvas.draw_idle()

    def _on_solve_complete(self, result):
        self._last_result = result
        lmax = result.L_max if result.L_max is not None else "—"
        il = f"{result.interface_length:.2f} m" if result.interface_length is not None else "—"
        self.status.set_text(
            f"Solve complete  ·  L_max = {lmax} m  ·  interface = {il}  ·  "
            f"{result.wall_time:.1f} s")
        self.result_text.set_text(f"Results → {result.run_dir}")
        self._display_result(result)
        for btn in (self.btn_dl, self.btn_plot):
            btn.ax.set_visible(True)
            btn.active = True
        self.fig.canvas.draw_idle()

    def _display_result(self, result):
        for a in self._result_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._result_artists = []

        # Remove the input element shapes / source-zone band from the domain
        # panel so the result plot shows ONLY the solved field (matching the
        # solver's own output plot). The left "Source zone" panel still shows
        # the input geometry.
        for a in self._dom_input:
            try:
                a.remove()
            except Exception:
                pass
        self._dom_input = []

        ax = self.ax_dom
        X, Y = np.meshgrid(result.xaxis, result.yaxis)
        max_val = float(np.max(result.result))
        min_val = float(np.min(result.result))

        if max_val > 0:
            cf_d = ax.contourf(X, Y, result.result, levels=np.linspace(0, max_val, 11),
                               cmap="Reds", extend="max", alpha=0.9, zorder=2)
            self._result_artists.append(cf_d)
        if min_val < 0:
            cf_a = ax.contourf(X, Y, result.result, levels=np.linspace(min_val, 0, 9),
                               cmap="Blues_r", extend="min", alpha=0.9, zorder=2)
            self._result_artists.append(cf_a)
        cs = ax.contour(X, Y, result.result, levels=[0], linewidths=2,
                        colors="k", zorder=5)
        self._result_artists.append(cs)
        if result.L_max is not None:
            vl = ax.axvline(result.L_max, color="navy", linestyle=(0, (4, 3)),
                            linewidth=2.6, zorder=6,
                            label=f"$L_{{max}}$ = {result.L_max} m")
            self._result_artists.append(vl)
        try:
            ax.set_xlim(float(result.xaxis.min()), float(result.xaxis.max()))
            ax.set_ylim(float(result.yaxis.min()), float(result.yaxis.max()))
        except Exception:
            pass
        leg = ax.legend(loc="upper right", fontsize=11, framealpha=0.7)
        if leg is not None:
            self._result_artists.append(leg)
        self.fig.canvas.draw_idle()

    def _clear_result(self):
        """Remove the solved-field overlay (contours, L_max line, legend).

        Must actually remove each artist — previously the list was just reset to
        [], which orphaned the contours on the axis so they never disappeared
        (the 'black line that won't clear' and duplicate L_max legend entries).
        """
        for a in self._result_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._result_artists = []
        self._last_result = None
        for btn in (self.btn_dl, self.btn_plot):
            btn.ax.set_visible(False)
            btn.active = False

    # ── Download / save ───────────────────────────────────────────────────────

    def _download_results(self):
        if self._last_result is None:
            return
        import shutil
        from designer_solver import bundle_results
        zip_src = bundle_results(self._last_result.run_dir, self._last_result.config_dict)
        run_name = Path(self._last_result.run_dir).name
        default_name = f"aem_results_{run_name}.zip"
        dest = _ask_save_file("Save results ZIP", ".zip", default_name,
                              [("ZIP archive", "*.zip")])
        if dest is None:                          # dialog unavailable → fallback
            dest = str(EXPORT_DIR / default_name)
        if not dest:                              # user cancelled
            return
        shutil.copy2(zip_src, dest)
        self.status.set_text(f"Results saved → {dest}")
        self.fig.canvas.draw_idle()

    def _save_plot(self):
        """Save the result plot in the chosen vector/raster format (PDF/PNG/SVG)."""
        if self._last_result is None:
            return
        run_name = Path(self._last_result.run_dir).name
        default_name = f"aem_result_{run_name}.pdf"
        dest = _ask_save_file("Save result plot", ".pdf", default_name,
                              [("PDF", "*.pdf"), ("PNG image", "*.png"),
                               ("SVG vector", "*.svg")])
        if dest is None:                       # dialog unavailable → fallback
            dest = str(EXPORT_DIR / default_name)
        if not dest:                           # cancelled
            return
        try:
            self._render_result_figure(dest)
            self.status.set_text(f"Plot saved → {dest}")
        except Exception as exc:
            self.status.set_text(f"Save failed: {exc}")
        self.fig.canvas.draw_idle()

    def _render_result_figure(self, path):
        """Render the solved field to a standalone figure and save it.

        Vector (PDF/SVG) stays vector; PNG is rasterised at 200 DPI. This is
        self-contained so every format produces a clean, correct file.
        """
        res = self._last_result
        ext = Path(path).suffix.lower().lstrip(".")
        fmt = ext if ext in ("pdf", "png", "svg") else "pdf"
        dpi = 200 if fmt == "png" else 100

        fig = plt.figure(figsize=(11, 6), dpi=dpi)
        ax = fig.add_axes([0.08, 0.12, 0.82, 0.80])
        ax.set_facecolor("white")
        X, Y = np.meshgrid(res.xaxis, res.yaxis)
        max_val = float(np.max(res.result)); min_val = float(np.min(res.result))
        mappable = None
        if max_val > 0:
            mappable = ax.contourf(X, Y, res.result, levels=np.linspace(0, max_val, 11),
                                   cmap="Reds", extend="max")
        if min_val < 0:
            ax.contourf(X, Y, res.result, levels=np.linspace(min_val, 0, 9),
                        cmap="Blues_r", extend="min")
        ax.contour(X, Y, res.result, levels=[0], linewidths=2, colors="k")
        if res.L_max is not None:
            ax.axvline(res.L_max, color="navy", linestyle=(0, (4, 3)), linewidth=2.6,
                       label=f"$L_{{max}}$ = {res.L_max} m")
            ax.legend(loc="upper right", fontsize=11, framealpha=0.7)
        g = self.d.scene.globals
        ax.set_xlabel("$x$ (m)", fontsize=12)
        ax.set_ylabel("$z$ (m)" if g.orientation == "vertical" else "$y$ (m)", fontsize=12)
        ax.set_title("Full simulation domain", fontsize=13, fontweight="bold")
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=ax, pad=0.02)
            cb.set_label(CB_LABEL, fontsize=10)
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # ── Refresh ─────────────────────────────────────────────────────────────

    def refresh(self):
        for a in self._dyn:
            try:
                a.remove()
            except Exception:
                pass
        self._dyn = []
        for a in self._dom_input:
            try:
                a.remove()
            except Exception:
                pass
        self._dom_input = []

        scene = self.d.scene
        all_circles = [c for comp in scene.composites for c in comp.circles]
        all_simple = scene.simple_sources

        if not all_circles and not all_simple:
            self.ax_src.set_xlim(-0.03, scene.globals.ws + 0.03)
            self.ax_src.set_ylim(0, GRID_HEIGHT)
            self.ax_dom.set_xlim(0, 150)
            self.ax_dom.set_ylim(0, GRID_HEIGHT)
            self.status.set_text("No elements — add sources in the Draw window")
            self.fig.canvas.draw_idle()
            return

        xs, ys, rs = [], [], []
        for c in all_circles:
            xs.append(c["x"]); ys.append(c["y"]); rs.append(c["r"])
        for ss in all_simple:
            xs.append(ss.x); ys.append(ss.y)
            rs.append(max(ss.half_width(), ss.half_height(), DEFAULT_RADIUS))

        # The source-zone width is the GLOBAL Ws guide (matches the Draw window),
        # not the element extent — so both windows agree on the zone boundary.
        gws = scene.globals.ws
        elem_xmax = max(x + r for x, r in zip(xs, rs))
        zone_xmax = max(gws, elem_xmax)
        ymin = min(y - r for y, r in zip(ys, rs))
        ymax = max(y + r for y, r in zip(ys, rs))
        pad_y = max((ymax - ymin) * 0.08, 0.5)

        ax_s = self.ax_src
        ax_s.set_xlim(-0.05 * max(gws, 0.01), 1.10 * max(zone_xmax, 0.01))
        ax_s.set_ylim(ymin - pad_y, ymax + pad_y)
        ax_s.set_aspect("equal")
        ax_s.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax_s.yaxis.set_major_locator(MaxNLocator(nbins=8))
        l1 = ax_s.axvline(0.0, color="gold", lw=1.2, ls="--", zorder=4)
        l2 = ax_s.axvline(gws, color="gold", lw=1.2, ls="--", zorder=4)
        t = ax_s.text(0.98, 0.99, f"Ws = {gws:g} m", transform=ax_s.transAxes,
                      ha="right", va="top", fontsize=9, color="goldenrod",
                      bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"))
        self._dyn.extend([l1, l2, t])
        self._dyn.extend(draw_elements_on_ax(ax_s, scene))

        ax_d = self.ax_dom
        ax_d.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax_d.yaxis.set_major_locator(MaxNLocator(nbins=8))

        if self._last_result is not None and self._result_artists:
            # A solved field is showing — keep the domain panel result-only and
            # framed to the solved grid (don't redraw input shapes over it).
            ax_d.set_xlim(float(self._last_result.xaxis.min()),
                          float(self._last_result.xaxis.max()))
            ax_d.set_ylim(float(self._last_result.yaxis.min()),
                          float(self._last_result.yaxis.max()))
        else:
            # No result yet — preview the input geometry in the domain panel.
            span = ax_d.axvspan(0.0, gws, color="gold", alpha=0.18, zorder=1,
                                label=f"Source zone (x ≤ {gws:g} m)")
            self._dom_input.append(span)
            self._dom_input.extend(draw_elements_on_ax(ax_d, scene))
            leg = ax_d.legend(loc="upper right", fontsize=9, framealpha=0.6)
            self._dom_input.append(leg)
            ax_d.set_xlim(min(xs) - 2.0, max(xs) + 150.0)
            ax_d.set_ylim(ymin - pad_y, ymax + pad_y)

        n_comp = len(scene.composites)
        n_simp = len(scene.simple_sources)
        self.status.set_text(
            f"{n_comp} polygon(s) · {len(all_circles)} packed circles · "
            f"{n_simp} simple source(s).  Press Solve ▶ to run.")
        self.fig.canvas.draw_idle()

    def on_globals_changed(self):
        g = self.d.scene.globals
        ylabel = "$z$ (m)" if g.orientation == "vertical" else "$y$ (m)"
        self.ax_src.set_ylabel(ylabel, fontsize=12)
        self.refresh()


# ── Controller ─────────────────────────────────────────────────────────────────

class SourceDesigner:

    def __init__(self):
        self.scene = Scene()
        self.current_verts: list[list[float]] = []
        self.selection: list[Selection] = []     # multi-select set (ordered)
        self.draw_win = DrawWindow(self)
        self.view_win: ViewWindow | None = None

    def show_view(self):
        # Recreate the window if it was never opened or was closed (its figure
        # number no longer exists), so "View" always reopens it.
        if (self.view_win is None
                or not plt.fignum_exists(self.view_win.fig.number)):
            self.view_win = ViewWindow(self)
        self.view_win.refresh()
        try:
            self.view_win.fig.canvas.manager.show()
        except Exception:
            pass

    def show_draw(self):
        try:
            self.draw_win.fig.canvas.manager.show()
        except Exception:
            pass
        self.draw_win.redraw()

    def on_globals_changed(self):
        self.draw_win.on_globals_changed()
        if self.view_win:
            self.view_win.on_globals_changed()

    # ── Export / Import ─────────────────────────────────────────────────────

    def export_json(self):
        if not self.scene.has_elements():
            self.draw_win._status("Nothing to export")
            return
        errors = self.scene.validate_for_export()
        if errors:
            self.draw_win._status(f"Export error: {errors[0]}")
            print(f"  Export blocked: {errors[0]}")
            return
        for w in self.scene.warnings_for_export():
            print(f"  Warning: {w}")
        try:
            config = self.scene.to_config_dict()
        except ValueError as e:
            self.draw_win._status(f"Export error: {e}")
            return
        fpath = EXPORT_DIR / f"source_config_{uuid.uuid4().hex[:8]}.json"
        with open(fpath, "w") as f:
            json.dump(config, f, indent=4)
        n = sum(len(c.circles) for c in self.scene.composites) + len(self.scene.simple_sources)
        msg = f"Exported {n} elements → {fpath.name}"
        print(f"\n  {msg}")
        # to_config_dict reindexed the live scene → refresh on-canvas labels.
        self.draw_win.redraw()
        self.draw_win._status(msg)
        if self.view_win:
            self.view_win.status.set_text(msg)
            self.view_win.fig.canvas.draw_idle()

    def import_json(self):
        path = _ask_open_file("Load JSON config",
                              [("JSON files", "*.json"), ("All files", "*")])
        if path is None:
            self.draw_win._status("File dialog unavailable")
            return
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            new_scene = Scene.from_config_dict(data)
        except Exception as e:
            print(f"  Load error: {e}")
            self.draw_win._status(f"Load error: {e}")
            return
        self.scene = new_scene
        self.current_verts = []
        self.selection = []
        # Rebind the new scene into both windows' settings panels.
        self.draw_win.settings.scene = new_scene
        self.draw_win.settings.refresh_values()
        self.draw_win._set_active_tool("select")
        self.draw_win.on_globals_changed()
        if self.view_win:
            self.view_win.settings.scene = new_scene
            self.view_win._clear_result()
            self.view_win.refresh()
        n = len(new_scene.simple_sources) + sum(len(c.circles) for c in new_scene.composites)
        msg = f"Loaded {n} elements from {Path(path).name}"
        print(f"\n  {msg}")
        self.draw_win._status(msg)

    def clear_all(self):
        self.scene = Scene(globals=self.scene.globals)   # keep global params
        self.current_verts = []
        self.selection = []
        self.draw_win.settings.scene = self.scene
        self.draw_win._set_active_tool("polygon")
        if self.view_win:
            self.view_win.settings.scene = self.scene
            self.view_win._clear_result()
            self.view_win.refresh()

    def run(self):
        plt.show()


if __name__ == "__main__":
    SourceDesigner().run()
