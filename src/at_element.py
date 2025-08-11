# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library:
import math
import cmath
from enum import Enum

# External library:
import numpy as np

# Local imports:
from mathieu_functions_OG import Mathieu


class ATElementType(Enum):
    """
    This enum class defines the various element types.

    - Circle: x, y and radius
    - Line: x1, y1 and x2, y2
    """
    Circle = 0
    Line = 1


class ATElement:
    def __init__(self, kind: ATElementType, x: float, y: float, c: float, r=1.0, theta: float = math.pi/2):
        self.kind = kind
        self.x: float = x
        self.y: float = y
        self.c: float = c
        self.r: float = r
        self.theta: float = theta
        self.theta_eff: float | None = None # effective angle after anisotropy stretch
        self.d: float = 0.0
        self.q: float = 0.0
        self.m: Mathieu = Mathieu(0.0)
        self.outline: list = []
        self.label: str = ""
        self.id: str = ""

    # ---- CHANGED: correct d, q per element, then init OG Mathieu ----
    def calc_d_q(self, alpha_t: float, alpha_l: float, beta: float):
        a = float(self.r)
        s = math.sqrt(alpha_l / alpha_t)
        eps = 1e-15

        if self.kind == ATElementType.Circle:
            # circle -> ellipse (a, s a)
            s2 = s * s
            if s2 <= 1.0 + eps:
                self.d = max(a * math.sqrt(max(s2 - 1.0, eps)), eps)
            else:
                self.d = math.sqrt((s * a) ** 2 - a ** 2)
            self.theta_eff = None  # irrelevant for circles

        elif self.kind == ATElementType.Line:
            # line deforms under global stretch: use stretched angle & half-length
            cθ, sθ = math.cos(self.theta), math.sin(self.theta)
            self.theta_eff = math.atan2(s * sθ, cθ)                 # θ_s
            self.d = max(a * math.sqrt(cθ * cθ + (s * sθ) ** 2), eps)  # a_s

        else:
            raise ValueError(f"Unknown element type: {self.kind}")

        self.q = (self.d ** 2 * beta ** 2) / 4.0
        self.m = Mathieu(self.q)

    def set_outline(self, num_cp: int):
        if self.kind == ATElementType.Circle:
            phi = np.linspace(0, 2 * math.pi, num_cp, endpoint=False)
            self.outline = [
                (self.x + self.r * math.cos(p),
                 self.y + self.r * math.sin(p))
                for p in phi
            ]

        elif self.kind == ATElementType.Line:
            half_len = self.r
            t_vals = np.linspace(-half_len, half_len, num_cp)
            self.outline = [
                (self.x + t * math.cos(self.theta),
                 self.y + t * math.sin(self.theta))
                for t in t_vals
            ]
        else:
            raise ValueError(f"Unknown element kind: {self.kind}")

    # ---- CHANGED: rotation-aware complex elliptical map (Bakker) ----
    def uv(self, x: float, y: float, alpha_l: float, alpha_t: float) -> tuple[float, float]:
        """
        Map local offsets (dx, dy) -> (eta, psi), respecting anisotropy and element type.
        Sequence:
          (i) global stretch (x, s*y)
          (ii) rotate into element frame:
               - Circle: no rotation
               - Line: rotate by -θ_s (stretched angle)
          (iii) acosh with stable branch:
               - Circle: z = (Y + i*X)/d  (matches your OG circle mapping)
               - Line:   z = (X + i*Y)/d  (puts η=0 along the line)
        """
        s = math.sqrt(alpha_l / alpha_t)
        x_s, y_s = x, s * y

        if self.kind == ATElementType.Line:
            # rotate by effective stretched angle
            θ = self.theta_eff if self.theta_eff is not None else self.theta
            cθ, sθ = math.cos(θ), math.sin(θ)
            X = cθ * x_s + sθ * y_s
            Y = -sθ * x_s + cθ * y_s
            z = complex(X, Y) / (self.d if self.d != 0.0 else 1e-15)  # (X + i*Y)/d
        else:
            # circle: no rotation
            X, Y = x_s, y_s
            z = complex(Y, X) / (self.d if self.d != 0.0 else 1e-15)  # (Y + i*X)/d

        # acosh via stable branch; enforce η >= 0 for consistency
        w = cmath.log(z + cmath.sqrt((z - 1.0) * (z + 1.0)))
        if w.real < 0.0:
            w = -w

        eta = w.real
        psi = w.imag % (2.0 * math.pi)
        return (eta, psi)

