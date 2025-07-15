# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library:
import math
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
        self.d: float = 0.0
        self.q: float = 0.0
        self.m: Mathieu = Mathieu(0.0)
        self.outline: list = []
        self.label: str = ""
        self.id: str = ""

    def calc_d_q(self, alpha_t: float, alpha_l: float, beta: float):
        a = self.r
        self.d = math.sqrt((a * math.sqrt(alpha_l / alpha_t))**2 - a**2)
        self.q = (self.d ** 2 * beta ** 2) / 4
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

    def uv(self, x: float, y: float, alpha_l: float,
            alpha_t: float) -> tuple[float, float]:
        d: float = self.d
        Y: float = math.sqrt(alpha_l / alpha_t) * y
        B: float = x**2 + Y**2 - d**2
        sqrt_term: float = math.sqrt(B**2 + 4 * d**2 * x**2)
        p: float = (-B + sqrt_term) / (2 * d**2)
        q: float = (-B - sqrt_term) / (2 * d**2)

        psi_0: float = math.asin(min(max(math.sqrt(p), -1.0), 1.0))
        if Y >= 0.0 and x >= 0.0:
            psi: float = psi_0
        elif Y < 0.0 and x >= 0.0:
            psi = math.pi - psi_0
        elif Y <= 0.0 and x < 0.0:
            psi = math.pi + psi_0
        else:
            psi = 2.0 * math.pi - psi_0

        eta: float = 0.5 * math.log(1 - 2 * q + 2 * math.sqrt(q**2 - q))
        return (eta, psi)
