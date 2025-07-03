import math
import numpy as np
from enum import Enum
from mathieu_functions_OG import Mathieu

class ATElementType(Enum):
    Circle = 0
    Line = 1

class ATElement:
    def __init__(self, kind: ATElementType, x: float, y: float, c: float, r = 1.0, theta: float = math.pi/2):
        self.kind = kind
        self.x = x
        self.y = y
        self.c = c
        self.r = r
        self.theta = theta
        self.d = None
        self.q = None
        self.outline = []
        self.m = None

    def calc_d_q(self, alpha_t, alpha_l, beta):
        r = self.r
        a = self.r
        self.d = math.sqrt((a * math.sqrt(alpha_l / alpha_t))**2 - a**2)
        self.q = (self.d ** 2 * beta ** 2) / 4
        self.m = Mathieu(self.q)

    def set_outline(self, num_cp):
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


    def uv(self, x, y, alpha_l, alpha_t):
        d = self.d
        Y = math.sqrt(alpha_l / alpha_t) * y
        B = x**2 + Y**2 - d**2
        sqrt_term = math.sqrt(B**2 + 4 * d**2 * x**2)
        p = (-B + sqrt_term) / (2 * d**2)
        q = (-B - sqrt_term) / (2 * d**2)

        psi_0 = math.asin(min(max(math.sqrt(p), -1.0), 1.0))
        if Y >= 0 and x >= 0:
            psi = psi_0
        elif Y < 0 and x >= 0:
            psi = math.pi - psi_0
        elif Y <= 0 and x < 0:
            psi = math.pi + psi_0
        else:
            psi = 2 * math.pi - psi_0

        eta = 0.5 * math.log(1 - 2 * q + 2 * math.sqrt(q**2 - q))
        return eta, psi

