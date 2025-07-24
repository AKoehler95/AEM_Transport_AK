# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library:
import math
import timeit
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import json
import os
from concurrent.futures import ProcessPoolExecutor

# External library:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_temp
import matplotlib as mpl

# Local imports:
from at_config import ATConfiguration
from at_element import ATElement, ATElementType


logger = logging.getLogger(__name__)

from multiprocessing import Pool, cpu_count

_shared_elements = None
_shared_coeff = None
_shared_num_terms = None
_shared_alpha_l = None
_shared_alpha_t = None
_shared_beta = None
_shared_ca = None
_shared_gamma = None

def _init_pool(elements, coeff, num_terms, alpha_l, alpha_t, beta, ca, gamma):
    """
    Initializes shared variables for worker processes in the multiprocessing pool.
    
    Parameters
    ----------
    elements : list
        List of ATElement objects.
    coeff : array_like
        Flattened coefficient array.
    num_terms : int
        Number of Mathieu function terms.
    alpha_l : float
        Longitudinal dispersivity.
    alpha_t : float
        Transverse dispersivity.
    beta : float
        Exponential decay coefficient.
    ca : float
        Acceptor concentration threshold.
    gamma : float
        Stoichiometric ratio.
    """
    global _shared_elements, _shared_coeff, _shared_num_terms
    global _shared_alpha_l, _shared_alpha_t, _shared_beta, _shared_ca, _shared_gamma

    _shared_elements = elements
    _shared_coeff = coeff
    _shared_num_terms = num_terms
    _shared_alpha_l = alpha_l
    _shared_alpha_t = alpha_t
    _shared_beta = beta
    _shared_ca = ca
    _shared_gamma = gamma


def _compute_point_shared(args):
    """
     Computes concentration at a point using shared variables.
    
     Parameters
     ----------
     args : tuple
         (x, y) coordinates.
    
     Returns
     -------
     float
         Concentration at the given point.
     """
    x, y = args
    return _compute_point((
        x, y, _shared_elements, _shared_coeff,
        _shared_num_terms, _shared_alpha_l, _shared_alpha_t,
        _shared_beta, _shared_ca, _shared_gamma
    ))


def _compute_point(args):
    """
    Computes the concentration at a single (x, y) point.

    Parameters
    ----------
    args : tuple
        Contains x, y coordinates and all necessary parameters
        for evaluating the concentration field.

    Returns
    -------
    float
        Calculated concentration at the point.
    """
    x, y, elements, coeff, num_terms, alpha_l, alpha_t, beta, ca, gamma = args
    total_terms = 2 * num_terms - 1

    F = 0.0
    for idx, elem in enumerate(elements):
        dx = x - elem.x
        dy = y - elem.y
        eta, psi = elem.uv(dx, dy, alpha_l, alpha_t)

        block = coeff[idx * total_terms: (idx + 1) * total_terms]

        Fi = block[0] * elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real

        for j in range(1, num_terms):
            Fi += block[2*j - 1] * elem.m.se(j, psi).real * elem.m.Ko(j, eta).real
            Fi += block[2*j    ] * elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real

        F += Fi

    total = F * np.exp(beta * x)
    if total > ca:
        conc = (total - ca) / gamma
    else:
        conc = total - ca

    for elem in elements:
        dx = x - elem.x
        dy = y - elem.y
        if elem.kind == ATElementType.Circle:
            if dx*dx + dy*dy <= elem.r**2:
                conc = elem.c
                break

    return conc


def create_mirrored_element(elem):
    """
    Creates a mirrored image of an ATElement across the x-axis.
    
    Parameters
    ----------
    elem : ATElement
        Original element to be mirrored.
    
    Returns
    -------
    ATElement
        Mirrored element with flipped y and concentration.
    """
    return ATElement(
        kind=elem.kind,
        x=elem.x,
        y=-elem.y,
        c=-elem.c,
        r=elem.r,
        theta=elem.theta if elem.kind == ATElementType.Line else math.pi / 2  # ensure theta is preserved
    )


class ATSimulation:
    def __init__(self, config: ATConfiguration):
        self.config: ATConfiguration = config
        self.coeff: np.ndarray = np.zeros((10, 10))
        self.xaxis: np.ndarray = np.arange(0, 10, 1)
        self.yaxis: np.ndarray = np.arange(0, 10, 1)
        self.result: np.ndarray = np.zeros((10, 10))
        self.L_max = None

    def run(self):
        if len(self.config.elements) < 1:
            raise ValueError("Simulation requires at least one element.")

        if self.config.orientation == "vertical":
            updated_elements = []
            for elem in self.config.elements:
                if elem.y > -(elem.r+0.1):
                    raise ValueError(f"Element '{elem.id}' must have y < -(r+0.1) for vertical orientation.")
                updated_elements.append(elem)
                mirrored = create_mirrored_element(elem)
                mirrored.id = f"image_{elem.id}"
                updated_elements.append(mirrored)
            self.config.elements = updated_elements
            self.config.dom_ymax = 0

        start: float = timeit.default_timer()

        alpha_t: float = self.config.alpha_t
        alpha_l: float = self.config.alpha_l
        beta: float = self.config.beta
        gamma: float = self.config.gamma
        ca: float = self.config.ca
        n: int = self.config.num_terms
        M: int = self.config.num_cp

        # update all elems
        for elem in self.config.elements:
            elem.calc_d_q(alpha_t, alpha_l, beta)
            elem.set_outline(M)

        self.solve_system(alpha_l, alpha_t, beta, gamma, ca, n, M)

        # print("Coefficients:")
        # print(self.coeff)

        self.conc_array(self.config.dom_xmin, self.config.dom_ymin,
                        self.config.dom_xmax, self.config.dom_ymax,
                        self.config.dom_inc)

        # print(self.result_tuple[0])
        # print(self.result_tuple[1])
        # print(self.result_tuple[2])

        stop: float = timeit.default_timer()
        cpu_time = timedelta(seconds=int(stop - start))
        print(f"Computation time [hh:mm:ss]: {cpu_time}")

        self.calculate_lmax()
        self.check_domain_adequacy()
        self.check_concentration_range()

        results_dir = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)
        self.run_index = self.get_next_run_index(results_dir)

        self.print_statistics(cpu_time)
        self.plot_result()

    def solve_system(self, alpha_l: float, alpha_t: float, beta: float,
            gamma: float, ca: float, n: int, M: int):
        A: list = []
        b: list = []

        num_elements = len(self.config.elements)
        for i, e_i in enumerate(self.config.elements):
            for (x_cp, y_cp) in e_i.outline:
                row: list = []
                for j, e_j in enumerate(self.config.elements):
                    # transform (x_cp, y_cp) to e_j's local coords
                    dx: float = x_cp - e_j.x
                    dy: float = y_cp - e_j.y
                    eta_j, psi_j = e_j.uv(dx, dy, alpha_l, alpha_t)
                    if e_j.kind == ATElementType.Line and i == j:
                        eta: float = 0.0
                    else:
                        eta = eta_j
                    psi: float = psi_j
                    row += self.build_row(e_j, eta, psi, n)
                b.append(self.f_target(x_cp, e_i, ca, gamma, beta))
                A.append(row)

        A: np.ndarray = np.array(A)
        b: np.ndarray = np.array(b)
        self.coeff = np.linalg.lstsq(A, b, rcond=None)[0]

    def build_row(self, elem, eta: float, psi: float, n: int) -> list:
        row: list = [elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real]
        for j in range(1, n):
            row.append(elem.m.se(j, psi).real * elem.m.Ko(j, eta).real)
            row.append(elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real)
        return row

    def f_target(self, x: float, elem, ca: float, gamma: float, beta: float):
        Ci: float = elem.c
        is_image = elem.id.lower().startswith("image")

        # Donor
        if Ci > 0 and not is_image:
            return (Ci * gamma + ca) * np.exp(-beta * x)

        # Acceptor
        if Ci < 0 and not is_image:
            return (Ci + ca) * np.exp(-beta * x)

        # Image‐donor (neg conc & image)
        if Ci < 0 and is_image:
            return -((abs(Ci) * gamma + ca) * np.exp(-beta * x))

        # Image‐acceptor (pos conc & image)
        if Ci > 0 and is_image:
            return -((Ci + ca) * np.exp(-beta * x))

    def calc_c(self, x: float, y: float) -> float:
        n: int = self.config.num_terms
        alpha_l: float = self.config.alpha_l
        alpha_t: float = self.config.alpha_t

        total_coeffs: list = self.coeff
        total_terms: int = 2 * n - 1
        F: float = 0.0

        for idx, elem in enumerate(self.config.elements):
            dx: float = x - elem.x
            dy: float = y - elem.y
            eta, psi = elem.uv(dx, dy, alpha_l, alpha_t)
            coeffs: list = total_coeffs[idx * total_terms:(idx + 1) * total_terms]

            Fi: float = coeffs[0] * elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real
            for j in range(1, n):
                Fi += coeffs[2 * j - 1] * elem.m.se(j, psi).real * elem.m.Ko(j, eta).real
                Fi += coeffs[2 * j] * elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real

            F += Fi

        total: float = F * np.exp(self.config.beta * x)
        if total > self.config.ca:
            return (total - self.config.ca) / self.config.gamma
        else:
            return total - self.config.ca

    def conc_array(self, xmin: float, ymin: float, xmax: float, ymax: float, inc: float):
        """
        Parallel computation of concentration over the domain grid

        Parameters
        ----------
        xmin : float
            Minimum x-coordinate of the domain
        ymin : float
            Minimum y-coordinate of the domain
        xmax : float
            Maximum x-coordinate of the domain
        ymax : float
            Maximum y-coordinate of the domain
        inc : float
            Grid Spacing
        """
        self.xaxis = np.arange(xmin, xmax + inc, inc)
        self.yaxis = np.arange(ymin, ymax + inc, inc)

        xs, ys = np.meshgrid(self.xaxis, self.yaxis)
        coords = list(zip(xs.ravel(), ys.ravel()))

        pool_args = (
            self.config.elements,
            self.coeff,
            self.config.num_terms,
            self.config.alpha_l,
            self.config.alpha_t,
            self.config.beta,
            self.config.ca,
            self.config.gamma
        )
    
        with Pool(processes=cpu_count(), initializer=_init_pool, initargs=pool_args) as pool:
            flat = pool.map(_compute_point_shared, coords)
    
        self.result = np.array(flat).reshape(xs.shape)
        self.result_tuple = (self.xaxis, self.yaxis, self.result)


    def generate_filename_suffix(self) -> str:
        n_elements: int = len(self.config.elements)
        num_terms: int = self.config.num_terms
        inc: float = self.config.dom_inc

        has_acceptor: bool = any(elem.c < 0 for elem in self.config.elements)
        acceptor_str: str = "acceptor-yes" if has_acceptor else "acceptor-no"

        return f"{n_elements}-elements_{num_terms}-terms_{acceptor_str}_inc-{inc}"

    def get_next_run_index(self, results_dir: str) -> int:
        if not os.path.exists(results_dir):
            return 1

        existing_files = os.listdir(results_dir)
        max_index: int = 0

        for filename in existing_files:
            if filename.startswith('run_') and '_' in filename[4:]:
                try:
                    index_str: str = filename[4:].split('_')[0]
                    index: int = int(index_str)
                    max_index = max(max_index, index)
                except ValueError:
                    continue

        return max_index + 1

    def calculate_lmax(self):
        fig_temp = plt_temp.figure()
        contour_temp = plt_temp.contour(self.result, levels=[0])

        paths = contour_temp.get_paths()
        if paths and len(paths[0].vertices) > 0:
            Lmax = paths[0]
            self.L_max = int(np.max(Lmax.vertices[:, 0]) * self.config.dom_inc)
        else:
            self.L_max = None

        plt_temp.close(fig_temp)

        if self.L_max is not None:
            print('Lmax =', self.L_max)
        else:
            print("L_max: -")

    def check_domain_adequacy(self):
        if self.L_max is None:
            return

        # domain dimension
        domain_width: float = self.config.dom_xmax - self.config.dom_xmin

        # check if L_max is close to the maximum x-domain using threshold
        x_threshold: float = 0.95
        max_x_distance: float = self.L_max - self.config.dom_xmin

        # Check X-direction (most critical for L_max)
        if max_x_distance >= x_threshold * domain_width:
            warning_msg = (f"WARNING: L_max ({self.L_max:.1f} m) is capped by x-domain boundary! "
                           f"L_max extends to {max_x_distance / domain_width * 100:.1f}% of domain width. ")
            print(warning_msg)

    def check_concentration_range(self):
        if len(self.config.elements) == 0:
            return

        # Get min and max concentration values from elements
        element_concentrations: list = [elem.c for elem in self.config.elements]
        min_expected: float = -8.0
        if min(element_concentrations) < min_expected:
            min_expected = min(element_concentrations)
        max_expected: float = max(element_concentrations)

        result_min: float = np.min(self.result)
        result_max: float = np.max(self.result)

        # tolerance
        tolerance: float = 0.01 * max(abs(min_expected), abs(max_expected))

        warnings_issued: bool = False

        # outside of expected range
        outside_range_count: int = np.sum((self.result > max_expected + tolerance) |
                                     (self.result < min_expected - tolerance))
        total_points: int = self.result.size

        percentage: float = (outside_range_count / total_points) * 100

        if percentage > 1.0:
            warning_msg: str = (f"WARNING: {outside_range_count} points ({percentage:.2f}%) are outside expected range "
                           f"[{min_expected:.3f}, {max_expected:.3f}] mg/l")
            print(warning_msg)
            warnings_issued = True

        if warnings_issued:
            print("\n*** CONCENTRATION RANGE SUMMARY ***")
            print(f"Expected range: [{min_expected:.3f}, {max_expected:.3f}] mg/l")
            print(f"Actual range: [{result_min:.3f}, {result_max:.3f}] mg/l")
            print(f"Element concentrations: {element_concentrations}")
            print("This may indicate numerical issues or insufficient domain resolution.\n")
        else:
            print(f"All concentration values within expected range [{min_expected:.3f}, {max_expected:.3f}] mg/l")

    def print_statistics(self, cpu_time):
        from math import cos, sin, pi

        phi = np.linspace(0, 2 * pi, 360)
        stats = {}
        x_test: np.ndarray = np.zeros((1, 1))
        y_test: np.ndarray = np.zeros((1, 1))

        print("\n=== ELEMENT BOUNDARY STATISTICS ===")
        for idx, elem in enumerate(self.config.elements):
            if elem.kind == ATElementType.Circle:
                phi = np.linspace(0, 2 * pi, 360)
                x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
                y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)

            elif elem.kind == ATElementType.Line:
                half_len:float = elem.r
                s = np.linspace(-half_len + 1e-9, half_len - 1e-9, 360)
                theta: float = elem.theta
                x_test = elem.x + s * cos(theta)
                y_test = elem.y + s * sin(theta)

            else:
                raise ValueError(f"Unknown element type: {elem.kind}")

            Err: list = [self.calc_c(x, y) for x, y in zip(x_test, y_test)]
            min_val: float = round(np.min(Err), 9)
            max_val: float = round(np.max(Err), 9)
            mean_val: float = round(np.mean(Err), 9)
            std_val: float = round(np.std(Err), 9)

            print(f'Element {idx + 1}:')
            print(f'  Min = {min_val} mg/l')
            print(f'  Max = {max_val} mg/l')
            print(f'  Mean = {mean_val} mg/l')
            print(f'  Standard Deviation = {std_val} mg/l')

            stats[f"Min{idx + 1}"] = min_val
            stats[f"Max{idx + 1}"] = max_val
            stats[f"Mean{idx + 1}"] = mean_val
            stats[f"Std{idx + 1}"] = std_val

        results_dir: str = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)

        filename_suffix: str = self.generate_filename_suffix()
        stats_filename: str = os.path.join(results_dir, f"run_{self.run_index:04d}_stats_{filename_suffix}.txt")

        # save statistics with config
        with open(stats_filename, "w") as f:
            # config parameters
            f.write("=== CONFIGURATION PARAMETERS ===\n")
            f.write(f"Number of elements: {len(self.config.elements)}\n")
            f.write(f"Number of terms: {self.config.num_terms}\n")
            f.write(f"Domain increment: {self.config.dom_inc}\n")
            f.write(f"Alpha_t: {self.config.alpha_t}\n")
            f.write(f"Alpha_l: {self.config.alpha_l}\n")
            f.write(f"Beta: {self.config.beta}\n")
            f.write(f"Gamma: {self.config.gamma}\n")
            f.write(f"Ca: {self.config.ca}\n")
            f.write(f"Number of control points: {self.config.num_cp}\n")
            f.write(f"Orientation: {self.config.orientation}\n")
            f.write(f"Domain: x[{self.config.dom_xmin}, {self.config.dom_xmax}], y[{self.config.dom_ymin}, {self.config.dom_ymax}]\n")

            # source geometry
            f.write("\n=== ELEMENTS ===\n")
            for idx, elem in enumerate(self.config.elements):
                if (elem.kind == ATElementType.Circle):
                    f.write(f"Element {idx + 1}:kind={elem.kind}, x={elem.x}, y={elem.y}, r={elem.r}, c={elem.c}, id={elem.id}\n")
                else:
                    f.write(f"Element {idx + 1}:kind={elem.kind}, x={elem.x}, y={elem.y}, r={elem.r}, theta={elem.theta}, c={elem.c}, id={elem.id}\n")

            f.write("\n=== COMPUTATION INFO ===\n")
            f.write(f"CPU Time [hh:mm:ss]: {cpu_time}\n")

            if self.L_max is not None:
                f.write(f"\nL_max: {self.L_max}\n")

            # statistics
            f.write("\n=== ELEMENT BOUNDARY STATISTICS ===\n")
            for k, v in stats.items():
                f.write(f"{k} = {v} mg/l\n")

        print(f"Statistics saved to: {stats_filename}")

        # --- PLOT ABSOLUTE ERROR ALONG BOUNDARIES ---
        mpl.rcParams.update({'font.size': 22})
        plt.figure(figsize=(16, 9), dpi=300)

        styles = ['-', '--', ':', '-.']
        for idx, elem in enumerate(self.config.elements):
            # compute absolute error
            Err: list = [abs(self.calc_c(x, y)) for x, y in zip(x_test, y_test)]

            plt.plot(
                phi,
                Err,
                linestyle=styles[idx % len(styles)],
                linewidth=2,
                label=f'Element {idx + 1}'
            )

        plt.xlim([0, 2 * pi])
        plt.xlabel('Angle (°)')
        plt.ylabel('Absolute error [mg/l]')
        plt.xticks(
            np.linspace(0, 2 * pi, 13),
            np.linspace(0, 360, 13).astype(int)
        )
        plt.legend(loc='best')
        plt.tight_layout()

        error_filename = os.path.join(
            results_dir,
            f"run_{self.run_index:04d}_error_{filename_suffix}.pdf"
        )
        plt.savefig(error_filename)
        print(f"Error plot saved to: {error_filename}")

    def plot_result(self):
        max_val: float = np.max(self.result)
        min_val: float = np.min(self.result)
        abs_min: float = abs(min_val)
        xmin, xmax = self.config.dom_xmin, self.config.dom_xmax
        ymin, ymax = self.config.dom_ymin, self.config.dom_ymax

        donor_levels = np.linspace(0, max_val, 11)
        acceptor_levels = np.linspace(-abs_min, 0, 9)

        plt.figure(figsize=(16, 9), dpi=300)
        mpl.rcParams.update({"font.size": 22})

        X, Y = np.meshgrid(self.xaxis, self.yaxis)

        donor = plt.contourf(
            X, Y,
            self.result,
            levels=donor_levels,
            cmap='Reds',
            extend='max'
        )
        acceptor = plt.contourf(
            X, Y,
            self.result,
            levels=acceptor_levels,
            cmap='Blues_r',
            extend='min'
        )
        Plume_max = plt.contour(
            X, Y,
            self.result,
            levels=[0],
            linewidths=2,
            colors='k'
        )

        x_ticks = list(np.arange(xmin, xmax, 100))
        if x_ticks[-1] < xmax:
            x_ticks.append(xmax)
        y_ticks = list(np.arange(ymin, ymax, 10))
        if y_ticks[-1] < ymax:
            y_ticks.append(ymax)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        plt.xlabel("$x$ (m)")
        plt.ylabel("$z$ (m)" if self.config.orientation == "vertical" else "$y$ (m)")

        plt.subplots_adjust(bottom=0.20)

        cbar_donor = plt.colorbar(
            donor,
            ticks=donor_levels,
            label='Electron donor concentration [mg/l]',
            location='bottom',
            pad=0.02,
            aspect=75,
        )
        cbar_donor.ax.tick_params(labelsize=14)

        cbar_acceptor = plt.colorbar(
            acceptor,
            ticks=acceptor_levels,
            label='Electron acceptor concentration [mg/l]',
            location='bottom',
            pad=0.12,
            aspect=75,
        )
        cbar_acceptor.set_ticklabels([f"{abs(t):.0f}" for t in acceptor_levels])
        cbar_acceptor.ax.tick_params(labelsize=14)

        if self.config.plot_aspect == "scaled":
            plt.axis("scaled")
        else:  # "auto"
            plt.axis("auto")

        plt.tight_layout()

        results_dir: str = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)
        filename_suffix: str = self.generate_filename_suffix()
        plot_filename: str = os.path.join(
            results_dir,
            f"run_{self.run_index:04d}_plot_{filename_suffix}.pdf"
        )
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")