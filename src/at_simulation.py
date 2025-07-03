import timeit
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import json
import os
import matplotlib.pyplot as plt_temp
import math
from concurrent.futures import ProcessPoolExecutor
from matplotlib.patches import Circle

from at_config import ATConfiguration
from at_element import ATElement, ATElementType

logger = logging.getLogger(__name__)

def _compute_point(args):
    x, y, elements, coeff, num_terms, alpha_l, alpha_t, beta, ca, gamma = args
    total_terms = 2 * num_terms - 1

    F = 0.0
    for idx, elem in enumerate(elements):
        dx = x - elem.x
        dy = y - elem.y
        eta, psi = elem.uv(dx, dy, alpha_l, alpha_t)

        block = coeff[idx * total_terms : (idx + 1) * total_terms]

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
        self.config = config
        self.coeff = None
        self.xaxis = None
        self.yaxis = None
        self.result = None
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

        start = timeit.default_timer()

        alpha_t = self.config.alpha_t
        alpha_l = self.config.alpha_l
        beta = self.config.beta
        gamma = self.config.gamma
        ca = self.config.ca
        n = self.config.num_terms
        M = self.config.num_cp

        # update all elems
        for elem in self.config.elements:
            elem.calc_d_q(alpha_t, alpha_l, beta)
            elem.set_outline(M)

        self.solve_system(alpha_l, alpha_t, beta, gamma, ca, n, M)

        #print("Coefficients:")
        #print(self.coeff)

        self.conc_array(self.config.dom_xmin, self.config.dom_ymin,
                        self.config.dom_xmax, self.config.dom_ymax,
                        self.config.dom_inc)

        #print(self.result_tuple[0])
        #print(self.result_tuple[1])
        #print(self.result_tuple[2])

        stop = timeit.default_timer()
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

    def solve_system(self, alpha_l, alpha_t, beta, gamma, ca, n, M):
        A = []
        b = []

        num_elements = len(self.config.elements)
        for i, e_i in enumerate(self.config.elements):
            for (x_cp, y_cp) in e_i.outline:
                row = []
                for j, e_j in enumerate(self.config.elements):
                    #transform (x_cp, y_cp) to e_j's local coords
                    dx = x_cp - e_j.x
                    dy = y_cp - e_j.y
                    eta_j, psi_j = e_j.uv(dx, dy, alpha_l, alpha_t)
                    if e_j.kind == ATElementType.Line and i==j:
                        eta = 0.0
                    else:
                        eta = eta_j
                    psi = psi_j
                    row += self.build_row(e_j, eta, psi, n)
                b.append(self.f_target(x_cp, e_i, ca, gamma, beta))
                A.append(row)

        A = np.array(A)
        b = np.array(b)
        self.coeff = np.linalg.lstsq(A, b, rcond=None)[0]

    def build_row(self, elem, eta, psi, n):
        row = [elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real]
        for j in range(1, n):
            row.append(elem.m.se(j, psi).real * elem.m.Ko(j, eta).real)
            row.append(elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real)
        return row

    def f_target(self, x, elem, ca, gamma, beta):
        Ci = elem.c
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

    def calc_c(self, x, y):
        n = self.config.num_terms
        alpha_l = self.config.alpha_l
        alpha_t = self.config.alpha_t

        total_coeffs = self.coeff
        total_terms = 2 * n - 1
        F = 0

        for idx, elem in enumerate(self.config.elements):
            dx = x - elem.x
            dy = y - elem.y
            eta, psi = elem.uv(dx, dy, alpha_l, alpha_t)
            coeffs = total_coeffs[idx * total_terms:(idx + 1) * total_terms]

            Fi = coeffs[0] * elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real
            for j in range(1, n):
                Fi += coeffs[2 * j - 1] * elem.m.se(j, psi).real * elem.m.Ko(j, eta).real
                Fi += coeffs[2 * j] * elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real

            F += Fi

        total = F * np.exp(self.config.beta * x)
        if total > self.config.ca:
            return (total - self.config.ca) / self.config.gamma
        else:
            return total - self.config.ca

    def conc_array(self, xmin, ymin, xmax, ymax, inc):
        self.xaxis = np.arange(xmin, xmax+inc, inc)
        self.yaxis = np.arange(ymin, ymax+inc, inc)

        xs, ys = np.meshgrid(self.xaxis, self.yaxis)
        tasks = [
            (
                float(x), float(y),
                self.config.elements,
                self.coeff,
                self.config.num_terms,
                self.config.alpha_l,
                self.config.alpha_t,
                self.config.beta,
                self.config.ca,
                self.config.gamma
            )
            for x, y in zip(xs.ravel(), ys.ravel())
        ]

        with ProcessPoolExecutor() as executor:
            flat = list(executor.map(_compute_point, tasks))

        self.result = np.array(flat).reshape(xs.shape)
        self.result_tuple = (self.xaxis, self.yaxis, self.result)

    def generate_filename_suffix(self):
        n_elements = len(self.config.elements)
        num_terms = self.config.num_terms
        inc = self.config.dom_inc

        has_acceptor = any(elem.c < 0 for elem in self.config.elements)
        acceptor_str = "acceptor-yes" if has_acceptor else "acceptor-no"

        return f"{n_elements}-elements_{num_terms}-terms_{acceptor_str}_inc-{inc}"

    def get_next_run_index(self, results_dir):
        if not os.path.exists(results_dir):
            return 1

        existing_files = os.listdir(results_dir)
        max_index = 0

        for filename in existing_files:
            if filename.startswith('run_') and '_' in filename[4:]:
                try:
                    index_str = filename[4:].split('_')[0]
                    index = int(index_str)
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
        domain_width = self.config.dom_xmax - self.config.dom_xmin

        # check if L_max is close to the maximum x-domain using threshold
        x_threshold = 0.95
        max_x_distance = self.L_max - self.config.dom_xmin

        # Check X-direction (most critical for L_max)
        if max_x_distance >= x_threshold * domain_width:
            warning_msg = (f"WARNING: L_max ({self.L_max:.1f} m) is capped by x-domain boundary! "
                           f"L_max extends to {max_x_distance / domain_width * 100:.1f}% of domain width. ")
            print(warning_msg)

    def check_concentration_range(self):
        if len(self.config.elements) == 0:
            return

        # Get min and max concentration values from elements
        element_concentrations = [elem.c for elem in self.config.elements]
        min_expected = -8.0
        if min(element_concentrations) < min_expected:
            min_expected = min(element_concentrations)
        max_expected = max(element_concentrations)

        result_min = np.min(self.result)
        result_max = np.max(self.result)

        # tolerance
        tolerance = 0.01 * max(abs(min_expected), abs(max_expected))

        warnings_issued = []

        #outside of expected range
        outside_range_count = np.sum((self.result > max_expected + tolerance) |
                                     (self.result < min_expected - tolerance))
        total_points = self.result.size

        percentage = (outside_range_count / total_points) * 100

        if percentage > 1.0:
            warning_msg = (f"WARNING: {outside_range_count} points ({percentage:.2f}%) are outside expected range "
                           f"[{min_expected:.3f}, {max_expected:.3f}] mg/l")
            print(warning_msg)
            warnings_issued.append("warning_msg")

        if warnings_issued:
            print(f"\n*** CONCENTRATION RANGE SUMMARY ***")
            print(f"Expected range: [{min_expected:.3f}, {max_expected:.3f}] mg/l")
            print(f"Actual range: [{result_min:.3f}, {result_max:.3f}] mg/l")
            print(f"Element concentrations: {element_concentrations}")
            print(f"This may indicate numerical issues or insufficient domain resolution.\n")
        else:
            print(f"All concentration values within expected range [{min_expected:.3f}, {max_expected:.3f}] mg/l")

    def print_statistics(self, cpu_time):
        from math import cos, sin, pi
    
        r = self.config.elements[0].r
        phi = np.linspace(0, 2 * pi, 360)
        stats = {}
    
        print("\n=== ELEMENT BOUNDARY STATISTICS ===")
        for idx, elem in enumerate(self.config.elements):
            if elem.kind == ATElementType.Circle:
                phi = np.linspace(0, 2 * pi, 360)
                x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
                y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)
    
            elif elem.kind == ATElementType.Line:
                half_len = elem.r / 2
                s = np.linspace(-half_len + 1e-9, half_len - 1e-9, 360)
                theta = elem.theta
                x_test = elem.x + s * cos(theta)
                y_test = elem.y + s * sin(theta)
    
            else:
                raise ValueError(f"Unknown element type: {elem.kind}")
    
            Err = [self.calc_c(x, y) for x, y in zip(x_test, y_test)]
            min_val = round(np.min(Err), 9)
            max_val = round(np.max(Err), 9)
            mean_val = round(np.mean(Err), 9)
            std_val = round(np.std(Err), 9)
    
            print(f'Element {idx + 1}:')
            print(f'  Min = {min_val} mg/l')
            print(f'  Max = {max_val} mg/l')
            print(f'  Mean = {mean_val} mg/l')
            print(f'  Standard Deviation = {std_val} mg/l')
    
            stats[f"Min{idx + 1}"] = min_val
            stats[f"Max{idx + 1}"] = max_val
            stats[f"Mean{idx + 1}"] = mean_val
            stats[f"Std{idx + 1}"] = std_val
    
        # --- save stats to file ---
        results_dir = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)
    
        filename_suffix = self.generate_filename_suffix()
        stats_filename = os.path.join(
            results_dir,
            f"run_{self.run_index:04d}_stats_{filename_suffix}.txt"
        )
    
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
            f.write(
                f"Domain: x[{self.config.dom_xmin}, {self.config.dom_xmax}], "
                f"y[{self.config.dom_ymin}, {self.config.dom_ymax}]\n"
            )
    
            # source geometry
            f.write(f"\n=== ELEMENTS ===\n")
            for idx, elem in enumerate(self.config.elements):
                if elem.kind == ATElementType.Circle:
                    f.write(
                        f"Element {idx + 1}: kind={elem.kind}, x={elem.x}, "
                        f"y={elem.y}, r={elem.r}, c={elem.c}, id={elem.id}\n"
                    )
                else:
                    f.write(
                        f"Element {idx + 1}: kind={elem.kind}, x={elem.x}, "
                        f"y={elem.y}, r={elem.r}, theta={elem.theta}, "
                        f"c={elem.c}, id={elem.id}\n"
                    )
    
            # computation info
            f.write(f"\n=== COMPUTATION INFO ===\n")
            f.write(f"CPU Time [hh:mm:ss]: {cpu_time}\n")
            if self.L_max is not None:
                f.write(f"\nL_max: {self.L_max}\n")
    
            # boundary stats
            f.write(f"\n=== ELEMENT BOUNDARY STATISTICS ===\n")
            for k, v in stats.items():
                f.write(f"{k} = {v} mg/l\n")
    
        print(f"Statistics saved to: {stats_filename}")
    
        # --- PLOT ABSOLUTE ERROR ALONG BOUNDARIES ---
        mpl.rcParams.update({'font.size': 22})
        plt.figure(figsize=(16, 9), dpi=300)
    
        styles = ['-', '--', ':', '-.']
        for idx, elem in enumerate(self.config.elements):
            # generate the same sampling for each element
            if elem.kind == ATElementType.Circle:
                x_test = elem.x + (elem.r + 1e-9) * np.cos(phi)
                y_test = elem.y + (elem.r + 1e-9) * np.sin(phi)
            elif elem.kind == ATElementType.Line:
                half_len = elem.r / 2
                s = np.linspace(-half_len + 1e-9, half_len - 1e-9, 360)
                theta = elem.theta
                x_test = elem.x + s * cos(theta)
                y_test = elem.y + s * sin(theta)
            else:
                continue
    
            # compute absolute error
            Err = [abs(self.calc_c(x, y)) for x, y in zip(x_test, y_test)]
    
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
        inc = self.config.dom_inc

        max_val = np.max(self.result)
        min_val = np.min(self.result)
        abs_min = abs(min_val)
        xmin, xmax = self.config.dom_xmin, self.config.dom_xmax
        ymin, ymax = self.config.dom_ymin, self.config.dom_ymax

        donor_levels = np.linspace(0, max_val, 11)
        acceptor_levels = np.linspace(-abs_min, 0, 9)

        plt.figure(figsize=(16, 9), dpi=300)
        mpl.rcParams.update({"font.size": 22})

        X, Y = np.meshgrid(self.xaxis, self.yaxis)

        donor = plt.contourf(
            X,Y,
            self.result,
            levels=donor_levels,
            cmap='Reds',
            extend='max'
        )
        acceptor = plt.contourf(
            X,Y,
            self.result,
            levels=acceptor_levels,
            cmap='Blues_r',
            extend='min'
        )
        Plume_max = plt.contour(
            X,Y,
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

        paths = Plume_max.get_paths()
        if paths and len(paths[0].vertices) > 0:
            verts = paths[0].vertices
            calculated_lmax = int(np.floor(np.max(verts[:, 0])))
            if self.L_max != calculated_lmax:
                print(f"Warning: L_max mismatch. Previously calculated: {self.L_max}, Current: {calculated_lmax}")
                self.L_max = calculated_lmax
        else:
            if self.L_max is None:
                print("L_max: -")

        if self.config.plot_aspect == "scaled":
            plt.axis("scaled")
        else:  # "auto"
            plt.axis("auto")

        plt.tight_layout()

        results_dir = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)
        filename_suffix = self.generate_filename_suffix()
        plot_filename = os.path.join(
            results_dir,
            f"run_{self.run_index:04d}_plot_{filename_suffix}.pdf"
        )
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")
