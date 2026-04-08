# Written by Willi Kappler, willi.kappler@uni-tuebingen.de
# Based on code from Anton Köhler

# Python std library:
import math
import timeit
from datetime import timedelta
import logging
import json
import os

# External library:
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_temp
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from multiprocessing import Pool, cpu_count
import multiprocessing

# Local imports:
from at_config import ATConfiguration
from at_element import ATElement, ATElementType

logger = logging.getLogger(__name__)

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
        if elem.kind == ATElementType.Ellipse:
            # Point-in-rotated-ellipse: ((x')/a)^2 + ((y')/b)^2 <= 1
            a = float(elem.r)
            b = float(elem.b if elem.b is not None else elem.r)
            cθ, sθ = math.cos(elem.theta), math.sin(elem.theta)
            xp =  cθ * dx + sθ * dy
            yp = -sθ * dx + cθ * dy
            if (xp*xp) / (a*a) + (yp*yp) / (b*b) <= 1.0:
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
    if elem.kind == ATElementType.Line:
        return ATElement(
            kind=elem.kind,
            x=elem.x,
            y=-elem.y,
            c=-elem.c,
            r=elem.r,
            theta=-elem.theta
        )
    elif elem.kind == ATElementType.Ellipse:
        return ATElement(
            kind=elem.kind,
            x=elem.x,
            y=-elem.y,
            c=-elem.c,
            r=elem.r,
            theta=-elem.theta,
            b=elem.b
        )
    else:  # Circle
        return ATElement(
            kind=elem.kind,
            x=elem.x,
            y=-elem.y,
            c=-elem.c,
            r=elem.r,
            theta=math.pi / 2
        )

def create_zero_isoline(elem: ATElement) -> ATElement:
    """
    Create a horizontal (theta=0) line element on the isoline (y=0) that enforces
    C = 0 between x in [elem.x - elem.r, elem.x + elem.r].

    Notes
    -----
    - For line sources, `r` is the half-length, so this has the same length.
    - For circles, we use the circle radius as half-length, so the isoline span is 2r.
    """
    return ATElement(
        kind=ATElementType.Line,
        x=elem.x,
        y=0.0,
        c=0.0,          # enforce zero concentration on the isoline segment
        r=elem.r,       # same half-length/radius as the input element
        theta=0.0       # horizontal along the isoline y=0
    )



class ATSimulation:
    def __init__(self, config: ATConfiguration):
        self.config: ATConfiguration = config
        self.coeff: np.ndarray = np.zeros((10, 10))
        self.xaxis: np.ndarray = np.arange(0, 10, 1)
        self.yaxis: np.ndarray = np.arange(0, 10, 1)
        self.result: np.ndarray = np.zeros((10, 10))
        self.L_max = None
        self.interface_length = None

    def run(self):
        if len(self.config.elements) < 1:
            raise ValueError("Simulation requires at least one element.")

        if self.config.orientation == "vertical":
            updated_elements = []
            for elem in self.config.elements:
                if elem.y > -(elem.r*np.sin(elem.theta)+0.1):
                    raise ValueError(f"Element '{elem.id}' must have y < -(r+0.1) for vertical orientation.")
                updated_elements.append(elem)
                mirrored = create_mirrored_element(elem)
                mirrored.id = f"image_{elem.id}"
                zero_isoline = create_zero_isoline(elem)
                zero_isoline.id = f"zero_{elem.id}"
                updated_elements.append(mirrored)
            updated_elements.append(zero_isoline)
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

        # Create a dedicated subdirectory for this run: Results/0001/
        self.run_dir = os.path.join(results_dir, f"{self.run_index:04d}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.plot_input()
        self.print_statistics(cpu_time)
        self.plot_result()
        #self.plot_quiver()

    @staticmethod
    def _ke_log_magnitude(q: float, eta: float) -> float:
        """
        Estimate log10 of |Ke_0(eta)| for the given q.

        The leading-order behavior of Ke_0 is dominated by the
        K-Bessel factor:
            Ke_0(eta) ~ exp(-sqrt(q) * exp(eta))

        This returns log10 of that estimate, i.e.:
            -sqrt(q) * exp(eta) / ln(10)

        A value of -16 means Ke is at double-precision machine epsilon.
        A value of -30 means it is far below.

        Parameters
        ----------
        q : float
            Mathieu parameter (positive).
        eta : float
            Radial elliptic coordinate.

        Returns
        -------
        float
            Approximate log10(|Ke_0(eta, q)|).
        """
        if q <= 0.0 or eta <= 0.0:
            return 0.0  # near the source, Ke is O(1)
        return -math.sqrt(q) * math.exp(eta) / math.log(10.0)

    def _are_elements_coupled(self, e_i: ATElement, e_j: ATElement,
                              alpha_l: float, alpha_t: float,
                              log10_floor: float = -14.0) -> bool:
        """
        Determine whether two elements are close enough that cross-terms
        in the coefficient matrix are numerically significant.

        Evaluates the actual estimated magnitude of Ke_0(eta, q) for each
        element's q value. If Ke is below 10^(log10_floor) in both directions,
        the cross-terms are negligible and the elements should be
        decoupled.

        The q-dependent criterion is essential because the decay rate
        of Ke_n(eta) depends on sqrt(q):
            Ke_0(eta) ~ exp(-sqrt(q) * exp(eta))
        For small q (e.g., q=0.14), Ke reaches machine epsilon at
        eta ~ 4.6, while for large q (e.g., q=100), this happens
        at eta ~ 1.3.

        Parameters
        ----------
        e_i, e_j : ATElement
            The two elements to test.
        alpha_l, alpha_t : float
            Dispersivities (needed for the coordinate transform).
        log10_floor : float
            Decoupling threshold as log10(|Ke|). Default -14.0 means
            elements are decoupled when Ke < 1e-14 (~ 100x above
            machine epsilon, providing a safety margin).

        Returns
        -------
        bool
            True if elements should be solved together (coupled),
            False if they should be solved independently.
        """
        if e_i is e_j:
            return True

        # eta from center of e_i in e_j's coordinate system
        dx_ij = e_i.x - e_j.x
        dy_ij = e_i.y - e_j.y
        eta_ij, _ = e_j.uv(dx_ij, dy_ij, alpha_l, alpha_t)
        # Ke magnitude when evaluating e_j's basis functions at e_i's location
        log_ke_j = self._ke_log_magnitude(e_j.q, eta_ij)

        # eta from center of e_j in e_i's coordinate system
        dx_ji = e_j.x - e_i.x
        dy_ji = e_j.y - e_i.y
        eta_ji, _ = e_i.uv(dx_ji, dy_ji, alpha_l, alpha_t)
        # Ke magnitude when evaluating e_i's basis functions at e_j's location
        log_ke_i = self._ke_log_magnitude(e_i.q, eta_ji)

        # Elements are coupled if EITHER direction has significant cross-terms
        coupled = max(log_ke_i, log_ke_j) > log10_floor

        logger.info(f"  Coupling test: elem({e_i.x},{e_i.y}) <-> elem({e_j.x},{e_j.y}): "
                     f"eta_ij={eta_ij:.2f}, eta_ji={eta_ji:.2f}, "
                     f"log10|Ke_j|={log_ke_j:.1f}, log10|Ke_i|={log_ke_i:.1f}, "
                     f"q_i={e_i.q:.4f}, q_j={e_j.q:.4f} -> "
                     f"{'COUPLED' if coupled else 'DECOUPLED'}")

        return coupled

    def _find_coupling_groups(self, elements: list, alpha_l: float,
                              alpha_t: float) -> list:
        """
        Partition elements into coupling groups using a union-find approach.

        Elements within the same group are close enough to influence each
        other's coefficient solve and must be solved together (coupled).
        Elements in different groups are too far apart for cross-terms to
        matter and are solved independently.

        The coupling criterion is q-dependent: it uses the estimated
        magnitude of Ke_0(eta, q) rather than a fixed eta threshold.

        Parameters
        ----------
        elements : list of ATElement
        alpha_l, alpha_t : float

        Returns
        -------
        list of list of int
            Each inner list contains the indices of elements in one group.
        """
        ne = len(elements)
        parent = list(range(ne))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(ne):
            for j in range(i + 1, ne):
                if self._are_elements_coupled(elements[i], elements[j],
                                              alpha_l, alpha_t):
                    union(i, j)

        groups = {}
        for i in range(ne):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return list(groups.values())

    def _build_group_system(self, group_indices: list, elements: list,
                            alpha_l: float, alpha_t: float, beta: float,
                            ca: float, gamma: float, n: int,
                            cross_term_floor: float = 1e-250) -> tuple:
        """
        Build the coefficient matrix A and RHS vector b for a group
        of coupled elements.

        Cross-element entries where all Ke/Ko values are below
        cross_term_floor are zeroed out to prevent near-zero entries
        from polluting the matrix.

        Parameters
        ----------
        group_indices : list of int
            Indices into the elements list for this coupling group.
        elements : list of ATElement
            Full element list.
        alpha_l, alpha_t, beta, ca, gamma : float
            Physical parameters.
        n : int
            Number of Mathieu function terms.
        cross_term_floor : float
            Entries smaller than this are zeroed. Default 1e-250 is
            generous; anything below ~1e-300 is noise.

        Returns
        -------
        A : np.ndarray, shape (num_rows, num_cols)
        b : np.ndarray, shape (num_rows,)
        """
        group_elements = [elements[k] for k in group_indices]
        ng = len(group_elements)
        total_terms = 2 * n - 1

        A_rows = []
        b_vec = []

        for i_local, e_i in enumerate(group_elements):
            for (x_cp, y_cp) in e_i.outline:
                row = []
                for j_local, e_j in enumerate(group_elements):
                    dx = x_cp - e_j.x
                    dy = y_cp - e_j.y
                    eta_j, psi_j = e_j.uv(dx, dy, alpha_l, alpha_t)

                    if e_j.kind == ATElementType.Line and i_local == j_local:
                        eta = 0.0
                    else:
                        eta = eta_j
                    psi = psi_j

                    block = self.build_row(e_j, eta, psi, n)

                    # Zero out cross-element entries that are below machine noise
                    if i_local != j_local:
                        max_abs = max(abs(v) for v in block)
                        if max_abs < cross_term_floor:
                            block = [0.0] * len(block)

                    row += block

                b_vec.append(self.f_target(x_cp, e_i, ca, gamma, beta))
                A_rows.append(row)

        return np.array(A_rows), np.array(b_vec)

    def _solve_scaled_lstsq(self, A: np.ndarray, b: np.ndarray,
                            rcond: float = 1e-10) -> tuple:
        """
        Solve the least-squares system with row scaling for numerical stability.

        Returns
        -------
        (coeffs, cond_number)
            coeffs: np.ndarray, shape (n,)
            cond_number: float, condition number of the scaled matrix
        """
        row_scales = np.max(np.abs(A), axis=1)
        row_scales = np.where(row_scales < 1e-300, 1.0, row_scales)

        A_scaled = A / row_scales[:, None]
        b_scaled = b / row_scales

        coeffs, residuals, rank, sv = np.linalg.lstsq(A_scaled, b_scaled, rcond=rcond)

        cond = np.inf
        if sv is not None and len(sv) > 0:
            cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
            logger.info(f"  lstsq: rank={rank}, cond={cond:.2e}, "
                        f"sv_max={sv[0]:.2e}, sv_min={sv[-1]:.2e}")

        return coeffs, cond

    def _evaluate_group_field(self, group_indices: list, elements: list,
                              coeff: np.ndarray, n: int,
                              x: float, y: float,
                              alpha_l: float, alpha_t: float) -> float:
        """
        Evaluate the Mathieu expansion field F (before the exp(beta*x)
        multiplication) contributed by a specific group of elements at
        a single point.

        Parameters
        ----------
        group_indices : list of int
            Which elements to include.
        elements : list of ATElement
            Full element list.
        coeff : np.ndarray
            Full coefficient vector (all elements).
        n : int
            Number of Mathieu terms.
        x, y : float
            Evaluation point.
        alpha_l, alpha_t : float
            Dispersivities.

        Returns
        -------
        float
            F contribution from the specified group at (x, y).
        """
        total_terms = 2 * n - 1
        F = 0.0
        for idx in group_indices:
            elem = elements[idx]
            dx = x - elem.x
            dy = y - elem.y
            eta, psi = elem.uv(dx, dy, alpha_l, alpha_t)

            block = coeff[idx * total_terms: (idx + 1) * total_terms]

            Fi = block[0] * elem.m.ce(0, psi).real * elem.m.Ke(0, eta).real
            for j in range(1, n):
                Fi += block[2*j - 1] * elem.m.se(j, psi).real * elem.m.Ko(j, eta).real
                Fi += block[2*j    ] * elem.m.ce(j, psi).real * elem.m.Ke(j, eta).real
            F += Fi
        return F

    def _validate_boundary_accuracy(self, elements: list, coeff: np.ndarray,
                                    n: int, alpha_l: float, alpha_t: float,
                                    beta: float, ca: float, gamma: float,
                                    max_rel_error: float = 0.05) -> tuple:
        """
        Check whether the current coefficients satisfy the boundary conditions
        to within an acceptable tolerance.

        Evaluates the concentration at a sample of control points on each
        element's boundary and compares against the prescribed target.

        Parameters
        ----------
        elements : list of ATElement
        coeff : np.ndarray
            Full coefficient vector.
        n : int
            Number of Mathieu terms.
        alpha_l, alpha_t, beta, ca, gamma : float
            Physical parameters.
        max_rel_error : float
            Maximum acceptable relative error on any element boundary.
            Default 0.05 (5%).

        Returns
        -------
        (is_valid, worst_elem_idx, worst_rel_error)
            is_valid: True if all boundaries meet tolerance.
            worst_elem_idx: index of element with largest error.
            worst_rel_error: the largest relative error found.
        """
        total_terms = 2 * n - 1
        worst_rel_error = 0.0
        worst_elem_idx = -1

        for idx, elem in enumerate(elements):
            target_conc = elem.c

            # Sample a subset of control points (every 10th for speed)
            outline = elem.outline
            step = max(1, len(outline) // 10)
            sample_points = outline[::step]

            errors = []
            for (x_cp, y_cp) in sample_points:
                # Evaluate the full field at this control point
                F = 0.0
                for eidx, e in enumerate(elements):
                    dx = x_cp - e.x
                    dy = y_cp - e.y
                    eta, psi = e.uv(dx, dy, alpha_l, alpha_t)
                    block = coeff[eidx * total_terms: (eidx + 1) * total_terms]

                    Fi = block[0] * e.m.ce(0, psi).real * e.m.Ke(0, eta).real
                    for j in range(1, n):
                        Fi += block[2*j - 1] * e.m.se(j, psi).real * e.m.Ko(j, eta).real
                        Fi += block[2*j    ] * e.m.ce(j, psi).real * e.m.Ke(j, eta).real
                    F += Fi

                total = F * np.exp(beta * x_cp)
                if total > ca:
                    conc = (total - ca) / gamma
                else:
                    conc = total - ca

                errors.append(abs(conc - target_conc))

            mean_error = np.mean(errors)
            denom = max(abs(target_conc), 1e-10)
            rel_error = mean_error / denom

            if rel_error > worst_rel_error:
                worst_rel_error = rel_error
                worst_elem_idx = idx

        is_valid = worst_rel_error < max_rel_error
        return is_valid, worst_elem_idx, worst_rel_error

    def _solve_coupled(self, elements: list, n: int, total_terms: int,
                       alpha_l: float, alpha_t: float, beta: float,
                       ca: float, gamma: float) -> np.ndarray:
        """
        Solve all elements in a single coupled system (original approach
        with row scaling).

        Returns
        -------
        np.ndarray
            Coefficient vector for all elements.
        """
        all_indices = list(range(len(elements)))
        A, b = self._build_group_system(
            all_indices, elements, alpha_l, alpha_t, beta, ca, gamma, n
        )
        coeffs, _ = self._solve_scaled_lstsq(A, b)
        coeff = np.zeros(len(elements) * total_terms)
        for local_idx, global_idx in enumerate(all_indices):
            start = global_idx * total_terms
            local_start = local_idx * total_terms
            coeff[start:start+total_terms] = coeffs[local_start:local_start+total_terms]
        return coeff

    def _solve_decoupled_iterative(self, elements: list, groups: list,
                                   n: int, total_terms: int,
                                   alpha_l: float, alpha_t: float,
                                   beta: float, ca: float, gamma: float) -> np.ndarray:
        """
        Solve element groups independently with iterative cross-group
        correction (Gauss-Seidel style).

        Returns
        -------
        np.ndarray
            Coefficient vector for all elements.
        """
        num_elements = len(elements)

        # Pre-build each group's A matrix and original RHS
        group_data = []
        for group in groups:
            A, b_original = self._build_group_system(
                group, elements, alpha_l, alpha_t, beta, ca, gamma, n
            )
            control_points = []
            for g_idx in group:
                for (x_cp, y_cp) in elements[g_idx].outline:
                    control_points.append((x_cp, y_cp))

            group_data.append({
                'indices': group,
                'A': A,
                'b_original': b_original,
                'control_points': control_points,
            })

        coeff = np.zeros(num_elements * total_terms)

        # Iterative solve
        max_iter = 8
        tol = 1e-6

        for iteration in range(max_iter):
            coeff_old = coeff.copy()

            for gi, gd in enumerate(group_data):
                group = gd['indices']
                b_corrected = gd['b_original'].copy()

                # Subtract cross-group contributions
                for gj, gd_other in enumerate(group_data):
                    if gi == gj:
                        continue
                    other_group = gd_other['indices']
                    for cp_idx, (x_cp, y_cp) in enumerate(gd['control_points']):
                        F_other = self._evaluate_group_field(
                            other_group, elements, coeff, n,
                            x_cp, y_cp, alpha_l, alpha_t
                        )
                        b_corrected[cp_idx] -= F_other

                coeffs, _ = self._solve_scaled_lstsq(gd['A'], b_corrected)

                for local_idx, global_idx in enumerate(group):
                    start = global_idx * total_terms
                    local_start = local_idx * total_terms
                    coeff[start:start+total_terms] = coeffs[local_start:local_start+total_terms]

            delta = np.linalg.norm(coeff - coeff_old)
            norm = np.linalg.norm(coeff)
            rel_change = delta / norm if norm > 0 else delta

            logger.info(f"  Iteration {iteration+1}: rel_change={rel_change:.2e}")

            if rel_change < tol:
                print(f"Decoupled solve converged after {iteration+1} iterations "
                      f"(rel_change={rel_change:.2e})")
                return coeff

        print(f"Decoupled solve: {max_iter} iterations (rel_change={rel_change:.2e})")
        return coeff

    @staticmethod
    def _find_atomic_units(elements: list) -> tuple:
        """
        Identify groups of elements that must always be solved together,
        and shared elements that must be included in every group.

        In vertical orientation, each real source has a paired image element.
        These two must never be separated. A single zero-isoline at the end
        of the element list enforces the surface boundary condition and must
        be included in every solve group.

        In horizontal orientation, each element is independent with no
        shared elements.

        Returns
        -------
        (units, shared)
            units: list of list of int — each inner list is an atomic unit
            shared: list of int — indices that must be included in every group
        """
        units = []
        shared = []
        i = 0
        while i < len(elements):
            eid = elements[i].id.lower() if elements[i].id else ""

            if eid.startswith("zero_"):
                # Zero-isoline is shared across all groups
                shared.append(i)
                i += 1
            elif (not eid.startswith("image_")
                  and i + 1 < len(elements)
                  and (elements[i+1].id or "").lower().startswith("image_")):
                # Source-image pair
                units.append([i, i+1])
                i += 2
            else:
                # Standalone element
                units.append([i])
                i += 1

        return units, shared

    def solve_system(self, alpha_l: float, alpha_t: float, beta: float,
            gamma: float, ca: float, n: int, M: int):
        """
        Solve for Mathieu function expansion coefficients.

        Strategy:
        1. Try solving all elements in one coupled matrix using the
           standard lstsq (same as the original code).
        2. Check the condition number.
        3. If well-conditioned (cond < 10^14), accept the result.
        4. If ill-conditioned, split into groups and solve iteratively
           (using row-scaled lstsq for the sub-group solves).
        """
        elements = self.config.elements
        num_elements = len(elements)
        total_terms = 2 * n - 1
        cond_threshold = 1e14

        # --- Step 1: Try coupled solve (same as OG code) ---
        all_indices = list(range(num_elements))
        A, b = self._build_group_system(
            all_indices, elements, alpha_l, alpha_t, beta, ca, gamma, n
        )

        # Use rcond=None (OG behavior) and no row scaling
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

        cond = np.inf
        if sv is not None and len(sv) > 0:
            cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf

        if cond < cond_threshold:
            # Well-conditioned: accept the coupled solve
            self.coeff = np.zeros(num_elements * total_terms)
            for li, gi in enumerate(all_indices):
                self.coeff[gi*total_terms:(gi+1)*total_terms] = \
                    coeffs[li*total_terms:(li+1)*total_terms]
            print(f"Coupled solve: OK (cond={cond:.1e})")
            return

        # --- Step 2: Ill-conditioned — split and solve iteratively ---
        print(f"Coupled solve ill-conditioned (cond={cond:.1e}) "
              f"— switching to decoupled solve")

        units, shared = self._find_atomic_units(elements)
        groups = self._split_until_valid(
            all_indices, elements, units, shared, n, total_terms,
            alpha_l, alpha_t, beta, ca, gamma,
            cond_threshold
        )

        self.coeff = np.zeros(num_elements * total_terms)

        if len(groups) == 1:
            # Split didn't help — store the coupled result anyway
            for li, gi in enumerate(all_indices):
                self.coeff[gi*total_terms:(gi+1)*total_terms] = \
                    coeffs[li*total_terms:(li+1)*total_terms]
            print(f"Could not split further — using coupled result")
            return

        print(f"Split into {len(groups)} groups: "
              f"{[sorted(g) for g in groups]}")

        self.coeff = self._solve_decoupled_iterative(
            elements, groups, n, total_terms,
            alpha_l, alpha_t, beta, ca, gamma
        )

    def _split_until_valid(self, indices: list, elements: list,
                           units: list, shared: list,
                           n: int, total_terms: int,
                           alpha_l: float, alpha_t: float,
                           beta: float, ca: float, gamma: float,
                           cond_threshold: float) -> list:
        """
        Recursively split a group of element indices until each sub-group
        has a well-conditioned solve (condition number < threshold).

        Splitting respects atomic units and shared elements.
        """
        index_set = set(indices)
        contained_units = [u for u in units
                           if all(i in index_set for i in u)]

        # Base case: single atomic unit — cannot split further
        if len(contained_units) <= 1:
            return [indices]

        # Try solving this group
        A, b = self._build_group_system(
            indices, elements, alpha_l, alpha_t, beta, ca, gamma, n
        )
        coeffs, cond = self._solve_scaled_lstsq(A, b)

        if cond < cond_threshold:
            return [indices]

        # Ill-conditioned — split at largest gap
        sorted_units = sorted(contained_units,
                              key=lambda u: (elements[u[0]].x,
                                             elements[u[0]].y))

        max_gap = -1
        split_pos = 1
        for k in range(1, len(sorted_units)):
            prev_src = elements[sorted_units[k-1][0]]
            curr_src = elements[sorted_units[k][0]]
            gap = math.sqrt((curr_src.x - prev_src.x)**2 +
                            (curr_src.y - prev_src.y)**2)
            if gap > max_gap:
                max_gap = gap
                split_pos = k

        left_units = sorted_units[:split_pos]
        right_units = sorted_units[split_pos:]

        left = [i for u in left_units for i in u] + shared
        right = [i for u in right_units for i in u] + shared

        # Deduplicate (shared might already be in units)
        left = sorted(set(left))
        right = sorted(set(right))

        logger.info(f"  Splitting {sorted(indices)} -> "
                    f"{left} + {right} (cond={cond:.1e}, gap={max_gap:.1f})")

        left_groups = self._split_until_valid(
            left, elements, units, shared, n, total_terms,
            alpha_l, alpha_t, beta, ca, gamma, cond_threshold
        )
        right_groups = self._split_until_valid(
            right, elements, units, shared, n, total_terms,
            alpha_l, alpha_t, beta, ca, gamma, cond_threshold
        )

        return left_groups + right_groups

    def _validate_group_boundary(self, group_indices: list, elements: list,
                                 coeff: np.ndarray, n: int,
                                 alpha_l: float, alpha_t: float,
                                 beta: float, ca: float, gamma: float,
                                 max_rel_error: float = 0.05) -> bool:
        """
        Check whether the current coefficients produce accurate boundary
        values for the elements in group_indices.

        Compares in F-space (the transformed variable that the matrix
        equation actually solves for) to correctly handle all element
        types: sources, images, and zero-isolines.
        """
        total_terms = 2 * n - 1

        for idx in group_indices:
            elem = elements[idx]
            outline = elem.outline
            step = max(1, len(outline) // 10)

            errors = []
            for (x_cp, y_cp) in outline[::step]:
                # Evaluate F from THIS GROUP's elements only
                F = 0.0
                for eidx in group_indices:
                    e = elements[eidx]
                    dx, dy = x_cp - e.x, y_cp - e.y
                    eta, psi = e.uv(dx, dy, alpha_l, alpha_t)
                    block = coeff[eidx*total_terms:(eidx+1)*total_terms]

                    Fi = block[0] * e.m.ce(0, psi).real * e.m.Ke(0, eta).real
                    for j in range(1, n):
                        Fi += block[2*j-1] * e.m.se(j, psi).real * e.m.Ko(j, eta).real
                        Fi += block[2*j  ] * e.m.ce(j, psi).real * e.m.Ke(j, eta).real
                    F += Fi

                # Compare in F-space against the actual boundary condition
                f_tgt = self.f_target(x_cp, elem, ca, gamma, beta)
                if f_tgt is not None:
                    errors.append(abs(F - f_tgt))

            if not errors:
                continue

            mean_err = np.mean(errors)
            # Normalise by the target magnitude
            f_tgt_sample = self.f_target(elem.x, elem, ca, gamma, beta)
            denom = max(abs(f_tgt_sample), 1e-30) if f_tgt_sample is not None else 1e-30
            rel_err = mean_err / denom
            if rel_err > max_rel_error:
                return False

        return True

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

        if Ci == 0 and not is_image:
            return ca * np.exp(-beta * x)

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

        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=cpu_count(), initializer=_init_pool, initargs=pool_args) as pool:
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

        max_index: int = 0
        for entry in os.listdir(results_dir):
            # Each run lives in a zero-padded 4-digit subdirectory, e.g. "0001"
            if os.path.isdir(os.path.join(results_dir, entry)):
                try:
                    index: int = int(entry)
                    max_index = max(max_index, index)
                except ValueError:
                    continue

        return max_index + 1

    def calculate_lmax(self):
        """
        Compute L_max (furthest x-intersection of the 0-contour) and the interface length
        (total polyline length of the chosen 0-contour) in meters.
        """
        fig_temp = plt_temp.figure()
        # Contour in index space; vertices are in pixel units, scale later by dom_inc
        contour_temp = plt_temp.contour(self.result, levels=[0])

        paths = contour_temp.get_paths()
        chosen = None
        self.interface_length = None
        self.L_max = None

        if paths:
            # Choose the path whose max-x (in index space) is largest → defines L_max
            max_x_overall = -np.inf
            for p in paths:
                v = p.vertices
                if v is None or len(v) == 0:
                    continue
                max_x_path = np.max(v[:, 0])
                if max_x_path > max_x_overall:
                    max_x_overall = max_x_path
                    chosen = v

            if chosen is not None and len(chosen) > 1:
                # L_max from furthest x (index units → meters)
                self.L_max = int(max_x_overall * self.config.dom_inc)

                # Interface length: sum of segment lengths along the chosen contour (index units → meters)
                diffs = np.diff(chosen, axis=0)
                seg_len_idx = np.hypot(diffs[:, 0], diffs[:, 1])
                self.interface_length = float(np.sum(seg_len_idx) * self.config.dom_inc)

        plt_temp.close(fig_temp)

        if self.L_max is not None:
            print('Lmax =', self.L_max)
            if self.interface_length is not None:
                print(f'Interface length = {self.interface_length:.3f} m')
            else:
                print('Interface length: -')
        else:
            print("L_max: -")
            print("Interface length: -")

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

            elif elem.kind == ATElementType.Ellipse:
                t = np.linspace(0, 2 * pi, 360, endpoint=False)
                a = float(elem.r)
                b = float(elem.b if elem.b is not None else elem.r)
                cθ, sθ = math.cos(elem.theta), math.sin(elem.theta)
                x_test = elem.x + a * np.cos(t) * cθ - b * np.sin(t) * sθ
                y_test = elem.y + a * np.cos(t) * sθ + b * np.sin(t) * cθ
                phi = t  # reuse t as angular parameter for plotting

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

        filename_suffix: str = self.generate_filename_suffix()
        stats_filename: str = os.path.join(
            self.run_dir, f"{self.run_index:04d}_stats_{filename_suffix}.txt"
        )

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
                elif (elem.kind == ATElementType.Ellipse):
                    f.write(f"Element {idx + 1}:kind={elem.kind}, x={elem.x}, y={elem.y}, a={elem.r}, b={elem.b}, theta={elem.theta}, c={elem.c}, id={elem.id}\n")
                else:
                    f.write(f"Element {idx + 1}:kind={elem.kind}, x={elem.x}, y={elem.y}, r={elem.r}, theta={elem.theta}, c={elem.c}, id={elem.id}\n")

            f.write("\n=== COMPUTATION INFO ===\n")
            f.write(f"CPU Time [hh:mm:ss]: {cpu_time}\n")

            if self.L_max is not None:
                f.write(f"\nL_max: {self.L_max}\n")
            if getattr(self, "interface_length", None) is not None:
                f.write(f"InterfaceLength: {self.interface_length:.6f} m\n")

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
            self.run_dir,
            f"{self.run_index:04d}_error_{filename_suffix}.pdf"
        )
        plt.savefig(error_filename)
        print(f"Error plot saved to: {error_filename}")

    def plot_input(self):
        """
        Generate an input plot of the source element layout, cropped to the
        source zone, and save it to the run subdirectory.
        """
        import matplotlib.patches as mpatches

        # Use only the original (non-image, non-zero) elements for the plot
        elements = [e for e in self.config.elements
                    if not e.id.lower().startswith("image_")
                    and not e.id.lower().startswith("zero_")]

        if not elements:
            return

        concs = [e.c for e in elements]
        max_conc = max(concs)
        min_conc = min(concs)
        has_donor = max_conc > 0
        has_acceptor = min_conc < 0
        ca = self.config.ca

        donor_cmap = plt.cm.Reds
        acceptor_cmap = plt.cm.Blues_r
        donor_norm = mpl.colors.Normalize(vmin=0, vmax=max_conc if has_donor else 1)
        acc_norm = mpl.colors.Normalize(vmin=min_conc if has_acceptor else -ca, vmax=0)

        # Tight bounding box around elements with padding
        all_x = [e.x for e in elements]
        all_y = [e.y for e in elements]
        all_r = [e.r for e in elements]
        r_max = max(all_r)
        span_x = max(max(all_x) - min(all_x), r_max * 2)
        span_y = max(max(all_y) - min(all_y), r_max * 2)
        pad = max(span_x, span_y) * 0.35
        xmin = min(all_x) - r_max - pad
        xmax = max(all_x) + r_max + pad
        ymin = min(all_y) - r_max - pad
        ymax = max(all_y) + r_max + pad

        mpl.rcParams.update({"font.size": 22})
        fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
        ax.set_facecolor(acceptor_cmap(1.0))

        for e in elements:
            c_val = e.c
            if c_val >= 0:
                color = donor_cmap(donor_norm(c_val)) if has_donor else donor_cmap(0.5)
            else:
                color = acceptor_cmap(acc_norm(c_val)) if has_acceptor else acceptor_cmap(0.5)

            if e.kind == ATElementType.Circle:
                ax.add_patch(mpatches.Circle(
                    (e.x, e.y), e.r,
                    facecolor=color, edgecolor="black", linewidth=1.2, zorder=3
                ))
            elif e.kind == ATElementType.Ellipse:
                ax.add_patch(mpatches.Ellipse(
                    (e.x, e.y),
                    width=2 * e.r,
                    height=2 * (e.b if e.b is not None else e.r),
                    angle=math.degrees(e.theta),
                    facecolor=color, edgecolor="black", linewidth=1.2, zorder=3
                ))
            elif e.kind == ATElementType.Line:
                hl = e.r
                x1 = e.x - hl * math.cos(e.theta)
                y1 = e.y - hl * math.sin(e.theta)
                x2 = e.x + hl * math.cos(e.theta)
                y2 = e.y + hl * math.sin(e.theta)
                ax.plot([x1, x2], [y1, y2], color="black",
                        linewidth=5, solid_capstyle="round", zorder=2)
                ax.plot([x1, x2], [y1, y2], color=color,
                        linewidth=4, solid_capstyle="round", zorder=3)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        from matplotlib.ticker import MaxNLocator as _MNL
        ax.xaxis.set_major_locator(_MNL(nbins=5))
        ax.yaxis.set_major_locator(_MNL(nbins=5))
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$z$ (m)" if self.config.orientation == "vertical" else "$y$ (m)")

        plt.subplots_adjust(bottom=0.20)

        if has_donor:
            donor_levels = np.linspace(0, max_conc, 11)
            sm = plt.cm.ScalarMappable(cmap="Reds", norm=mpl.colors.Normalize(0, max_conc))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, ticks=donor_levels,
                                label="Electron donor concentration [mg/l]",
                                location="bottom", pad=0.02, aspect=75)
            cbar.ax.tick_params(labelsize=14)

        abs_min = abs(min_conc) if has_acceptor else ca
        acc_levels = np.linspace(-abs_min, 0, 9)
        sm_acc = plt.cm.ScalarMappable(cmap="Blues_r", norm=mpl.colors.Normalize(-abs_min, 0))
        sm_acc.set_array([])
        cbar_a = plt.colorbar(sm_acc, ax=ax, ticks=acc_levels,
                              label="Electron acceptor concentration [mg/l]",
                              location="bottom", pad=0.12 if has_donor else 0.02, aspect=75)
        cbar_a.set_ticklabels([f"{abs(t):.0f}" for t in acc_levels])
        cbar_a.ax.tick_params(labelsize=14)

        plt.tight_layout()

        filename_suffix: str = self.generate_filename_suffix()
        input_filename: str = os.path.join(
            self.run_dir,
            f"{self.run_index:04d}_input_{filename_suffix}.pdf"
        )
        plt.savefig(input_filename)
        plt.close()
        print(f"Input plot saved to: {input_filename}")

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

        #L_max using Matplotlib (non-deprecated)
        total_length = 0
        for path in Plume_max.get_paths():
            v = path.vertices
            x, y = v[:, 0], v[:, 1]
            segment_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            total_length += np.sum(segment_lengths)
        print(f"Total interface length: {total_length:.4f}")

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))  # ~ticks for X
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

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
            self.run_dir,
            f"{self.run_index:04d}_plot_{filename_suffix}.pdf"
        )
        plt.savefig(plot_filename)
        print(f"Plot saved to: {plot_filename}")


    def plot_quiver(self):
        """
        Show donor and acceptor flux-like fields overlaid on the
        donor/acceptor concentration background.

        Interpretation:
          - self.result is the net concentration field:
              C > 0 → donor-dominated region
              C < 0 → acceptor-dominated region
          - We construct two proxies:
              C_D(x, y) = max(C, 0)      (donor)
              C_A(x, y) = max(-C, 0)     (acceptor "deficit" / magnitude below zero)
          - From these we compute flux-like fields:
              J_D ∝ -∇C_D  (donor flux)
              J_A ∝ -∇C_A  (acceptor/oxygen flux)

        Visual encoding:
          - Background: donor (red) and acceptor (blue) concentration contours
                        as in plot_result (semi-transparent).
          - Red quiver:    donor flux ~ -∇C_D, continuous red gradient by |∇C_D|.
          - Blue quiver:   acceptor flux ~ -∇C_A, continuous blue gradient by |∇C_A|.
          - Arrow length ∝ |∇C| for each field separately (each rescaled so its
            own largest arrow has reasonable size).

        This highlights how donor moves down its gradient and how acceptor
        (oxygen) tends to move towards the reaction zone from the surrounding aquifer.
        """
        plt.figure(figsize=(16, 9), dpi=300)
        mpl.rcParams.update({"font.size": 22})

        # --- Background: donor/acceptor contours (same as plot_result, but with alpha) ---
        max_val: float = np.max(self.result)
        min_val: float = np.min(self.result)
        abs_min: float = abs(min_val)

        donor_levels = np.linspace(0, max_val, 11) if max_val > 0 else [0.0]
        acceptor_levels = np.linspace(-abs_min, 0, 9) if abs_min > 0 else [0.0]

        X, Y = np.meshgrid(self.xaxis, self.yaxis)

        if max_val > 0:
            plt.contourf(
                X, Y,
                self.result,
                levels=donor_levels,
                cmap="Reds",
                extend="max",
                alpha=0.5,
            )

        if abs_min > 0:
            plt.contourf(
                X, Y,
                self.result,
                levels=acceptor_levels,
                cmap="Blues_r",
                extend="min",
                alpha=0.5,
            )

        # 0-contour (interface)
        plt.contour(
            X, Y,
            self.result,
            levels=[0],
            linewidths=2,
            colors="k",
        )

        # --- Donor and acceptor proxies ---
        Cnet = self.result
        C_donor = np.where(Cnet > 0.0, Cnet, 0.0)
        C_acceptor = np.where(Cnet < 0.0, -Cnet, 0.0)  # magnitude below zero

        # Gradients for donor and acceptor; -∇C points "downhill"
        gy_d, gx_d = np.gradient(C_donor, self.config.dom_inc, self.config.dom_inc)
        gy_a, gx_a = np.gradient(C_acceptor, self.config.dom_inc, self.config.dom_inc)
        U_d, V_d = -gx_d, -gy_d
        U_a, V_a = -gx_a, -gy_a

        # Downsample to keep quiver readable (spacing ~1.5 m)
        step = max(1, int(round(1.5 / max(self.config.dom_inc, 1e-9))))
        Xq = X[::step, ::step]
        Yq = Y[::step, ::step]
        C_dq = C_donor[::step, ::step]
        C_aq = C_acceptor[::step, ::step]
        U_dq = U_d[::step, ::step]
        V_dq = V_d[::step, ::step]
        U_aq = U_a[::step, ::step]
        V_aq = V_a[::step, ::step]

        # Masks: only draw arrows where there is a meaningful donor/acceptor presence
        donor_eps = 1e-6 * max(float(np.nanmax(C_donor)), 1.0)
        acceptor_eps = 1e-6 * max(float(np.nanmax(C_acceptor)), 1.0)
        m_d = C_dq > donor_eps
        m_a = C_aq > acceptor_eps

        # Zero out vectors outside masks
        U_dq = np.where(m_d, U_dq, 0.0)
        V_dq = np.where(m_d, V_dq, 0.0)
        U_aq = np.where(m_a, U_aq, 0.0)
        V_aq = np.where(m_a, V_aq, 0.0)

        tiny = 1e-12

        # --- Donor flux scaling and colouring ---
        mag_d = np.hypot(U_dq, V_dq)
        mag_d = np.where(mag_d < tiny, 0.0, mag_d)
        max_mag_d = float(np.nanmax(mag_d)) if np.any(mag_d > 0.0) else 0.0
        if max_mag_d > 0.0:
            max_arrow_len_d = 3.0  # meters
            sf_d = max_arrow_len_d / max_mag_d
            U_dq *= sf_d
            V_dq *= sf_d
        else:
            U_dq[:] = 0.0
            V_dq[:] = 0.0

        # --- Acceptor flux scaling and colouring ---
        mag_a = np.hypot(U_aq, V_aq)
        mag_a = np.where(mag_a < tiny, 0.0, mag_a)
        max_mag_a = float(np.nanmax(mag_a)) if np.any(mag_a > 0.0) else 0.0
        if max_mag_a > 0.0:
            max_arrow_len_a = 3.0  # meters
            sf_a = max_arrow_len_a / max_mag_a
            U_aq *= sf_a
            V_aq *= sf_a
        else:
            U_aq[:] = 0.0
            V_aq[:] = 0.0

        # Common quiver style
        qkws = dict(
            pivot="tail",
            scale_units="xy",
            scale=1,
            width=0.0015,
            headwidth=2.5,
            headlength=4,
            minlength=0,
        )

        # Donor flux (red gradient)
        if np.any(m_d):
            norm_d = mpl.colors.Normalize(vmin=0.0, vmax=max_mag_d if max_mag_d > 0.0 else 1.0)
            plt.quiver(
                Xq[m_d], Yq[m_d],
                U_dq[m_d], V_dq[m_d],
                mag_d[m_d],
                cmap="Reds",
                norm=norm_d,
                alpha=0.9,
                **qkws,
            )

        # Acceptor flux (blue gradient)
        if np.any(m_a):
            norm_a = mpl.colors.Normalize(vmin=0.0, vmax=max_mag_a if max_mag_a > 0.0 else 1.0)
            plt.quiver(
                Xq[m_a], Yq[m_a],
                U_aq[m_a], V_aq[m_a],
                mag_a[m_a],
                cmap="Blues",
                norm=norm_a,
                alpha=0.9,
                **qkws,
            )

        # Axes / labels
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.xlabel("$x$ (m)")
        plt.ylabel("$z$ (m)" if self.config.orientation == "vertical" else "$y$ (m)")

        # Legend (colour semantics only; magnitudes are relative per field)
        legend_handles = [
            Line2D([0], [0], color="red",  lw=3, label="Donor flux ~ -∇C_D"),
            Line2D([0], [0], color="blue", lw=3, label="Acceptor flux ~ -∇C_A"),
        ]
        plt.legend(handles=legend_handles, loc="best")

        if self.config.plot_aspect == "scaled":
            plt.axis("scaled")
        else:
            plt.axis("auto")

        plt.tight_layout()

        results_dir: str = os.path.join("Results")
        os.makedirs(results_dir, exist_ok=True)
        filename_suffix: str = self.generate_filename_suffix()
        plot_filename: str = os.path.join(
            self.run_dir,
            f"{self.run_index:04d}_quiver_{filename_suffix}.pdf"
        )
        plt.savefig(plot_filename)
        print(f"Quiver plot saved to: {plot_filename}")