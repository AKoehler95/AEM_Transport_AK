# mathieu_wrapper.py
import math
import numpy as np
from fortran import mathieu_fortran as mf

class Mathieu:
    """
    Wrap the Fortran matfcn subroutine to match the API of the original
    mathieu_functions_OG.Mathieu class.
    """

    def __init__(self, q: float):
        self.q   = q
        self.icq = 1                        # we always input q directly
        self.isq = 1 if q >= 0 else -1      # sign flag for negative q

    def _call_matfcn(self, *, lnum, ioprad, iopang, izxi,
                     arg_array: np.ndarray) -> tuple:
        """
        Core call to matfcn.  Returns the full output tuple.
        """
        # size parameter: q when icq=1
        qc    = self.q
        # dummy radial coordinate (we only want angular or radial as specified)
        r_val = 1.0
        # for angular: angles in DEGREES
        # for radial-only calls, arg_array is ignored by matfcn if ioprad=1
        arg1  = float(arg_array[0])
        if arg_array.size > 1:
            darg = float(arg_array[1] - arg_array[0])
        else:
            darg = 0.0
        narg  = int(arg_array.size)

        return mf.matfcn(
            int(lnum), int(ioprad), int(iopang), int(izxi),
            int(self.icq), int(self.isq),
            float(qc),    float(r_val),
            float(arg1),  float(darg),
            int(narg),    arg_array
        )

    def ce(self, n, psi):
        """
        even first-kind angular Mathieu function.
        mirrors mathieu_functions_OG.Mathieu.ce :contentReference[oaicite:1]{index=1}
        """
        # ensure array shapes
        n_arr   = np.atleast_1d(n).astype(int)
        psi_rad = np.atleast_1d(psi)
        # convert to degrees for matfcn
        psi_deg = psi_rad * 180.0 / math.pi

        # we only need angular (no radial)
        lnum   = int(n_arr.max()) + 1
        ioprad = 0   # no radial
        iopang = 1   # angular only
        izxi   = 1   # dummy

        out = self._call_matfcn(
            lnum=lnum, ioprad=ioprad, iopang=iopang, izxi=izxi,
            arg_array=psi_deg
        )
        # matfcn returns (..., ce, ced, se, sed, nacca)
        ce = out[-5]

        # select only the requested orders
        result = ce[n_arr, :]
        return result if psi_rad.ndim>0 else result.squeeze()

    def se(self, n, psi):
        """
        odd first-kind angular Mathieu function.
        mirrors mathieu_functions_OG.Mathieu.se :contentReference[oaicite:2]{index=2}
        """
        n_arr   = np.atleast_1d(n).astype(int)
        psi_rad = np.atleast_1d(psi)
        psi_deg = psi_rad * 180.0 / math.pi

        lnum   = int(n_arr.max()) + 1
        ioprad = 0
        iopang = 1
        izxi   = 1

        out = self._call_matfcn(
            lnum=lnum, ioprad=ioprad, iopang=iopang, izxi=izxi,
            arg_array=psi_deg
        )
        se = out[-3]

        result = se[n_arr, :]
        return result if psi_rad.ndim>0 else result.squeeze()

    def Ke(self, n, eta):
        """
        even second-kind radial Mathieu function.
        mirrors mathieu_functions_OG.Mathieu.Ke :contentReference[oaicite:3]{index=3}
        """
        n_arr   = np.atleast_1d(n).astype(int)
        eta_arr = np.atleast_1d(eta)

        # for radial we don’t need angular args, pass dummy [0]
        arg_dummy = np.array([0.0], dtype=np.float64)

        lnum   = int(n_arr.max()) + 1
        ioprad = 1   # radial functions
        iopang = 0   # no angular
        izxi   = 1   # z coordinate

        out = self._call_matfcn(
            lnum=lnum, ioprad=ioprad, iopang=iopang, izxi=izxi,
            arg_array=arg_dummy
        )
        # matfcn returns (mc1c, mc1dc, mc1e, mc1de,
        #                 mc23c, mc23dc, mc23e, mc23de,
        #                 naccrc,
        #                 ms1c, …,
        #                 ms23c, …,
        #                 naccrs,
        #                 ce, ced, se, sed, nacca)
        mc23c = out[4]   # cosine radial 2nd-kind
        result = mc23c[n_arr, :]
        return result if eta_arr.ndim>0 else result.squeeze()

    def Ko(self, n, eta):
        """
        odd second-kind radial Mathieu function.
        mirrors mathieu_functions_OG.Mathieu.Ko :contentReference[oaicite:4]{index=4}
        """
        n_arr   = np.atleast_1d(n).astype(int)
        eta_arr = np.atleast_1d(eta)
        arg_dummy = np.array([0.0], dtype=np.float64)

        lnum   = int(n_arr.max()) + 1
        ioprad = 1
        iopang = 0
        izxi   = 1

        out = self._call_matfcn(
            lnum=lnum, ioprad=ioprad, iopang=iopang, izxi=izxi,
            arg_array=arg_dummy
        )
        ms23c = out[13]  # sine radial 2nd-kind
        result = ms23c[n_arr, :]
        return result if eta_arr.ndim>0 else result.squeeze()
