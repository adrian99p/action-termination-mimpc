"""Direct Gurobi C API access through ctypes and Model._cmodel, avoiding gurobipy's overhead per node QP."""

import ctypes
import glob
import os
from functools import cache

import gurobipy as gp
import numpy as np

_VP = ctypes.c_void_p
_INT = ctypes.c_int


@cache
def _lib() -> ctypes.CDLL:
    """Load the Gurobi shared library bundled with gurobipy."""
    folder = os.path.dirname(gp.__file__)
    for pattern in ("gurobi*.dll", "libgurobi*.so*", "libgurobi*.dylib"):
        paths = glob.glob(os.path.join(folder, pattern))
        if paths:
            break
    else:
        raise RuntimeError(f"Gurobi shared library not found in {folder}")
    lib = ctypes.CDLL(paths[0])

    # Array arguments use raw NumPy buffer addresses.
    lib.GRBgetintattrarray.argtypes = [_VP, ctypes.c_char_p, _INT, _INT, _VP]
    lib.GRBsetintattrarray.argtypes = [_VP, ctypes.c_char_p, _INT, _INT, _VP]
    lib.GRBgetdblattrarray.argtypes = [_VP, ctypes.c_char_p, _INT, _INT, _VP]
    lib.GRBsetdblattrlist.argtypes = [_VP, ctypes.c_char_p, _INT, _VP, _VP]
    lib.GRBgetintattr.argtypes = [_VP, ctypes.c_char_p, _VP]
    lib.GRBgetdblattr.argtypes = [_VP, ctypes.c_char_p, _VP]
    lib.GRBoptimize.argtypes = [_VP]
    lib.GRBgetenv.argtypes = [_VP]
    lib.GRBgetenv.restype = _VP
    lib.GRBsetdblparam.argtypes = [_VP, ctypes.c_char_p, ctypes.c_double]
    lib.GRBgeterrormsg.argtypes = [_VP]
    lib.GRBgeterrormsg.restype = ctypes.c_char_p
    return lib


class CModel:
    """C API access to a live, updated gurobipy model."""

    def __init__(self, model: gp.Model) -> None:
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = _VP
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

        self.model = model
        self.lib = _lib()
        self.ptr = _VP(get_pointer(model._cmodel, None))
        self.env = _VP(self.lib.GRBgetenv(self.ptr))
        self.n_vars = model.NumVars
        self.n_constrs = model.NumConstrs

        # Persistent buffers and their addresses.
        self.bound_idx = np.empty(self.n_vars, np.int32)
        self.bound_lb = np.empty(self.n_vars)
        self.bound_ub = np.empty(self.n_vars)
        self._x = np.empty(self.n_vars)
        self._bound_idx_ref = self.bound_idx.ctypes.data
        self._bound_lb_ref = self.bound_lb.ctypes.data
        self._bound_ub_ref = self.bound_ub.ctypes.data
        self._x_ref = self._x.ctypes.data

        self._int = ctypes.c_int()
        self._dbl = ctypes.c_double()
        self._int_ref = ctypes.addressof(self._int)
        self._dbl_ref = ctypes.addressof(self._dbl)

    def _check(self, rc: int) -> None:
        if rc:
            raise gp.GurobiError(rc, self.lib.GRBgeterrormsg(self.env).decode())

    def set_bounds(self, n: int) -> None:
        """Apply the first n entries of bound_idx/bound_lb/bound_ub."""
        self._check(self.lib.GRBsetdblattrlist(self.ptr, b"LB", n, self._bound_idx_ref, self._bound_lb_ref))
        self._check(self.lib.GRBsetdblattrlist(self.ptr, b"UB", n, self._bound_idx_ref, self._bound_ub_ref))

    def get_basis(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Return basis arrays and their addresses; keep the tuple alive."""
        vbasis = np.empty(self.n_vars, np.int32)
        cbasis = np.empty(self.n_constrs, np.int32)
        v_ref, c_ref = vbasis.ctypes.data, cbasis.ctypes.data
        self._check(self.lib.GRBgetintattrarray(self.ptr, b"VBasis", 0, self.n_vars, v_ref))
        self._check(self.lib.GRBgetintattrarray(self.ptr, b"CBasis", 0, self.n_constrs, c_ref))
        return vbasis, cbasis, v_ref, c_ref

    def set_basis(self, basis: tuple[np.ndarray, np.ndarray, int, int]) -> None:
        """Set a basis returned by get_basis() on this model."""
        self._check(self.lib.GRBsetintattrarray(self.ptr, b"VBasis", 0, self.n_vars, basis[2]))
        self._check(self.lib.GRBsetintattrarray(self.ptr, b"CBasis", 0, self.n_constrs, basis[3]))

    def set_cutoff(self, value: float) -> None:
        """Stop a solve early (status CUTOFF) once its objective is proven above value."""
        self._check(self.lib.GRBsetdblparam(self.env, b"Cutoff", value))

    def optimize(self) -> None:
        self._check(self.lib.GRBoptimize(self.ptr))

    def status(self) -> int:
        self._check(self.lib.GRBgetintattr(self.ptr, b"Status", self._int_ref))
        return self._int.value

    def iter_count(self) -> int:
        self._check(self.lib.GRBgetdblattr(self.ptr, b"IterCount", self._dbl_ref))
        return int(self._dbl.value)

    def obj_val(self) -> float:
        self._check(self.lib.GRBgetdblattr(self.ptr, b"ObjVal", self._dbl_ref))
        return self._dbl.value

    def get_x(self) -> np.ndarray:
        """Return a copy of the solution."""
        self._check(self.lib.GRBgetdblattrarray(self.ptr, b"X", 0, self.n_vars, self._x_ref))
        return self._x.copy()
