r"""Numerical solvers interfaces.

Our general problem formulation is as follows

.. math::

    \begin{array}{ll}

    \text{minimize}   & c^T x + \frac{1}{2} x^T Q x \\
    \text{subject to} & A x = b \\
                      & d_l \leq E x \leq d_u \\
                      & F x - g \in \mathcal{K} 
                      & l \leq x \leq u
    \end{array}

where :math:`\mathcal{k}` is a composite cone made up of various base cones.
Depending on which features a solver supports, we provide the problem data
in different form.

"""
import numpy as np
import scipy.sparse as sps
from scipy.optimize import linprog

class Solver:
    """Base solver class."""

    matrix_type = None
    variable_bounds = False
    quadratic_objective = False
    two_sided_lpcone = False
    zero_cone = False
    lp_cone = False
    socp_cone = False
    exp_cone = False
    sdp_cone = False
    pow_cone = False

class LeastSquaresDense(Solver):
    """Solve dense quadratic program with affine equalities only.
    
    :param Q: Quadratic symmetric, positive semi-definite.
    :type Q: np.array
    :param c: Linear objective array.
    :type c: np.array
    :param A: Affine equality matrix.
    :type A: np.array
    :param b: Affine equality right-hand-side array.
    :type b: np.array
    
    :returns: Solution and dual vector.
    :rtype: (np.array, np.array)
    """

    matrix_type = np.array
    quadratic_objective = True
    zero_cone = True

    def __init__(self, Q, c, A, b):

        mat = np.block(
            [
                [Q, A.T],
                [A, np.zeros((A.shape[1], A.shape[1]))]
            ]
        )
        rhs = np.concatenate([-c, b])
        res = np.linalg.solve(mat, rhs)
        x = res[:len(c)]
        y = res[len(c):]
        return x, y



class Linprog(Solver):
    """Scipy's linear programming solver."""

    matrix_type = np.array
    variable_bounds = True
    zero_cone = True
    lp_cone = True

    def __init__(self, c, A, b, E, d, l, u):
        pass

class Osqp(Solver):
    """Interface to OSQP."""

    matrix_type = sps.csc_matrix
    quadratic_objective = True
    zero_cone = True
    lp_cone = True
    two_sided_lpcone = True

    def __init__(self, Q, c, E, d_l, d_u):
        pass

class Daqp(Solver):
    """Interface to DAQP."""

    matrix_type = np.array
    variable_bounds = True
    quadratic_objective = True
    zero_cone = True
    lp_cone = True
    two_sided_lpcone = True

    def __init__(self, Q, c, E, d_l, d_u, l, u):
        pass