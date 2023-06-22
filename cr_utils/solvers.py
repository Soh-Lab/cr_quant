"""Collection of solvers and solver-helper functions."""
from functools import partial
from multiprocessing import Pool
from typing import Callable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint

from cr_utils.utils import get_standard_physical_bounds


def _generate_LP_constraints(K_A: np.ndarray,
                             affine_bounds: np.ndarray,
                             physiological_bounds: np.ndarray,
                             T_scaled: cp.Variable) -> list[Constraint]:
    """
    Internal helper function that generates constraints.

    2022-12-09 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2) lower (index 0) and upper (index 1) bounds on the m measurements.
    :param physiological_bounds: (n, 2) lower (index 0) and upper (index 1) bounds on the n targets.
    :param T_scaled: scaled version of target concentration cvxpy.Variable.
    :return: list of convex constraints.
    """
    m_reagents, n_targets = K_A.shape
    constraints = []
    # add constraints from reagents
    for i in range(m_reagents):
        lower, upper = affine_bounds[i, 0], affine_bounds[i, 1]
        K_A_i = K_A[i, :]
        if lower > 0:
            constraints.append(K_A_i @ T_scaled >= lower)
        if upper < np.infty:
            constraints.append(K_A_i @ T_scaled <= upper)

    # add physiological constraints
    for j in range(n_targets):
        lower, upper = physiological_bounds[j, 0], physiological_bounds[j, 1]
        constraints.append(T_scaled[j] >= np.maximum(0.0, lower))
        if upper < np.infty:
            constraints.append(T_scaled[j] <= upper)
    return constraints


def lower_upper_bounds_solver(K_A: np.ndarray,
                              affine_bounds: np.ndarray,
                              physiological_bounds: np.ndarray
                              ):
    """
    Calculate the lower and upper bounds on all target concentrations.

    2022-12-09 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2) lower (index 0) and upper (index 1) bounds on the m measurements.
    :param physiological_bounds: (n, 2) lower (index 0) and upper (index 1) bounds on the n targets.

    :return: (n, 2) lower and upper bounds on the n targets.
    """
    m_reagents, n_targets = K_A.shape
    if np.all(affine_bounds[:, 1] == np.infty):  # all affinity reagents are maxed out
        # no upper bounds exist for all targets (and no lower bound)
        output = np.zeros((n_targets, 2))
        output[:, 1] = np.infty
        return output

    # normalize T vector
    corr_factor = np.mean(K_A, axis=0)
    T_scaled = cp.Variable(n_targets)
    K_A = K_A / corr_factor[np.newaxis, :]

    constraints = _generate_LP_constraints(K_A, affine_bounds, physiological_bounds, T_scaled)

    # solve the linear programs
    T_bounds = np.zeros((n_targets, 2))
    for k in range(n_targets):
        # lower bound
        prob = cp.Problem(cp.Minimize(T_scaled[k]), constraints)
        prob.solve()
        T_bounds[k, 0] = T_scaled[k].value
        # upper bound
        prob = cp.Problem(cp.Maximize(T_scaled[k]), constraints)
        prob.solve()
        T_bounds[k, 1] = T_scaled[k].value

    return T_bounds / corr_factor[:, np.newaxis]


def boundedness_solver(K_A: np.ndarray,
                       affine_bounds: np.ndarray,
                       physiological_bounds: np.ndarray = None):
    """
    Calculate the boundedness of all target concentrations as given by the affine_bounds alone.
    Physiological bounds are completely ignored.
        - 0: lower bound on target exists.
        - 1: no lower bound on target exists
        - 2: neither lower bound nor upper bound on the target exists

    2022-12-09 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2) lower (index 0) and upper (index 1) bounds on the m measurements.
    :param physiological_bounds: (n, 2) is ignored. Lower (index 0) and upper (index 1) bounds on
        the n targets.

    :return: (n, 1) boundedness of all target molecules.
    """
    m_reagents, n_targets = K_A.shape
    if np.all(affine_bounds[:, 1] == np.infty):  # all affinity reagents are maxed out
        # no upper bounds exist for all targets (and no lower bound)
        return np.ones((n_targets, 1)) * 2
    if np.all(affine_bounds[:, 0] <= 0.0):  # all affinity reagents are bottoming out
        # no lower bound exist for all targets
        return np.ones((n_targets, 1))

    # ensure that physiological bounds are ignored
    physiological_bounds = get_standard_physical_bounds(n_targets)

    T = cp.Variable(n_targets)

    # normalize T vector
    corr_factor = np.max(K_A, axis=0)
    T_scaled = T / corr_factor

    constraints = _generate_LP_constraints(K_A, affine_bounds, physiological_bounds, T_scaled)

    boundedness = np.zeros((n_targets, 1))
    for k in range(n_targets):  # iterate over target molecules
        # check whether it is possible for k-th target concentration to be zero
        # set constraint that all target concentrations should be greater than 0
        # except for current concentration which is zero
        prob = cp.Problem(cp.Minimize(T[k]), constraints + [T[k] == 0])
        prob.solve()

        if prob.status == 'infeasible':  # lower bound on i-th target exists
            boundedness[k, 0] = 0
        elif prob.status == 'optimal':  # i-th target may be zero, so no lower bound exists
            boundedness[k, 0] = 1
        else:
            # unbounded, so cannot be 0? -> lower bound on i-th target exists
            boundedness[k, 0] = 0
    return boundedness


def ellipsoid_solver(K_A: np.ndarray,
                     affine_bounds: np.ndarray,
                     physiological_bounds: np.ndarray = None) -> np.ndarray:
    """

    2022-12-10 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2) lower (index 0) and upper (index 1) bounds on the m measurements.
    :param physiological_bounds: (n, 2) lower (index 0) and upper (index 1) bounds on the n targets.

    :return: (n, n + 1) Where first column is the centroid of the ellipsoid and the remaining
        square matrix is the matrix describing the eccentricity of the ellipsoid.
    """
    m_reagents, n_targets = K_A.shape
    if np.all(affine_bounds[:, 1] == np.infty):  # all affinity reagents are maxed out
        # no upper bounds exist for all targets (and no lower bound)
        T = np.ones(n_targets)  # ellipsoid center is infinity
        T[:] = np.infty
        B = np.zeros((n_targets, n_targets))  # ellipsoid matrix has zeros everywhere
        return np.column_stack([T, B])

    B = cp.Variable((n_targets, n_targets), PSD=True)

    T_scaled = cp.Variable(n_targets)
    corr_factor = np.mean(K_A, axis=0)
    K_A = K_A / corr_factor[np.newaxis, :]

    constraints = []
    # add constraints from reagents
    for i in range(m_reagents):
        lower, upper = affine_bounds[i, 0], affine_bounds[i, 1]
        K_A_i = K_A[i, :]

        if lower > 0:  # constraint - K_A_i @ T_scaled <= -lower
            constraints.append(cp.norm2(B @ K_A_i) - K_A_i @ T_scaled <= -lower)
        if upper < np.infty:  # constraint K_A_i @ T_scaled <= upper
            constraints.append(cp.norm2(B @ K_A_i) + K_A_i @ T_scaled <= upper)

    # add physiological constraints
    for j in range(n_targets):
        lower, upper = physiological_bounds[j, 0], physiological_bounds[j, 1]
        lower = np.maximum(0.0, lower)
        e_j = np.zeros(n_targets)
        e_j[j] = 1.0
        # constraint - e_j @ T_scaled <= -lower
        constraints.append(cp.norm2(B @ e_j) - T_scaled[j] <= -lower)

        if upper < np.infty:  # constraint e_j @ T_scaled <= upper
            constraints.append(cp.norm2(B @ e_j) + T_scaled[j] <= upper)

    prob = cp.Problem(cp.Maximize(cp.log_det(B)), constraints)
    prob.solve()
    return np.column_stack([T_scaled.value / corr_factor, B.value / corr_factor[:, np.newaxis]])


def apply_solver(K_A: np.ndarray,
                 affine_bounds: np.ndarray,
                 physiological_bounds: np.ndarray,
                 solver: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]):
    """
    Helper function to apply a solver function to a lot of affine bounds conditions one after
    another.

    2022-12-09 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2, ...) lower and upper bound on the m measurements.
    :param physiological_bounds: (n, 2) Lower and upper bounds on the n targets.
    :param solver: Solver to apply to all measurements.

    :return: (n, x, ...) output of the solver for all samples. Last dimensions match the last
        dimensions of affine_bounds.
    """
    bounds_shape = affine_bounds.shape
    n_targets = physiological_bounds.shape[0]

    affine_bounds = affine_bounds.reshape((bounds_shape[0], bounds_shape[1], -1))
    results = []
    result_shape = None
    for k in range(affine_bounds.shape[2]):
        result = solver(K_A, affine_bounds[:, :, k], physiological_bounds)
        if result_shape is None:
            result_shape = result.shape
        results.append(result)
    results = np.stack(results, axis=-1)

    return results.reshape((n_targets, *result_shape[1:], *bounds_shape[2:]))


def _parallelization_helper(K_A, affine_bounds, physiological_bounds, solver, index):
    """Helper function for parallelization. Has to be global function to be pickled."""
    return solver(K_A, affine_bounds[:, :, index], physiological_bounds)


def apply_solver_parallel(K_A: np.ndarray,
                          affine_bounds: np.ndarray,
                          physiological_bounds: np.ndarray,
                          solver: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                          n_cores: int = 2):
    """
    Helper function to apply a solver function to a lot of affine bounds conditions one after
    another, leveraging multithreading.

    2022-12-09 Linus A. Hein.

    :param K_A: (m, n) Affinity coefficients of all m affinity reagents against all n targets.
    :param affine_bounds: (m, 2, ...) lower and upper bound on the m measurements.
    :param physiological_bounds: (n, 2) Lower and upper bounds on the n targets.
    :param solver: Solver to apply to all measurements.
    :param n_cores: Number of cores to parallelize across.

    :return: (n, x, ...) output of the solver for all samples. Last dimensions match the last
        dimensions of affine_bounds.
    """
    bounds_shape = affine_bounds.shape
    n_targets = physiological_bounds.shape[0]

    affine_bounds = affine_bounds.reshape((bounds_shape[0], bounds_shape[1], -1))
    n_problems = affine_bounds.shape[2]
    with Pool(np.minimum(n_cores, n_problems)) as pool:
        inputs = list(range(n_problems))
        inputs = [(index,) for index in inputs]
        func = partial(_parallelization_helper,
                       K_A, affine_bounds, physiological_bounds,
                       solver)
        results = pool.starmap(func, inputs)

    result_shape = results[0].shape
    results = np.stack(results, axis=-1)

    return results.reshape((n_targets, *result_shape[1:], *bounds_shape[2:]))
