"""


2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt

from cr_utils.plotting import plot_2d_fields
from cr_utils.solvers import apply_solver_parallel, boundedness_solver, lower_upper_bounds_solver
from cr_utils.utils import get_standard_physical_bounds, get_readouts, get_r_bounds, \
    get_affine_bounds

if __name__ == '__main__':
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 6
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    K_D = np.array([
        [1.0, 100],
        [100, 1.0],
        # [10, 10]
    ])

    K_A = 1.0 / K_D

    phys_bounds = get_standard_physical_bounds(2)

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
    AA, BB = np.meshgrid(A, B)  # (log_steps, log_steps)

    # generate the affine bounds needed to run the solver
    target_concs = np.stack([AA, BB], axis=0)  # (n_targets=2, log_steps, log_steps)
    readouts = get_readouts(K_A, target_concs)  # (m_affinity=2, log_steps, log_steps)
    r_bounds = get_r_bounds(readouts, 0.05)  # (m_affinity=2, 2, log_steps, log_steps)
    affine_bounds = get_affine_bounds(r_bounds)  # (m_affinity=2, 2, log_steps, log_steps)

    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, boundedness_solver,
                                   n_cores=N_CORES)  # (n_targets=2, 1, log_steps, log_steps)

    # calculate for which indices we have non-infinite bounds
    t_is_bounded = bounds[:, 0, ...] == 0
    any_is_bounded = np.logical_or(t_is_bounded[0], t_is_bounded[1])

    resolution = np.ones((2, 2, log_steps, log_steps))

    result_flat = apply_solver_parallel(K_A, affine_bounds[:, :, any_is_bounded], phys_bounds,
                                        lower_upper_bounds_solver,
                                        n_cores=N_CORES)
    # (n_targets, result_index, log_steps, log_steps)
    resolution[:, :, any_is_bounded] = result_flat
    # (n_targets, log_steps, log_steps)
    resolution = np.log10(resolution[:, 1]) - np.log10(resolution[:, 0])
    worst_performance = np.nanmax(resolution) + 1
    resolution[0, ~t_is_bounded[0]] = worst_performance
    resolution[1, ~t_is_bounded[1]] = worst_performance

    plot_2d_fields(target_concs, resolution, 'Log-Confidence in Target', limits=(0.0, 3.0))
    plt.show()
