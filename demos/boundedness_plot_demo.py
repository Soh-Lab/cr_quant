"""


2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt

from cr_utils.plotting import plot_2d_boundedness_results
from cr_utils.solvers import apply_solver_parallel, boundedness_solver
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
        [100, 1.0]
    ])

    K_A = 1.0 / K_D

    phys_bounds = get_standard_physical_bounds(2)

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
    AA, BB = np.meshgrid(A, B)

    # generate the affine bounds needed to run the solver
    target_concs = np.stack([AA, BB], axis=0)
    readouts = get_readouts(K_A, target_concs)
    r_bounds = get_r_bounds(readouts, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, boundedness_solver,
                                   n_cores=N_CORES)

    axs = plot_2d_boundedness_results(target_concs, bounds, 'log')
    plt.show()
