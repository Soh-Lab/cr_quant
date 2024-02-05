"""
Demo to show how to evaluate the performance of our quantification results.
Left column displays the log-error of taking the average of the lower and upper bound vs the true
concentration: log((upper bound + lower bound) / 2) - log(true concentration).
Right column shows how confident we are in our estimate: log(upper bound) - log(lower bound).

2022-12-10 Linus A. Hein
"""
import numpy as np

from cr_utils.plotting import plot_lower_upper_performance
from cr_utils.solvers import lower_upper_bounds_solver, apply_solver_parallel
from cr_utils.utils import get_readouts, get_r_bounds, get_affine_bounds, \
    get_standard_physical_bounds

if __name__ == '__main__':
    K_D = np.array([
        [1.0, 0.01],
        [0.01, 1.0]
    ])

    K_A = 1.0 / K_D

    target_concs = np.array([
        [1.49e-2, 0.01, 1],
        [5.90e-3, 0.01, 1]
    ])

    phys_bounds = get_standard_physical_bounds(2)

    r = get_readouts(K_A, target_concs)
    r_bounds = get_r_bounds(r, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, lower_upper_bounds_solver,
                                   n_cores=10)
    plot_lower_upper_performance(bounds, target_concs)
