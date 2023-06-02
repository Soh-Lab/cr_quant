"""


2022-12-10 Linus A. Hein
"""
import numpy as np

from cr_utils.plotting import plot_lower_upper_performance
from cr_utils.solvers import lower_upper_bounds_solver, apply_solver_parallel
from cr_utils.utils import get_r_bounds, get_affine_bounds, \
    get_standard_physical_bounds
from applications.fit_multi_KD import read_data

if __name__ == '__main__':
    # setting K_D values using parameters from fit_multi_KD
    K_D = np.array([
        [10. ** -2.856, 10. ** -3.875],
        [10. ** -0.473, 10. ** -2.792]
    ])

    K_A = 1.0 / K_D

    df = read_data()
    # only select cross-reactive samples
    df = df[~df.singleplex]
    # normalizing the SK1 reads to [0; 1] (using parameters from fit_multi_KD)
    a, d = 1345., 7526.
    df.read_SK1 = (df.read_SK1 - a) / (d - a)
    # normalizing the XA1 reads to [0; 1] (using parameters from fit_multi_KD)
    a, d = 500., 9700.
    df.read_XA1 = (df.read_XA1 - a) / (d - a)

    # convert everything to numpy arrays
    xa, kyn = df.xa_M.to_numpy(), df.kyn_M.to_numpy()
    read_SK1 = df.read_SK1.to_numpy()
    read_XA1 = df.read_XA1.to_numpy()

    # convert target concentrations into single array
    target_concs = np.stack([kyn, xa], axis=0)

    # add the bounds zero and infinity
    phys_bounds = get_standard_physical_bounds(2)

    # get the bounds on the reads
    r = np.stack([read_SK1, read_XA1], axis=0)
    r_bounds = get_r_bounds(r, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    # calculate upper and lower bounds
    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, lower_upper_bounds_solver,
                                   n_cores=10)

    # plot the results
    plot_lower_upper_performance(bounds, target_concs)
