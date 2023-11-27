"""


2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from cr_utils.plotting import plot_lower_upper_performance, plot_ellipse, plot_2d_fields, \
    plot_feasible_region
from cr_utils.solvers import apply_solver_parallel, ellipsoid_solver
from cr_utils.utils import get_readouts, get_r_bounds, get_affine_bounds, \
    get_standard_physical_bounds

if __name__ == '__main__':
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 6
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300
    colors = ['r', 'orange', 'b']

    K_D = np.array([
        [1.0, 0.01],
        [0.01, 1.0]
    ])

    true_conc = np.array([[1.49e-2], [5.90e-3]])

    K_A = 1.0 / K_D
    m_reagents = K_D.shape[0]

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
    AA, BB = np.meshgrid(A, B)
    target_concs = np.stack([AA, BB], axis=0)
    readouts = get_readouts(K_A, target_concs)

    # preparing plotting
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, m_reagents, figure=fig)
    combined_ax = fig.add_subplot(gs[1, m_reagents // 2])
    combined_ax.set_aspect('equal')
    axs = [fig.add_subplot(gs[0, i], sharex=combined_ax, sharey=combined_ax) for i in
           range(m_reagents)]

    # plot background
    plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs)

    # generate the affine bounds needed to run the solver
    readouts = get_readouts(K_A, true_conc)
    r_bounds = get_r_bounds(readouts, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    for i in range(m_reagents):
        plot_feasible_region(K_A[i, :],
                             affine_bounds[i, :, 0],
                             axs[i],
                             (log_from, log_to),
                             color=colors[i])
        plot_feasible_region(K_A[i, :],
                             affine_bounds[i, :, 0],
                             combined_ax,
                             (log_from, log_to),
                             color=colors[i])
        axs[i].set_aspect('equal')
    combined_ax.grid()
    fig.set_size_inches(3 * m_reagents, 6)

    # plotting ellipse on the combined axis
    phys_bounds = get_standard_physical_bounds(2)

    r = get_readouts(K_A, true_conc)
    r_bounds = get_r_bounds(r, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    results = apply_solver_parallel(K_A, affine_bounds, phys_bounds, ellipsoid_solver,
                                    n_cores=10)
    plot_ellipse(results[:, 0, 0], results[:, 1:, 0], combined_ax)
    plt.show()
