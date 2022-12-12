"""


2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from cr_utils.plotting import plot_2d_fields, plot_feasible_region
from cr_utils.utils import get_readouts, get_r_bounds, get_affine_bounds

if __name__ == '__main__':
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 6
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    K_D = np.array([
        # [1.0, 0.25],
        [100, 1.0],
        [1.0, 100.0]
    ])

    colors = ['r', 'orange', 'b']

    # true_conc = np.array([[1e3], [1e-4]])
    true_conc = np.array([[10], [2]])
    # true_conc = np.array([[1], [10]])

    K_A = 1.0 / K_D

    m_reagents = K_D.shape[0]

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
    AA, BB = np.meshgrid(A, B)
    # generate the affine bounds needed to run the solver
    target_concs = np.stack([AA, BB], axis=0)
    readouts = get_readouts(K_A, target_concs)

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, m_reagents, figure=fig)

    combined_ax = fig.add_subplot(gs[1, m_reagents // 2])
    axs = [fig.add_subplot(gs[0, i], sharex=combined_ax, sharey=combined_ax) for i in
           range(m_reagents)]

    plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs)

    readouts = get_readouts(K_A, true_conc)
    r_bounds = get_r_bounds(readouts, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    for i in range(m_reagents):
        plot_feasible_region(K_A[i, :],
                             affine_bounds[i, :, 0],
                             axs[i],
                             (log_from, log_to),
                             color=colors[i])
        axs[i].scatter(true_conc[0], true_conc[1], color='r', marker='x')
        plot_feasible_region(K_A[i, :],
                             affine_bounds[i, :, 0],
                             combined_ax,
                             (log_from, log_to),
                             color=colors[i])
    combined_ax.grid()
    fig.set_size_inches(4 * m_reagents, 9)
    plt.show()
