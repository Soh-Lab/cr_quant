"""
Demo to show how a feasible region solver works on toy data.

2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from cr_utils.plotting import plot_2d_fields, plot_feasible_region, plot_feasible_line, \
    save_figure
from cr_utils.utils import get_readouts, get_r_bounds, get_affine_bounds

if __name__ == '__main__':
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300
    # whether to plot lines or regions
    LINE_MODE = False

    K_D = np.array([  # example: affinities from the paper, Fig. 1
        [10e-3, 0.1e-3],  # affinity reagent 1
        [0.1e-3, 10e-3]  # affinity reagent 2
    ])

    true_conc = np.array([[1e-4], [1e-4]])

    colors = ['r', 'orange', 'b']

    K_A = 1.0 / K_D

    m_reagents = K_D.shape[0]

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
    AA, BB = np.meshgrid(A, B)
    target_concs = np.stack([AA, BB], axis=0)
    readouts = get_readouts(K_A, target_concs)

    # setup plotting
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(2, m_reagents, figure=fig)
    combined_ax = fig.add_subplot(gs[1, m_reagents // 2])
    axs = [fig.add_subplot(gs[0, i], sharex=combined_ax, sharey=combined_ax) for i in
           range(m_reagents)]

    # plot the background
    plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs)

    # generate the affine bounds needed to run the solver
    readouts = get_readouts(K_A, true_conc)
    if LINE_MODE:
        r_bounds = get_r_bounds(readouts, 0.0)
    else:
        r_bounds = get_r_bounds(readouts, 0.05)
    affine_bounds = get_affine_bounds(r_bounds)

    for i in range(m_reagents):
        if LINE_MODE:
            plot_feasible_line(K_A[i, :],
                               np.average(affine_bounds[i, :, 0]),
                               axs[i],
                               (log_from, log_to),
                               color=colors[i])
            plot_feasible_line(K_A[i, :],
                               np.average(affine_bounds[i, :, 0]),
                               combined_ax,
                               (log_from, log_to),
                               color=colors[i])
        else:
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

        axs[i].scatter(true_conc[0], true_conc[1], color='r', marker='x')
        axs[i].set_aspect(1)
    combined_ax.grid()
    combined_ax.set_aspect(1)
    combined_ax.scatter(true_conc[0], true_conc[1], color='r', marker='x')
    combined_ax.set_xlabel('$T_1$')
    combined_ax.set_ylabel('$T_2$')
    fig.set_size_inches(3 * m_reagents, 6)
    fig.set_layout_engine('compressed')

    save_figure(plt.gcf(), '../output/overlaps.svg')
    plt.show()
