"""


2023-05-30 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from cr_utils.plotting import plot_2d_fields, plot_feasible_region
from cr_utils.utils import get_readouts, get_affine_bounds, get_r_bounds_measured
from applications.fit_multi_KD import read_data

if __name__ == '__main__':
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -8, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    ind = 0

    # setting K_D values using parameters from fit_multi_KD
    K_D = np.array([
        [10. ** -2.856, 10. ** -3.875],
        [10. ** -0.473, 10. ** -2.792]
    ])
    K_A = 1.0 / K_D

    colors = ['r', 'orange', 'b']

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

    for ind in range(len(read_SK1)):  # iterate over every sample
        true_conc = np.array([[kyn[ind]], [xa[ind]]])

        m_reagents = K_A.shape[0]

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

        # plot binding "curves" (the heatmap in the background)
        plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs)

        # calculate mean and standard deviation of readouts across replicates
        indices = (kyn == kyn[ind]) & (xa == xa[ind])

        SK1 = np.average(read_SK1[indices])
        SK1_std = np.std(read_SK1[indices])
        XA1 = np.average(read_XA1[indices])
        XA1_std = np.std(read_XA1[indices])

        # print mean and standard deviation of all samples
        print('-' * 20)
        print(f'Log true concs: {np.log10(true_conc)}')
        print(f'SK1: {SK1 * (7526. - 1345.) + 1345.:.1f} +- {SK1_std:.4f}')
        print(f'XA1: {XA1 * (9700. - 500.) + 500.:.1f} +- {XA1_std:.4f}')

        # convert to a readout array that the plotters can use
        readouts = np.array([[SK1],
                             [XA1]])
        readout_stds = np.array([[SK1_std],
                                 [XA1_std]])
        # assuming standard deviation to be equal to 0.05
        # SK1_std = 0.05
        # XA1_std = 0.05
        # calculate theoretical readout if our model was perfect and the concentrations accurate.
        # readouts = get_readouts(K_A, true_conc)
        # r_bounds = get_r_bounds(readouts, 0.05)#, max_rel_error=0.1)

        # calculate upper and lower bounds on the readout values (+- n * std)
        r_bounds = get_r_bounds_measured(readouts, readout_stds, 5)
        # r_bounds = np.zeros((readouts.shape[0], 2, readouts.shape[1]))
        # n_stds = 5
        # r_bounds[0, 0, 0], r_bounds[0, 1, 0] = SK1 - SK1_std * n_stds, SK1 + SK1_std * n_stds
        # r_bounds[1, 0, 0], r_bounds[1, 1, 0] = XA1 - XA1_std * n_stds, XA1 + XA1_std * n_stds

        affine_bounds = get_affine_bounds(r_bounds)

        for i in range(m_reagents):
            axs[i].scatter(true_conc[0], true_conc[1], color='r', marker='x')
            axs[i].set_aspect(1)

            # plot regions
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
            # plot lines
            # plot_feasible_line(K_A[i, :],
            #                  np.average(affine_bounds[i, :, 0]),
            #                  axs[i],
            #                  (log_from, log_to),
            #                  color=colors[i])
            # plot_feasible_line(K_A[i, :],
            #                      np.average(affine_bounds[i, :, 0]),
            #                      combined_ax,
            #                      (log_from, log_to),
            #                      color=colors[i])
        # plot true concentration in combined plot
        combined_ax.grid()
        combined_ax.scatter(true_conc[0], true_conc[1], color='r', marker='x')
        combined_ax.set_aspect(1)
        fig.set_size_inches(4 * m_reagents, 9)
        # plt.show()
        plt.savefig(
            f'/Users/linus/workspace/cr_quant/output/{int(kyn[ind] * 1e6)}_{int(xa[ind] * 1e6)}.png',
            dpi=300)
        plt.close(fig)
