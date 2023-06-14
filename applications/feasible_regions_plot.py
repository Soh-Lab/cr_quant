"""


2023-05-30 Linus A. Hein
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_2d_fields, plot_feasible_region, plot_feasible_line
from cr_utils.utils import get_readouts, get_affine_bounds, get_r_bounds_measured
from applications.data_handling import read_data, read_metadata_json, convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files

if __name__ == '__main__':
    # load data
    meta_file_name = 'data/2023_05_22_CR8.json'
    data_file_name = 'data/2023_05_22_CR8_combined_2colreads.csv'
    metadata, df = read_data_files(meta_file_name, data_file_name)

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)
    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    colors = ['r', 'orange', 'b']

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    for ind in range(concs.shape[1]):  # iterate over every sample
        true_conc = concs[:, ind]

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

        # print mean and standard deviation of all samples
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]
        print('-' * 20)
        print(f'Log true concs: {np.log10(true_conc)}')
        for reagent_ind, reagent in enumerate(metadata['reagents']):
            print(
                f'{reagent["display_name"]}: {readouts[reagent_ind, 0]:.4f} +- {readout_stds[reagent_ind, 0]:.4f}')

        # assuming standard deviation to be equal to 0.05
        # SK1_std = 0.05
        # XA1_std = 0.05
        # calculate theoretical readout if our model was perfect and the concentrations accurate.
        # readouts = get_readouts(K_A, true_conc)
        # r_bounds = get_r_bounds(readouts, 0.05)#, max_rel_error=0.1)

        # calculate upper and lower bounds on the readout values (+- n * std)
        r_bounds = get_r_bounds_measured(readouts, readout_stds, 2.5)
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

        directory_name = os.path.dirname(__file__)
        fig_file_location = os.path.join(directory_name, os.pardir,
                                         f'output/{int(true_conc[0] * 1e6)}_{int(true_conc[1] * 1e6)}.png')
        plt.savefig(fig_file_location, format='png', dpi=300)
        plt.close(fig)
