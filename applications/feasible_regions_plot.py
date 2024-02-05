"""
Show how a feasible region solver works on real data.

2023-05-30 Linus A. Hein
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import t as t_dist

from applications.data_handling import convert_dataframe_to_numpy, convert_dataframe_to_avg_std, \
    read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_2d_fields, plot_feasible_region, save_figure, \
    apply_paper_formatting
from cr_utils.utils import get_readouts, get_affine_bounds, get_r_bounds_measured

if __name__ == '__main__':
    apply_paper_formatting(18)
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_filtered.csv'

    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 100
    # whether to plot two rows (first row, individual aptamers, second row combined plot) or just a
    # single combined plot

    # plot detailed curves of feasible sets for each affinity reagent (like in Fig 2bc)
    DETAILED_PLOTS = False
    # replaces second affinity reagent with perfect information (like in Fig S1)
    PERFECT_INFORMATION = False
    # whether to assume that every affinity reagent is perfectly specific (naive approach)
    NAIVE = False

    # load data
    root_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    data_folder = os.path.join(root_directory, 'data')
    metadata, df = read_data_files(os.path.join(data_folder, metadata_name),
                                   os.path.join(data_folder, data_name))

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)
    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D
    m_reagents = K_A.shape[0]

    if NAIVE:
        K_A = np.diag(np.diag(K_A))
        K_A[K_A == 0] = 1e-9

    colors = ['r', 'orange', 'b']

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)
    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    for ind in range(concs.shape[1]):  # iterate over every experimental condition
        true_conc = concs[:, ind]

        fig = plt.figure()
        if DETAILED_PLOTS:
            gs = GridSpec(2, m_reagents, figure=fig)

            combined_ax = fig.add_subplot(gs[1, m_reagents // 2])
            axs = [fig.add_subplot(gs[0, i], sharex=combined_ax, sharey=combined_ax) for i in
                   range(m_reagents)]

            # create a 2D-log-meshgrid of target concentrations
            A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to,
                                                                         log_steps)
            AA, BB = np.meshgrid(A, B)
            # generate the affine bounds needed to run the solver
            target_concs = np.stack([AA, BB], axis=0)
            readouts = get_readouts(K_A, target_concs)

            # plot binding "curves" (the heatmap in the background)
            plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs, colorbar_scale=0.8)
            for i, ax in enumerate(axs):
                ax.set_title(f'$A_{i + 1}$')
        else:
            combined_ax = fig.add_subplot()
            combined_ax.set_xscale('log')
            combined_ax.set_yscale('log')
            combined_ax.set_xlim(10 ** log_from, 10 ** log_to)
            if PERFECT_INFORMATION:
                combined_ax.set_ylim(10 ** log_from, 10 ** -3)
            else:
                combined_ax.set_ylim(10 ** log_from, 10 ** log_to)

        # print mean and standard deviation of all samples
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]
        print('-' * 20)
        print(f'Log true concs: {np.log10(true_conc)}')
        for reagent_ind, reagent in enumerate(metadata['reagents']):
            print(
                f'{reagent["display_name"]}: {readouts[reagent_ind, 0]:.4f} +- {readout_stds[reagent_ind, 0]:.4f}')

        # calculate upper and lower bounds on the readout values (+- n * std)
        n_samples = 3
        factor = t_dist.ppf(0.975, n_samples - 1) / np.sqrt(n_samples)
        r_bounds = get_r_bounds_measured(readouts, readout_stds, factor)
        affine_bounds = get_affine_bounds(r_bounds)

        for i in range(m_reagents):
            if DETAILED_PLOTS:
                axs[i].set_aspect(1)
                # plot regions
                plot_feasible_region(K_A[i, :],
                                     affine_bounds[i, :, 0],
                                     axs[i],
                                     (log_from, log_to),
                                     color=colors[i])
                axs[i].scatter(true_conc[0], true_conc[1], color='r', marker='x')
            if PERFECT_INFORMATION and (i > 0):
                continue
            plot_feasible_region(K_A[i, :],
                                 affine_bounds[i, :, 0],
                                 combined_ax,
                                 (log_from, log_to),
                                 color=colors[i])
        # plot true concentration in combined plot
        combined_ax.grid()
        if PERFECT_INFORMATION:
            combined_ax.plot([0, 1e6], [true_conc[1], true_conc[1]], linestyle='-', color='#fda927')
        combined_ax.scatter(true_conc[0], true_conc[1], color='r', marker='x', zorder=2.)
        combined_ax.set_aspect(1)

        plt.show()
        fig.set_size_inches(3.7, 3.7)
        fig.set_layout_engine('compressed')

        fig_path = os.path.join(root_directory, 'output', 'tmp',
                                f'{int(true_conc[0] * 1e6)}_{int(true_conc[1] * 1e6)}.svg')
        save_figure(fig, fig_path)
        plt.close(fig)
