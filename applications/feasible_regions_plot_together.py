"""


2023-06-21 Linus A. Hein
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from applications.data_handling import convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_feasible_region, save_figure, apply_paper_formatting
from cr_utils.utils import get_affine_bounds, get_r_bounds_measured


def plot_one_tile(ax, K_A, true_conc, readouts, readout_stds):
    # calculate upper and lower bounds on the readout values (+- n * std)
    r_bounds = get_r_bounds_measured(readouts, readout_stds, 2.4841)
    affine_bounds = get_affine_bounds(r_bounds)

    for i in range(K_A.shape[0]):
        plot_feasible_region(K_A[i, :],
                             affine_bounds[i, :, 0],
                             ax,
                             (log_from, log_to),
                             color=colors[i])
    ax.scatter(true_conc[0], true_conc[1], color='r', marker='x')


def plot_one_tile_by_index(metadata, ax, K_A, concs, read_avgs, read_stds, ind):
    plot_one_tile(ax, K_A, concs[:, ind], read_avgs[:, ind:ind + 1], read_stds[:, ind:ind + 1])
    format_axis(ax, metadata)


def format_axis(ax, metadata):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(10 ** log_from, 10 ** log_to)
    ax.set_ylim(10 ** log_from, 10 ** log_to)
    ax.set_xlabel(metadata['targets'][0]['display_name'])  # '$T_1$')
    ax.set_ylabel(metadata['targets'][1]['display_name'])  # '$T_2$')
    ax.set_xticks([10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0])
    ax.set_yticks([10 ** -6, 10 ** -4, 10 ** -2, 10 ** 0])
    ax.grid()
    ax.set_aspect(1)


def print_sample(true_conc, readouts, readout_stds):
    print('-' * 20)
    print(f'Log true concs: {np.log10(true_conc)}')
    for reagent_ind, reagent in enumerate(metadata['reagents']):
        print(
            f'{reagent["display_name"]}: '
            f'{readouts[reagent_ind, 0]:.4f} +- {readout_stds[reagent_ind, 0]:.4f}')


def plot_full_grid(metadata, concs, read_avgs, read_stds):
    grid_coor0 = {conc: ind for ind, conc in enumerate(np.unique(concs[1]))}
    grid_coor1 = {conc: ind for ind, conc in enumerate(np.unique(concs[0]))}

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(len(grid_coor0), len(grid_coor1), figure=fig)
    gs.update(hspace=0.0, wspace=0.0)

    for ind in range(concs.shape[1]):  # iterate over every sample
        true_conc = concs[:, ind]

        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        ind0 = grid_coor0[true_conc[1]]
        ind1 = grid_coor1[true_conc[0]]
        ax = fig.add_subplot(gs[len(grid_coor0) - 1 - ind0, ind1])

        print_sample(true_conc, readouts, readout_stds)
        plot_one_tile(ax, K_A, true_conc, readouts, readout_stds)
        format_axis(ax, metadata)

    fig.set_size_inches(len(grid_coor1) * 1.75, len(grid_coor0) * 1.75)
    plt.show()


def plot_extremes(metadata, concs, read_avgs, read_stds, save_folder=None):
    summed = concs[0] + concs[1]
    diffed = concs[0] - concs[1]
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all')
    axs = axs.flatten()
    inds = [np.argmin(diffed), np.argmax(summed), np.argmin(summed), np.argmax(diffed)]

    for ind, ax in zip(inds, axs):
        true_conc = concs[:, ind]
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        plot_one_tile(ax, K_A, true_conc, readouts, readout_stds)
        format_axis(ax, metadata)
    fig.set_size_inches(7, 7)
    fig.tight_layout()
    if save_folder is not None:
        save_figure(fig, os.path.join(save_folder, 'extremes.svg'))
    plt.show()


def plot_conc_range(metadata, concs, read_avgs, read_stds,
                    target_ind, target_conc_ind, save_folder=None):
    target_conc = np.unique(concs[target_ind])[target_conc_ind]
    inds = np.isclose(concs[target_ind], target_conc)
    inds = np.where(inds)[0]

    fig, axs = plt.subplots(nrows=1, ncols=len(inds), sharex='all', sharey='all')
    axs = axs.flatten()

    for ind, ax in zip(inds, axs):
        true_conc = concs[:, ind]
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        plot_one_tile(ax, K_A, true_conc, readouts, readout_stds)
        format_axis(ax, metadata)
    fig.set_size_inches(3 * len(inds), 5)
    fig.tight_layout()
    if save_folder is not None:
        save_figure(fig, os.path.join(save_folder, 'conc_range.svg'))
    plt.show()


if __name__ == '__main__':
    apply_paper_formatting(22)

    # load data
    output_folder = 'output'
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_.csv'

    root_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    data_folder = os.path.join(root_directory, 'data')
    output_folder = os.path.join(root_directory, output_folder)
    metadata, df = read_data_files(os.path.join(data_folder, metadata_name),
                                   os.path.join(data_folder, data_name))

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)
    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    # uncomment these lines to assume mono-specificity
    # off_diag_mat = (np.ones_like(K_A) - np.diag(np.diag(np.ones_like(K_A))))
    # diag_mat = np.ones_like(K_A) - off_diag_mat
    # K_A = off_diag_mat * 1e-9 + np.diag(np.diag(K_A))  # get main diagonal
    # K_A = K_A - np.diag(np.diag(K_A)) + diag_mat * 1e-9 # get off-diagonal

    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    colors = ['r', 'orange', 'b']

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    # plot all samples together
    # plot_full_grid(metadata, concs, read_avgs, read_stds)
    # plot only the extremes
    plot_extremes(metadata, concs, read_avgs, read_stds, output_folder)
    # plot only one column of the full grid (as indexed by the last two inputs)
    plot_conc_range(metadata, concs, read_avgs, read_stds, 0, 1, output_folder)

    # plot and save a single tile

    for ind in range(concs.shape[1]):
        fig, ax = plt.subplots(1, 1)
        plot_one_tile_by_index(metadata, ax, K_A, concs, read_avgs, read_stds, ind)
        fig_path = os.path.join(output_folder,
                                f'/plot_index_{ind}.svg')
        save_figure(fig, fig_path)
        plt.close(fig)
