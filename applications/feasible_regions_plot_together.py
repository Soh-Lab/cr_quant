"""


2023-06-21 Linus A. Hein
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
            f'{reagent["display_name"]}: {readouts[reagent_ind, 0]:.4f} +- {readout_stds[reagent_ind, 0]:.4f}')


def plot_full_grid(concs, read_avgs, read_stds):
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

def save_figure(name):
    directory_name = os.path.dirname(__file__)
    fig_file_location = os.path.join(directory_name, os.pardir,
                                     f'output/{name}.svg')
    plt.savefig(fig_file_location, format='svg', dpi=300)

def plot_extremes(concs, read_avgs, read_stds, save_fig=False):
    summed = concs[0] + concs[1]
    diffed = concs[0] - concs[1]
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    axs = axs.flatten()
    inds = [np.argmin(diffed), np.argmax(summed), np.argmin(summed), np.argmax(diffed)]

    for ind, ax in zip(inds, axs):
        true_conc = concs[:, ind]
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        plot_one_tile(ax, K_A, true_conc, readouts, readout_stds)
        format_axis(ax, metadata)
    fig.set_size_inches(7, 7)
    fig.tight_layout() #pad=2.0, w_pad=0.5, h_pad=0.5)
    if save_fig: save_figure('extremes')
    plt.show()


def plot_conc_range(concs, read_avgs, read_stds, target_ind, target_conc_ind, save_fig=False):
    target_conc = np.unique(concs[target_ind])[target_conc_ind]
    inds = np.isclose(concs[target_ind], target_conc)
    inds = np.where(inds)[0]

    fig, axs = plt.subplots(nrows=1, ncols=len(inds), sharex=True, sharey=True)
    axs = axs.flatten()

    for ind, ax in zip(inds, axs):
        true_conc = concs[:, ind]
        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        plot_one_tile(ax, K_A, true_conc, readouts, readout_stds)
        format_axis(ax, metadata)
    fig.set_size_inches(3 * len(inds), 5)
    fig.tight_layout() #pad=0.1, w_pad=0.5, h_pad=0.5)
    if save_fig: save_figure('conc_range')
    plt.show()

import matplotlib as mpl
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 22, 'font.family':'serif',
                         'xtick.labelsize':15, 'ytick.labelsize':15})

    # load data
    meta_file_name = 'data/2023_05_22_CR8.json'
    data_file_name = 'data/2023_06_20_colreads.csv'
    metadata, df = read_data_files(meta_file_name, data_file_name)

    # Remove outlier from SK1:xa highest read 1.5mM
    # df.loc[(df.xa_M == 0.00150) & (df.singleplex), 'read_SK1'] = np.nan
    df = df.loc[~((df.xa_M == 0.00150) & (df.singleplex))] # Also drops XA1 highest conc... but this is easy

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)
    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -6, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    colors = ['r', 'orange', 'b']

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    # plot all samples together
    # plot_full_grid(concs, read_avgs, read_stds)
    # plot only the extremes
    plot_extremes(concs, read_avgs, read_stds, save_fig=True)
    # plot only one column of the full grid (as indexed by the last two inputs)
    plot_conc_range(concs, read_avgs, read_stds, 0, 1, save_fig=True)

