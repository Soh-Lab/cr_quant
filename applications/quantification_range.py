"""
Generate plots like Fig 2i.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from applications.data_handling import read_data_files, convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import save_figure, apply_paper_formatting
from cr_utils.solvers import apply_solver_parallel, lower_upper_bounds_solver
from cr_utils.utils import get_standard_physical_bounds, get_r_bounds_measured, get_affine_bounds


def plot_CIs(metadata, ax, concs, bounds1, bounds2,
             y_axis_target_ind, column_ind, same_target=False):
    x_axis_target_ind = 1 - y_axis_target_ind
    y_conc = np.unique(concs[y_axis_target_ind])[column_ind]
    inds = np.isclose(concs[y_axis_target_ind], y_conc)
    inds = np.where(inds)[0]

    x = concs[x_axis_target_ind, inds]

    if same_target:
        y_axis_target_ind = x_axis_target_ind

    for color, bounds in [('tab:blue', bounds1), ('tab:orange', bounds2)]:
        heights = bounds[y_axis_target_ind, 1, inds] - bounds[y_axis_target_ind, 0, inds]
        bottom = bounds[y_axis_target_ind, 0, inds]
        ax.bar(np.log10(x), heights, 0.15, bottom, color=color, log=True, alpha=0.5)

    if same_target:
        ax.step(np.log10(x), x, color='g', linestyle='-.', where='mid')
    else:
        ax.axhline(y=y_conc, color='g', linestyle='-.')

    exponents = np.floor(np.log10(x))
    bases = np.float_power(10, np.log10(x) - exponents)
    ax.set_xticks(np.log10(x),
                  [f'${base:.1f}\\times 10^' + '{' + f'{exponent:.0f}' + '}$' if base != 1
                   else f'$10^' + '{' + f'{exponent:.0f}' + '}$'
                   for base, exponent in zip(bases, exponents)])
    ax.set_xlabel(metadata['targets'][x_axis_target_ind]['display_name'])
    ax.set_ylabel(metadata['targets'][y_axis_target_ind]['display_name'])


def plot_CIs2(metadata, ax, concs, bounds1, bounds2,
             y_axis_target_ind, column_ind, same_target=False):
    x_axis_target_ind = 1 - y_axis_target_ind
    y_conc = np.unique(concs[y_axis_target_ind])[column_ind]
    inds = np.isclose(concs[y_axis_target_ind], y_conc)
    inds = np.where(inds)[0]

    x = concs[x_axis_target_ind, inds]

    if same_target:
        y_axis_target_ind = x_axis_target_ind

    for color, bounds in [('tab:orange', bounds1), ('tab:blue', bounds2)]:
        heights = bounds[y_axis_target_ind, 1, inds] - bounds[y_axis_target_ind, 0, inds]
        bottom = bounds[y_axis_target_ind, 0, inds]
        ax.barh(np.log10(x), heights, 0.15, bottom, color=color, log=True, alpha=0.5)

    if same_target:
        ax.step(np.log10(x), x, color='g', linestyle='-.', where='mid')
    else:
        # ax.axvline(x=y_conc, color='g', linestyle='-.')
        ax.scatter([y_conc] * len(x), np.log10(x), color='r', marker='x')

    x = np.logspace(-6, -3, 4)
    exponents = np.floor(np.log10(x))
    bases = np.float_power(10, np.log10(x) - exponents)
    ax.set_yticks(np.log10(x),
                  [f'${base:.1f}\\times 10^' + '{' + f'{exponent:.0f}' + '}$' if base != 1
                   else f'$10^' + '{' + f'{exponent:.0f}' + '}$'
                   for base, exponent in zip(bases, exponents)])
    ax.set_ylabel(metadata['targets'][x_axis_target_ind]['display_name'])
    ax.set_xlabel(metadata['targets'][y_axis_target_ind]['display_name'])


if __name__ == '__main__':
    apply_paper_formatting(18)
    # load data
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_filtered.csv'

    root_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    data_folder = os.path.join(root_directory, 'data')
    metadata, df = read_data_files(os.path.join(data_folder, metadata_name),
                                   os.path.join(data_folder, data_name))

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)

    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)  # (m_apt, n_samples)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds,
                                std=True)  # (m_apt, n_samples)

    # add the bounds zero and infinity
    phys_bounds = get_standard_physical_bounds(2)  # (n_samples, 2)

    # get the bounds on the reads

    r_bounds = get_r_bounds_measured(read_avgs, read_stds, 2.4841)  # (m_apt, 2, n_samples)
    affine_bounds = get_affine_bounds(r_bounds)  # (m_apt, 2, n_samples)

    # calculate upper and lower bounds
    bounds_full_model = apply_solver_parallel(K_A, affine_bounds, phys_bounds,
                                              lower_upper_bounds_solver,
                                              n_cores=10)  # (n_targets, x_solution, n_samples)

    off_diag_mat = (np.ones_like(K_A) - np.diag(np.diag(np.ones_like(K_A))))
    K_A = off_diag_mat * 1e-9 + np.diag(np.diag(K_A))  # get main diagonal
    # diag_mat = np.ones_like(K_A) - off_diag_mat
    # K_A = K_A - np.diag(np.diag(K_A)) + diag_mat * 1e-9  # get off-diagonal

    bounds_naive_model = apply_solver_parallel(K_A, affine_bounds, phys_bounds,
                                               lower_upper_bounds_solver,
                                               n_cores=10)  # (n_targets, x_solution, n_samples)
    # Plot kyn quant for concentration range of increasing xa
    fig, ax = plt.subplots(1, 1)
    plot_CIs2(metadata, ax, concs, bounds_full_model, bounds_naive_model,
             0, 1, same_target=False)
    ax.set_xlim([1e-4, 1e-2])
    plt.grid()
    fig.set_layout_engine('compressed')
    plt.gcf().set_size_inches(6.5, 3)
    save_figure(fig, os.path.join(root_directory, 'output', 'kyn_vs_xa.svg'))
    plt.show()
    plt.close(fig)

    # Plot xa quant for concentration range of increasing xa
    fig, ax = plt.subplots(1, 1)
    plot_CIs(metadata, ax, concs, bounds_full_model, bounds_naive_model,
             0, 1, same_target=True)
    save_figure(fig, os.path.join(root_directory, 'output', 'xa_vs_xa.svg'))
    plt.close(fig)
