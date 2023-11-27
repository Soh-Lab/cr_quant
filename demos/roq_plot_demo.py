"""
Code used to generate Figures 3ajkl.

2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t as t_dist

from applications.data_handling import read_metadata_json, read_data, convert_dataframe_to_avg_std
from cr_utils.plotting import plot_2d_fields, apply_paper_formatting, save_figure
from cr_utils.solvers import apply_solver_parallel, boundedness_solver, lower_upper_bounds_solver
from cr_utils.utils import get_standard_physical_bounds, get_readouts, get_r_bounds, \
    get_affine_bounds

if __name__ == '__main__':
    apply_paper_formatting(18)
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 100
    # set the noise terms (constants generated from fit_CV_bg.py)
    CV, bg = 5.33827119e-02, 8.67409439e-05
    n_samples = 3  # number of sample replicates
    percentage_CI = 95  # confidence interval percentage

    K_D = np.array([  # example: affinities from the paper
        [10. ** -2.704, 10. ** -3.882],  # affinity reagent 1
        [10. ** -0.578, 10. ** -3.192]  # affinity reagent 2
    ])

    K_A = 1.0 / K_D

    phys_bounds = get_standard_physical_bounds(2)

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(-4, -1, log_steps), np.logspace(-6, -3, log_steps)
    AA, BB = np.meshgrid(A, B)  # shape: (log_steps, log_steps)

    # generate the affine bounds needed to run the solver
    target_concs = np.stack([AA, BB], axis=0)  # shape: (n_targets=2, log_steps, log_steps)
    readouts = get_readouts(K_A, target_concs)  # shape: (m_affinity=2, log_steps, log_steps)
    # estimate noise profile
    SD = np.sqrt(np.square(readouts * CV) + bg)
    factor = t_dist.ppf(1 - (1 - percentage_CI / 100.) / 2, n_samples - 1) / np.sqrt(n_samples)
    # shape: (m_affinity=2, 2, log_steps, log_steps)
    r_bounds = get_r_bounds(readouts, SD * factor, 0.0)
    affine_bounds = get_affine_bounds(r_bounds)  # shape: (m_affinity=2, 2, log_steps, log_steps)

    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, boundedness_solver,
                                   n_cores=N_CORES)  # shape: (n_targets=2, 1, log_steps, log_steps)

    # calculate for which indices we have non-infinite bounds
    t_is_bounded = bounds[:, 0, ...] == 0
    any_is_bounded = np.logical_or(t_is_bounded[0], t_is_bounded[1])

    resolution = np.ones((2, 2, log_steps, log_steps))

    # shape: (n_targets, result_index, log_steps, log_steps)
    result_flat = apply_solver_parallel(K_A, affine_bounds[:, :, any_is_bounded], phys_bounds,
                                        lower_upper_bounds_solver,
                                        n_cores=N_CORES)
    resolution[:, :, any_is_bounded] = result_flat  # shape: (n_targets, log_steps, log_steps)

    resolution = np.log10(resolution[:, 1]) - np.log10(resolution[:, 0])
    worst_performance = np.nanmax(resolution) + 1
    resolution[0, ~t_is_bounded[0]] = worst_performance
    resolution[1, ~t_is_bounded[1]] = worst_performance

    axs = plot_2d_fields(target_concs, resolution, '', limits=(0.0, 3.0))

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_ylabel('xa (M)')
        ax.set_xlabel('kyn (M)')
        ax.set_title('')

    # load data
    metadata = read_metadata_json('../data/2023_05_22_CR8.json')
    df = read_data('../data/2023_06_20_colreads_filtered.csv', metadata)

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)
    for ax in axs:
        ax.scatter(concs[0], concs[1], color='r', marker='x', zorder=2.)

    save_figure(plt.gcf(), '../output/roq.svg')
    plt.show()
