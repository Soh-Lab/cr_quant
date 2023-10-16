"""


2022-12-10 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import t as t_dist

from cr_utils.plotting import plot_2d_fields, apply_paper_formatting
from cr_utils.solvers import apply_solver_parallel, boundedness_solver, lower_upper_bounds_solver
from cr_utils.utils import get_standard_physical_bounds, get_readouts, get_r_bounds, \
    get_affine_bounds

if __name__ == '__main__':
    apply_paper_formatting(18)
    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    # log_from, log_to = -6, -1
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300

    K_D = np.array([
        [10. ** -2.704, 10. ** -3.882],
        [10. ** -0.578, 10. ** -3.192]
    ])

    K_A = 1.0 / K_D

    phys_bounds = get_standard_physical_bounds(2)

    # create a 2D-log-meshgrid of target concentrations
    A, B = np.logspace(-4, -1, log_steps), np.logspace(-6, -3, log_steps)
    AA, BB = np.meshgrid(A, B)  # (log_steps, log_steps)

    # generate the affine bounds needed to run the solver
    target_concs = np.stack([AA, BB], axis=0)  # (n_targets=2, log_steps, log_steps)
    readouts = get_readouts(K_A, target_concs)  # (m_affinity=2, log_steps, log_steps)

    SD = np.sqrt(np.square(readouts * 5.33827119e-02) + 8.67409439e-05)
    n_samples = 3
    factor = t_dist.ppf(0.975, n_samples - 1) / np.sqrt(n_samples)
    r_bounds = get_r_bounds(readouts, SD * factor, 0.0)  # (m_affinity=2, 2, log_steps, log_steps)
    affine_bounds = get_affine_bounds(r_bounds)  # (m_affinity=2, 2, log_steps, log_steps)

    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, boundedness_solver,
                                   n_cores=N_CORES)  # (n_targets=2, 1, log_steps, log_steps)

    # calculate for which indices we have non-infinite bounds
    t_is_bounded = bounds[:, 0, ...] == 0
    any_is_bounded = np.logical_or(t_is_bounded[0], t_is_bounded[1])

    resolution = np.ones((2, 2, log_steps, log_steps))

    result_flat = apply_solver_parallel(K_A, affine_bounds[:, :, any_is_bounded], phys_bounds,
                                        lower_upper_bounds_solver,
                                        n_cores=N_CORES)
    # (n_targets, result_index, log_steps, log_steps)
    resolution[:, :, any_is_bounded] = result_flat
    # (n_targets, log_steps, log_steps)
    resolution = np.log10(resolution[:, 1]) - np.log10(resolution[:, 0])
    worst_performance = np.nanmax(resolution) + 1
    resolution[0, ~t_is_bounded[0]] = worst_performance
    resolution[1, ~t_is_bounded[1]] = worst_performance

    axs = plot_2d_fields(target_concs, resolution, '', limits=(0.0, 3.0),colorbar_scale=1.0)

    for ax in axs:
        ax.set_aspect('equal')
        ax.set_ylabel('xa [M]')
        ax.set_xlabel('kyn [M]')
        ax.set_title('')

    axs[1].remove()
    axs[0].set_position([0.1, 0.1, 0.8, 0.8])

    # load data
    # metadata = read_metadata_json('/Users/linus/workspace/cr_quant/data/2023_05_22_CR8.json')
    # df = read_data('/Users/linus/workspace/cr_quant/data/2023_06_20_colreads_.csv',
    #                metadata)

    # concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)
    # inds = concs[0] < 2e-3
    # for ax in axs:
    #     ax.scatter(concs[0,inds], concs[1,inds], color='r', marker='x')
    # plt.gcf().set_size_inches(8.6, 4.3)
    # plt.gcf().set_layout_engine('compressed')
    # save_figure(plt.gcf(), '/Users/linus/workspace/cr_quant/poster/res.svg')
    plt.show()
