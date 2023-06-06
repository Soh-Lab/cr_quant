"""


2023-06-02 Linus A. Hein
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from applications.data_handling import read_data, read_metadata_json, convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_lower_upper_performance, plot_ellipse, plot_2d_fields, \
    plot_feasible_region
from cr_utils.solvers import apply_solver_parallel, ellipsoid_solver, apply_solver
from cr_utils.utils import get_readouts, get_r_bounds, get_affine_bounds, \
    get_standard_physical_bounds, get_r_bounds_measured
import os

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

    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[~df.singleplex], metadata)

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -8, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300
    colors = ['r', 'orange', 'b']

    for ind in range(concs.shape[1]):  # iterate over every sample
        true_conc = concs[:, ind:ind+1]

        ### plotting background

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

        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        # calculate upper and lower bounds on the readout values (+- n * std)
        r_bounds = get_r_bounds_measured(readouts, readout_stds, 5)

        affine_bounds = get_affine_bounds(r_bounds)

        for i in range(m_reagents):
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
        combined_ax.grid()
        fig.set_size_inches(4 * m_reagents, 9)

        ### plotting ellipse
        phys_bounds = get_standard_physical_bounds(2)

        results = apply_solver(K_A, affine_bounds, phys_bounds, ellipsoid_solver)

        print(f'{true_conc[0, 0]}, {true_conc[1, 0]}, {results[0, 0, 0]}, {results[1, 0, 0]}')
        combined_ax.scatter(results[0, 0, 0], results[1, 0, 0], color='r', marker='x')
        plot_ellipse(results[:, 0, 0], results[:, 1:, 0], combined_ax)
        # plt.show()

        directory_name = os.path.dirname(__file__)
        fig_file_location = os.path.join(directory_name, os.pardir,
                                         f'output/ellipse/{int(true_conc[0] * 1e6)}_{int(true_conc[1] * 1e6)}.png')
        plt.savefig(fig_file_location, dpi=300)
        plt.close()
        # plot_lower_upper_performance(bounds, target_concs)
