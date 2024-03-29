"""
Demonstrate how to write a solver that inscribes an ellipsoid inside the feasible region,
which could give some intuition for the covariance of the feasible set.

2023-06-02 Linus A. Hein
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import t as t_dist

from applications.data_handling import convert_dataframe_to_numpy, convert_dataframe_to_avg_std, \
    read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_ellipse, plot_2d_fields, plot_feasible_region
from cr_utils.solvers import ellipsoid_solver, apply_solver
from cr_utils.utils import get_readouts, get_affine_bounds, get_standard_physical_bounds, \
    get_r_bounds_measured

if __name__ == '__main__':
    # load data
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_filtered.csv'

    # number of processors to use to generate the plot
    N_CORES = 10
    # set the bounds of the log-scale [10^log_from, 10^log_to]
    log_from, log_to = -8, 0
    # set the resolution on each axis of the log-scale (reduce this number to get faster runtime)
    log_steps = 300
    colors = ['r', 'orange', 'b']

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

    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds, std=True)

    for ind in range(concs.shape[1]):  # iterate over every experimental condition
        true_conc = concs[:, ind:ind + 1]

        # plotting background
        m_reagents = K_D.shape[0]
        # create a 2D-log-meshgrid of target concentrations
        A, B = np.logspace(log_from, log_to, log_steps), np.logspace(log_from, log_to, log_steps)
        AA, BB = np.meshgrid(A, B)
        target_concs = np.stack([AA, BB], axis=0)
        readouts = get_readouts(K_A, target_concs)

        # preparing plotting
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2, m_reagents, figure=fig)
        combined_ax = fig.add_subplot(gs[1, m_reagents // 2])
        axs = [fig.add_subplot(gs[0, i], sharex=combined_ax, sharey=combined_ax) for i in
               range(m_reagents)]

        plot_2d_fields(target_concs, readouts, 'Affinity Reagent', axs=axs)

        readouts = read_avgs[:, ind:ind + 1]
        readout_stds = read_stds[:, ind:ind + 1]

        # calculate upper and lower affine bounds
        n_samples = 3
        factor = t_dist.ppf(0.975, n_samples - 1) / np.sqrt(n_samples)
        r_bounds = get_r_bounds_measured(readouts, readout_stds, factor)
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
            axs[i].set_aspect('equal')
        combined_ax.grid()
        combined_ax.set_aspect('equal')
        fig.set_size_inches(3 * m_reagents, 6)

        # plotting ellipse
        phys_bounds = get_standard_physical_bounds(2)
        results = apply_solver(K_A, affine_bounds, phys_bounds, ellipsoid_solver)

        print(f'True concentrations: {true_conc[0, 0]:.2e}, {true_conc[1, 0]:.2e}, Estimated concentrations: {results[0, 0, 0]:.2e}, {results[1, 0, 0]:.2e}')
        plot_ellipse(results[:, 0, 0], results[:, 1:, 0], combined_ax)
        # mark centroid using blue o, and true concentration using red x
        combined_ax.scatter(results[0, 0, 0], results[1, 0, 0], color='b', marker='o')
        combined_ax.scatter(true_conc[0, 0], true_conc[1, 0], color='r', marker='x')
        plt.show()
