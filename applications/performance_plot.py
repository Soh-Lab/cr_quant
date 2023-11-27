"""
Generate data for tables S4, S5.

2022-12-10 Linus A. Hein
"""
import os

import numpy as np
import pandas as pd

from applications.data_handling import convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_lower_upper_performance
from cr_utils.solvers import lower_upper_bounds_solver, apply_solver_parallel
from cr_utils.utils import get_affine_bounds, \
    get_standard_physical_bounds, get_r_bounds_measured

if __name__ == '__main__':
    # load data
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_filtered.csv'

    # whether to assume that every affinity reagent is perfectly specific (naive approach)
    # set to true for data like in Fig S4.
    NAIVE = False

    root_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    data_folder = os.path.join(root_directory, 'data')
    metadata, df = read_data_files(os.path.join(data_folder, metadata_name),
                                   os.path.join(data_folder, data_name))

    # fit KD values
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)

    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    if NAIVE:
        K_A = np.diag(np.diag(K_A))
        K_A[K_A == 0] = 1e-9

    # off_diag_mat = (np.ones_like(K_A) - np.diag(np.diag(np.ones_like(K_A))))
    # diag_mat = np.ones_like(K_A) - off_diag_mat
    # K_A = off_diag_mat * 1e-9 + np.diag(np.diag(K_A))  # get main diagonal
    # K_A = K_A - np.diag(np.diag(K_A)) + diag_mat * 1e-9  # get off-diagonal

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
    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, lower_upper_bounds_solver,
                                   n_cores=10)  # (n_targets, x_solution, n_samples)

    d = {}
    for target_ind, target in enumerate(metadata['targets']):
        d[target['name']] = []
        d[target['name'] + '_lower'] = []
        d[target['name'] + '_upper'] = []

    for sample_ind in range(concs.shape[1]):
        sample_concs = concs[:, sample_ind]
        print('-' * 30)
        for target_ind, target in enumerate(metadata['targets']):
            d[target['name']].append(sample_concs[target_ind])
            d[target['name'] + '_lower'].append(bounds[target_ind, 0, sample_ind])
            d[target['name'] + '_upper'].append(bounds[target_ind, 1, sample_ind])
            print(
                f'{target["display_name"]}: [{np.log10(bounds[target_ind, 0, sample_ind]):.2f}; '
                f'{np.log10(bounds[target_ind, 1, sample_ind]):.2f}]'
                f'\t true: {np.log10(sample_concs[target_ind]):.2f}')
    df = pd.DataFrame(data=d)
    csv_path = os.path.join(root_directory, 'output', 'bounds.csv')
    df.to_csv(csv_path, index=False)
    # plot the results
    plot_lower_upper_performance(bounds, concs)
