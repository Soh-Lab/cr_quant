"""


2023-10-16 Linus A. Hein
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from applications.data_handling import convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_lower_upper_performance
from cr_utils.solvers import lower_upper_bounds_solver, apply_solver_parallel
from cr_utils.utils import get_affine_bounds, \
    get_standard_physical_bounds, get_r_bounds_measured, get_readouts


def variance_estimator(mean, CV, bg_var):
    return np.square(mean * CV) + bg_var


if __name__ == '__main__':
    # load data
    metadata_name = '2023_05_22_CR8.json'
    data_name = '2023_06_20_colreads_.csv'

    root_directory = os.path.join(os.path.dirname(__file__), os.pardir)
    data_folder = os.path.join(root_directory, 'data')
    metadata, df = read_data_files(os.path.join(data_folder, metadata_name),
                                   os.path.join(data_folder, data_name))

    # isolate singleplex/binding curve measurements
    concs, reads = convert_dataframe_to_numpy(df[df.singleplex], metadata)

    # fit KD values
    K_D, lower_bounds, upper_bounds, \
        K_D_matrix_std, lower_bounds_std, upper_bounds_std = fit_multi_KD(concs, reads)
    K_A = 1.0 / K_D

    # get statistics over replicates
    concs, read_avgs, read_stds = convert_dataframe_to_avg_std(df[df.singleplex], metadata)

    # normalize the statistics
    read_avgs = normalize_reads(read_avgs, lower_bounds, upper_bounds)  # (m_apt, n_samples)
    read_stds = normalize_reads(read_stds, lower_bounds, upper_bounds,
                                std=True)  # (m_apt, n_samples)
    read_vars = np.square(read_stds)

    # fit CV and bg
    p0 = (0.05, 0.001)
    popt = curve_fit(variance_estimator, read_avgs.flatten(), read_vars.flatten(), p0,
                     bounds=([0, 0], [1, 1]))

    # print results
    print(f'CV = {popt[0][0]:.3e}')
    print(f'bg_var = {popt[0][1]:.3e}')

    # plot the fit
    plt.scatter(read_avgs[0], read_vars[0], label="SK1")
    plt.scatter(read_avgs[1], read_vars[1], label="XA1")
    readouts = np.linspace(0, 1.0, 1000)
    plt.plot(readouts, variance_estimator(readouts, *popt[0]), label=f'$Var=(\mu\cdot {popt[0][0]:.3e})^2+{popt[0][1]:.3e}$')
    plt.xlabel('Norm. Readout')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid()
    plt.show()
