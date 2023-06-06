"""


2022-12-10 Linus A. Hein
"""

from applications.data_handling import read_data, read_metadata_json, convert_dataframe_to_numpy, \
    convert_dataframe_to_avg_std, read_data_files
from applications.fit_multi_KD import fit_multi_KD, normalize_reads
from cr_utils.plotting import plot_lower_upper_performance
from cr_utils.solvers import lower_upper_bounds_solver, apply_solver_parallel
from cr_utils.utils import get_affine_bounds, \
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

    # add the bounds zero and infinity
    phys_bounds = get_standard_physical_bounds(2)

    # get the bounds on the reads

    r_bounds = get_r_bounds_measured(read_avgs, read_stds, 3)
    affine_bounds = get_affine_bounds(r_bounds)

    # calculate upper and lower bounds
    bounds = apply_solver_parallel(K_A, affine_bounds, phys_bounds, lower_upper_bounds_solver,
                                   n_cores=10)

    # plot the results
    plot_lower_upper_performance(bounds, concs)
