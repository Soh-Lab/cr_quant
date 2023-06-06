import numpy as np
import pandas as pd

import json
import os


def read_metadata_json(json_file):
    """
    Reads metadata data from a JSON file.

    :param json_file: Path to the JSON file containing normalizers.
    :return: List of normalizers as tuples.
    """
    return json.load(open(json_file))


def read_data(path: str,
              metadata):
    """
    Reads the data, removes NaNs and normalizes any concentrations to be given in M.

    Adapted from code by Sharon S. Newman.

    :param path: Path to the data.
    :param normalizers: List of normalizations to apply to the data.
        List elements should be of format: (col_name_old, col_name_new, factor).

    :return: pandas dataframe of the real-world data.
    """
    df = pd.read_csv(path)
    df = df.dropna()
    # Convert all concentrations to M
    for target in metadata['targets']:
        df[target['name']] = df[target['old_name']] * target['factor']
        df = df.drop(columns=[target['old_name']])
    return df


def convert_dataframe_to_numpy(dataframe, metadata):
    concentrations = np.stack([dataframe[target['name']] for target in metadata['targets']], axis=0)
    readouts = np.stack([dataframe[reagent['name']] for reagent in metadata['reagents']], axis=0)

    return concentrations, readouts


def convert_dataframe_to_avg_std(dataframe, metadata):
    grouped = dataframe.groupby([target['name'] for target in metadata['targets']])
    grouped = grouped.agg({reagent['name']: ['mean', 'std'] for reagent in metadata['reagents']})

    readout_avgs = grouped[[(reagent['name'], 'mean') for reagent in metadata['reagents']]].values
    readout_stds = grouped[[(reagent['name'], 'std') for reagent in metadata['reagents']]].values

    concentrations = np.vstack(grouped.index.values)

    return concentrations.T, readout_avgs.T, readout_stds.T


def read_data_files(meta_file_name, data_file_name):
    directory_name = os.path.dirname(__file__)
    meta_data_file_location = os.path.join(directory_name, os.pardir, meta_file_name)
    data_file_location = os.path.join(directory_name, os.pardir, data_file_name)
    metadata = read_metadata_json(meta_data_file_location)
    df = read_data(data_file_location, metadata)
    return metadata, df

if __name__ == '__main__':
    # load data
    metadata = read_metadata_json('/Users/linus/workspace/cr_quant/data/2023_05_22_CR8.json')
    df = read_data('/Users/linus/workspace/cr_quant/data/2023_05_22_CR8_combined_2colreads.csv',
                   metadata)
    df = df[~df.singleplex]
