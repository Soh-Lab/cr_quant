import json

import numpy as np
import pandas as pd


def read_metadata_json(json_file) -> dict:
    """
    Reads metadata from a JSON file.

    :param json_file: Path to the JSON file containing normalizers.
    :return: List of normalizers as tuples.
    """
    return json.load(open(json_file))


def read_data(path: str, metadata: dict) -> pd.DataFrame:
    """
    Reads the data, removes NaNs and normalizes any concentrations to be given in M.

    Adapted from code by Sharon S. Newman.

    :param path: Path to the data.
    :param metadata: Metadata object for the data.

    :return: Pandas dataframe of the real-world data.
    """
    df = pd.read_csv(path)
    df = df.dropna()
    # Convert all concentrations to M
    for target in metadata['targets']:
        df[target['name']] = df[target['old_name']] * target['factor']
        df = df.drop(columns=[target['old_name']])
    return df


def convert_dataframe_to_numpy(dataframe: pd.DataFrame, metadata: dict) \
        -> tuple[np.array, np.array]:
    """
    Convert dataformat from a Pandas dataframe of real-world data to numpy arrays.

    :param dataframe: Pandas dataframe of the real-world data.
    :param metadata: Metadata object for the data.

    :return: (n_targets, k) concentrations for each experimental condition,
        (m_affinity, k) recorded signal readouts of all affinity reagents. Each replicate
        of a condition is stored in a separate row.
    """
    concentrations = np.stack([dataframe[target['name']] for target in metadata['targets']], axis=0)
    readouts = np.stack([dataframe[reagent['name']] for reagent in metadata['reagents']], axis=0)

    return concentrations, readouts


def convert_dataframe_to_avg_std(dataframe: pd.DataFrame, metadata: dict) \
        -> tuple[np.array, np.array, np.array]:
    """
    Summarize a Pandas dataframe of real-world data by grouping identical experimental conditions
    and calculating the mean and standard deviation for each condition.

    :param dataframe: Pandas dataframe of the real-world data.
    :param metadata: Metadata object for the data.
    :return: (n_targets, k) concentrations of each experimental condition,
        (m_affinity, k) average signal readout of all affinity reagents,
        (m_affinity, k) estimate of standard deviation of the signal readout of all affinity
         reagents.
    """
    # group by experimental conditions
    grouped = dataframe.groupby([target['name'] for target in metadata['targets']])
    # calculate average and standard deviation
    grouped = grouped.agg({reagent['name']: ['mean', 'std'] for reagent in metadata['reagents']})

    # convert to numpy arrays
    readout_avgs = grouped[[(reagent['name'], 'mean') for reagent in metadata['reagents']]].values
    readout_stds = grouped[[(reagent['name'], 'std') for reagent in metadata['reagents']]].values
    concentrations = np.vstack(grouped.index.values)

    return concentrations.T, readout_avgs.T, readout_stds.T


def read_data_files(metadata_path: str, data_path: str) -> tuple[dict, pd.DataFrame]:
    """
    Load and return all the information related to a real-world experiment.

    :param metadata_path: Path to the metadata file (.json file).
    :param data_path: Path to the real world measurements file (.csv file).

    :return: tuple of metadata object, and Pandas dataframe containing the real-world data.
    """
    metadata = read_metadata_json(metadata_path)
    df = read_data(data_path, metadata)
    return metadata, df


if __name__ == '__main__':
    # load data
    metadata = read_metadata_json('../data/2023_05_22_CR8.json')
    df = read_data('../data/2023_05_22_CR8_combined_2colreads.csv',
                   metadata)
    df = df[~df.singleplex]
    print(df)