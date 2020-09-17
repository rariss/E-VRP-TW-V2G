import pandas as pd
import numpy as np
from scipy.spatial import distance


def parse_csv_tables(filepath: str,
                     table_names: tuple = ('D', 'S', 'M', 'Parameters', 'W', 'T')) -> dict:
    """
    Parses an csv with multiple tables into a dictionary of pandas DataFrames for each table.

    :param filepath: csv file path
    :type filepath: str
    :param table_names: list of table names in CSV
    :type table_names: tuple
    :return: dictionary of table_names as key, pd.DataFrames for each table as values
    :rtype: dict
    """
    # Import tabulated CSV
    df = pd.read_csv(filepath, header=None)

    # Check if first column values match table_names
    groups = df[0].isin(table_names).cumsum()

    # Parse out tables by projecting forward from the table_names, dropping columns and rows that
    # are all NaN
    tables = {g.iloc[0, 0]: g.iloc[1:].dropna(axis=0, how='all').dropna(axis=1, how='all') for k, g
              in df.groupby(groups)}

    # Iterate through tables to form dictionary and reformat df
    for k, table in tables.items():

        # Set first column as new df index
        table.rename(columns={0: table.iloc[0, 0]}, inplace=True)
        table.set_index(table.columns[0], drop=True, inplace=True)

        # Create new column names base of first row (and second multiindex row for time series
        # tables)
        if k in ['T']:
            tables[k] = table.T.set_index(pd.MultiIndex.from_arrays(table.iloc[0:2].values)).T.drop(
                table.index[0:2])
            tables[k].index = tables[k].index.astype(int)
        else:
            tables[k] = table.rename(columns=table.iloc[0]).drop(table.index[0])

        # Convert anything dtypes to numeric if possible
        tables[k] = tables[k].apply(pd.to_numeric, errors='ignore')

    return tables


def calculate_distance_matrix(df: 'pd.DataFrame', metric: str = 'euclidean') -> 'pd.DataFrame':
    """
    Calculates a square distance matrix given a set of vertices with (x,y) coordinates.

    :param df: Mx2 matrix for M vertices with (x,y) coordinates
    :type df: np.array
    :param metric: scipy distance metric, default is 'euclidean'
    :type metric: str
    :return: Square matrix of distances between every vertex.
    :rtype: pd.DataFrame
    """
    # Sets columns and index to original vertex index
    return pd.DataFrame(distance.cdist(df.values, df.values, metric), index=df.index, columns=df.index)


def generate_index_mapping(v: 'ndarray') -> [dict, dict]:
    """
    Generates two dictionaries, one mapping from the given values to 0-indexed integers,
    and the inverse mapping.

    :param v: list of values to be mapped
    :type v: list
    :return: index-to-value and value-to-index dicts
    :rtype: [dict, dict]
    """
    i2v = dict(enumerate(v))
    v2i = dict(zip(v, range(len(v))))
    return i2v, v2i
