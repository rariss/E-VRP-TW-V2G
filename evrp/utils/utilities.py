import pandas as pd


def parse_csv_tables(fpath: str,
                     table_names: tuple = ('D', 'S', 'M', 'Parameters', 'W', 'T')) -> dict:
    """Parses an csv with multiple tables into a dictionary of pandas DataFrames for each table.

    :param fpath: csv file path
    :type fpath: str
    :param table_names: list of table names in CSV
    :type table_names: tuple
    :return: dictionary of table_names as key, pd.DataFrames for each table as values
    :rtype: dict
    """
    # Import tabublated CSV
    df = pd.read_csv(fpath, header=None)

    # Check if first column values match table_names
    groups = df[0].isin(table_names).cumsum()

    # Parse out tables by projecting forward from the table_names, dropping columns and rows that are all NaN
    tables = {g.iloc[0, 0]: g.iloc[1:].dropna(axis=0, how='all').dropna(axis=1, how='all') for k, g
              in df.groupby(groups)}

    for k, table in tables.items():
        table.rename(columns={0: table.iloc[0, 0]}, inplace=True)
        table.set_index(table.columns[0], drop=True, inplace=True)

        if k in ['T']:
            tables[k] = table.T.set_index(pd.MultiIndex.from_arrays(table.iloc[0:2].values)).T.drop(
                table.index[0:2])
            tables[k].index = tables[k].index.astype(int)
        else:
            tables[k] = table.rename(columns=table.iloc[0]).drop(table.index[0])

    return tables
