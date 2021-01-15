import pandas as pd
import numpy as np
from scipy.spatial import distance
import datetime
import logging
import pprint

now = datetime.datetime.now()


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

    :param df: Vx2 matrix for V vertices with (x,y) coordinates
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


def create_flat_distance_matrix(d: 'pd.DataFrame') -> 'pd.DataFrame':
    """
    Creates a flat distance matrix from a square distance matrix.

    :param d: square distance matrix
    :type d: pd.DataFrame
    :return: flat distance matrix
    :rtype: pd.DataFrame
    """
    # Stacks columns as new rows
    dflat = d.stack()

    # relabel multiindex names to "from" and "to" nodes
    dflat.index.set_names(['from', 'to'], inplace=True)

    # Reset index and rename the distance column "d"
    dflat = dflat.reset_index()
    dflat.rename({0: 'd'}, axis=1, inplace=True)
    return dflat


def create_coordinate_dict(v: 'pd.DataFrame') -> dict:
    """
    Creates a dictionary mapping node_id keys to their array(x, y) coordinates.

    :param v: dataframe of nodes and their coordinates
    :type v: pd.DataFrame
    :return: dictionary mapping node_ids to coordinates
    :rtype: dict
    """
    return dict(zip(v.index, v[['d_x', 'd_y']].values))


def create_plotting_edges(v: 'pd.DataFrame', d: 'pd.DataFrame') -> 'np.array':
    """
    Creates a 2D np.array of edges betwee provided from and to nodes, with (None, None) indices
    creating the disconnection between edges in the array.

    :param v: dataframe of nodes and their coordinates
    :type v: pd.DataFrame
    :param d: square distance matrix
    :type d: pd.DataFrame
    :return: 2*Vx2 np.array of edge coordinates
    :rtype: np.array
    """
    d_flat = create_flat_distance_matrix(d)
    coordinates = create_coordinate_dict(v)

    edges = np.concatenate([np.vstack((coordinates[a], coordinates[b], [None, None])) for a, b in d_flat[['from', 'to']].values])
    return edges


def instance_to_df(instance, **kwargs):
    logFile = open('results/' + now.strftime('%y%m%d-%H%M%S') + '_gurobi_' + instance.instance_name + '.txt', 'a+')

    logging.info('===============Summary===============')
    logging.info('Total distance traveled:     {}'.format(round(instance.m.obj, 2)))
    # logging.info(instance.m.pprint())
    instance.m.pprint(logFile)

    dfs = []

    # Get scalars
    scalars = {str(v): v.value for v in instance.component_data_objects(Var) if v.index() is None}
    summary = {'obj_net_profit': -round(value(obj_net_profit(instance)),2),
               'RD': round(value(RD(instance)),2),
               'RE': round(value(RE(instance)),2),
               'Ccap': round(value(Ccap(instance)),2),
               'Cop': round(value(Cop(instance)),2),
               'net_profit_truck': -round(value(obj_net_profit(instance))/value(instance.xN),2)}
    scalars.update(summary)
    dfs.append(pd.Series(scalars))

    # Create I set
    df = pd.DataFrame([floor(t/instance.tI) for t in instance.T.data()], columns=['I'])
    df['xd'] = [getattr(instance, 'xd').extract_values()[floor(t/instance.tI)] for t in instance.T.data()]
    df['IP'] = df['xd'] * -value(IP(instance))

    for n in ['xg', 'xb', 'E', 'P_E']:
        df[n] = df.index.map(getattr(instance, n).extract_values())

    dfs.append(df)
    logFile.close()
    return dfs


def create_flat_optimal_edges(m: 'obj', tol=1e-4, **kwargs) -> 'pd.DataFrame':
    if 'graph' in kwargs.keys():
        graph = kwargs['graph']
    else:
        graph = 'V'

    c = create_coordinate_dict(m.data[graph])

    if 'x' in kwargs.keys():
        e = kwargs['x']
    else:
        e = pd.DataFrame([(*k, v()) for k, v in m.instance.xgamma.items()])
        if len(e.columns) == 3:
            col_renames = {0: 'from', 1: 'to', 2: 'state'}
        elif len(e.columns) == 4:
            col_renames = {0: 'vehicle', 1: 'from', 2: 'to', 3: 'state'}
        else:
            logging.error('Check size of xgamma columns. Must be 3 (if not vehicle indexed) or 4 (if vehicle indexed)')
        e.rename(col_renames, axis=1, inplace=True)

    e['from_d_x'], e['from_d_y'] = np.vstack([c[e] for e in e['from']]).T
    e['to_d_x'], e['to_d_y'] = np.vstack([c[e] for e in e['to']]).T
    e = e[e['state'] > tol]

    return e


def create_optimal_edges(m: 'obj', **kwargs) -> 'pd.DataFrame':
    e_flat = create_flat_optimal_edges(m, **kwargs)

    return e_flat.pivot(index='from', columns='to', values='state'), e_flat


def merge_variable_results(m: 'obj', var_list: str, include_vehicle=False) -> 'pd.DataFrame':
    v0 = var_list[0]
    v = v0
    var_list.remove(v)
    x = getattr(m.instance, v)

    keys, values = zip(*x.get_values().items())
    idx = pd.MultiIndex.from_tuples(keys)
    x = pd.DataFrame(values, index=idx)
    x.columns = [v]
    x.reset_index(inplace=True)
    if include_vehicle and (v0 == 'xgamma'):
        x.columns = ['vehicle', 'from', 'to', 'state']
    elif v0 == 'xgamma':
        x.columns = ['from', 'to', 'state']
    elif v0 == 'xkappa':
        x.columns = ['node', 't', 'state']
    else:
        logging.error('Must provide either "xgamma" or "xkappa" as first value in var_list.')
        AttributeError

    for v in var_list:
        x_right = getattr(m.instance, v)
        keys, values = zip(*x_right.get_values().items())
        if (not include_vehicle) & (v0 == 'xgamma'):
            idx = keys
        else:
            idx = pd.MultiIndex.from_tuples(keys)
        x_right = pd.DataFrame(values, index=idx)
        x_right.columns = [v]
        x_right.reset_index(inplace=True)
        if include_vehicle:
            x_right.rename(columns={'level_0': 'vehicle', 'level_1': 'to'}, inplace=True)
            x = pd.merge(x, x_right, how='inner', on=['vehicle', 'to'])
        elif v0 == 'xgamma':
            x_right.rename(columns={'index': 'to'}, inplace=True)
            x = pd.merge(x, x_right, how='inner', on=['to'])
        elif v0 == 'xkappa':
            x_right.rename(columns={'level_0': 'node', 'level_1': 't'}, inplace=True)
            x = pd.merge(x, x_right, how='inner', on=['node', 't'])
    return x


def trace_routes_util(from_node, end_node, visited, current_route, routes, graph):
    current_route.append(from_node)

    # if from_node == end_node:
    #     routes.append(tuple(current_route))

    if end_node in from_node:
        routes.append(tuple(current_route))

    else:

        for n in graph[from_node]:
            if n not in visited:
                trace_routes_util(n, end_node, visited, current_route, routes, graph)

    current_route.pop()

    return routes


def trace_routes(m: 'obj', tol=1e-4):
    x_val = getattr(m.instance, 'xgamma').extract_values()
    active_arcs = [n for n in list(x_val.items()) if n[1] > tol]
    active_arcs_list = list(list(zip(*active_arcs))[0])

    active_arcs_dict = {}  # dict of active routes as a tree representing routes
    for a, b in active_arcs_list:
        active_arcs_dict.setdefault(a, []).append(b)

    start_node = m.instance.start_node.value_list[0]
    end_node = m.instance.end_node.value_list[0]

    visited = set()

    current_route = []
    routes = []

    routes = trace_routes_util(start_node, end_node, visited, current_route, routes,
                               active_arcs_dict)

    return routes