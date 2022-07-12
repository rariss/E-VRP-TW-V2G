import logging
import datetime
import json
import pathlib
import googlemaps
import numpy as np
import pandas as pd

from pyomo.environ import *
from scipy.spatial import distance
import evrptwv2g.config.LOCAL_CONFIG as LOCAL_CONFIG

log = logging.getLogger('root')


class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types taken from `numpyencoder 0.3.0 <https://pypi.org/project/numpyencoder/>`_."""

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)


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
            tables[k].index = pd.to_numeric(tables[k].index, errors='coerce').astype(int)
        else:
            tables[k] = table.rename(columns=table.iloc[0]).drop(table.index[0])

        # Convert anything dtypes to numeric if possible
        tables[k] = tables[k].apply(pd.to_numeric, errors='ignore')

    return tables

def calculate_distance_matrix(df: 'pd.DataFrame', metric: str = 'euclidean', dist_type = 'scipy',
                              instance: str='', save: bool=False) -> 'pd.DataFrame':
    """
    Calculates a square distance matrix given a set of vertices with (x,y) coordinates.

    :param df: Vx2 matrix for V vertices with (x,y) coordinates
    :type df: np.array
    :param metric: scipy distance metric, default is 'euclidean'
    :type metric: str
    :return: Square matrix of distances between every vertex.
    :rtype: pd.DataFrame
    """
    # Calculate a traditional
    if dist_type == 'scipy':
        log.info('Using Scipy euclidian distances to generate distance matrix')
        # TODO: Add in euclidian distance given lat and longitudes
        # pd.DataFrame(self.data['V_'][['d_x', 'd_y']] < 0).any(axis=None) for checking if lat/lon negative
        # Sets columns and index to original vertex index
        dist_matrix = pd.DataFrame(distance.cdist(df.values, df.values, metric), index=df.index, columns=df.index)
    elif pathlib.Path(dist_type).suffix == '.csv':
        log.info(f'Using CSV distance matrix: {dist_type}')
        dist_matrix = pd.read_csv(dist_type, index_col=0)
    elif dist_type == 'googlemaps':
        log.info('Using Google Maps Distance API to generate distance matrix')
        # Import google maps client to use their distance matrix api
        gmaps = googlemaps.Client(key=LOCAL_CONFIG.GOOGLE_API_KEY)

        # TODO: Distance matrix has a limit of 100 server-side requests. For more than 10 unique locations, need to split by row and concatenate
        # Create a mapping from unique locations to codes
        codes, latlons = pd.factorize([(y, x) for x, y in df.values])

        # Query google for the distance matrix
        dist = gmaps.distance_matrix(origins=latlons, destinations=latlons, mode='driving')
        dist_df = pd.DataFrame(dist)

        # Normalize the elements of the distance matrix into a flat dataframe
        dist_matrix_df = pd.json_normalize(data=dist, record_path=['rows', 'elements'])

        # Reindex to google's origin and destination addresses
        dist_matrix_df.index = pd.MultiIndex.from_product(
            [dist_df['origin_addresses'], dist_df['destination_addresses']])

        # Turn into a square distance matrix
        m2km = 1 / 1000
        dist_square_df = pd.DataFrame(
            data=dist_matrix_df['distance.value'].values.reshape(len(dist_df['origin_addresses']), -1),
            index=dist_df['origin_addresses'], columns=dist_df['destination_addresses']) * m2km

        # Use codes to produce distance matrix results into the original matrix order from unique values
        output_df = dist_square_df.iloc[codes].T.iloc[codes]

        # Rename to original node index
        output_df.index = df.index
        output_df.columns = df.index

        dist_matrix = output_df

    # Save distance matrix to CSV
    if save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f'{LOCAL_CONFIG.DIR_OUTPUT}/{instance}_{timestamp}_{dist_type}.csv'
        dist_matrix.to_csv(save_path)
        log.info(f'Output distance matrix to CSV: {save_path}')
    return dist_matrix

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
            log.error('Check size of xgamma columns. Must be 3 (if not vehicle indexed) or 4 (if vehicle indexed)')
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
        log.error('Must provide either "xgamma" or "xkappa" as first value in var_list.')
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
            x = pd.merge(x, x_right, how='outer', on=['vehicle', 'to'])
        elif v0 == 'xgamma':
            x_right.rename(columns={'index': 'to'}, inplace=True)
            x = pd.merge(x, x_right, how='outer', on=['to'])
        elif v0 == 'xkappa':
            x_right.rename(columns={'level_0': 'node', 'level_1': 't'}, inplace=True)
            x = pd.merge(x, x_right, how='outer', on=['node', 't'])
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

    start_node = m.instance.start_node.ordered_data()[0]
    end_node = m.instance.end_node.ordered_data()[0]

    visited = set()

    current_route = []
    routes = []

    routes = trace_routes_util(start_node, end_node, visited, current_route, routes,
                               active_arcs_dict)

    return routes

def results_to_dfs(m: 'obj', tol=1e-4):
    # Merge power decisions
    xp_var_list = ['xkappa', 'xp']
    xp = merge_variable_results(m, xp_var_list)
    xp = xp[xp['state'] > tol].sort_values(['t'], ascending=True)

    # Merge other variables
    var_list = ['xgamma', 'xw', 'xq', 'xa']
    x = merge_variable_results(m, var_list)

    # Generate route traces (list of tuples)
    traces = trace_routes(m)

    # Generate route df (appended NaN row for D0 states)
    routes = x[x['state'] > tol].append(x[np.isnan(x['state'])]).sort_values(['xw']).set_index(['from', 'to'])

    # Create (D0, D0) row for starting states
    routes.reset_index(inplace=True)
    idx = routes['from'].isna()
    routes.loc[idx, 'from'] = routes.loc[idx, 'to']
    routes.set_index(['from', 'to'], inplace=True)

    return x, xp, traces, routes


def create_json_out(instance, results=None, output_file=None):
    """Passing in results through args will output model time and gap."""
    # Initialize json dict output
    data_out = {}
    data_out['name'] = instance.name

    # Read out variable results
    for v in instance.component_objects(Var):
        data_out[v.name] = {}
        for index in v:
            try:
                data_out[v.name][str(index)] = value(v[index])
            except:
                data_out[v.name][str(index)] = None

    # Read out objective result
    data_out['obj'] = instance.obj.expr()

    if results is not None:
        gap = results['Problem'][0]
        gap = (gap['Upper bound'] - gap['Lower bound']) / gap['Upper bound'] * 100
        data_out['gap'] = gap  # outputs gap as percent
        data_out['solve_time'] = results['Solver'][0]['Time']  # seconds

    # Generate json from dict
    json_outputs = json.dumps(data_out, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))

    # Output json
    if output_file is None:
        output_file = 'output'

    with open('{}.json'.format(output_file), 'w') as outfile:
        json.dump(data_out, outfile, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))
    return json_outputs


def create_json_inputs(inputs, output_file=None):
    """Outputs the input parameters dictionary to a json"""
    params = inputs.copy()
    # convert tuple keys to strings
    for k, v in params[None].items():
        if k in ['d', 'ce', 'G']:
            params[None][k] = {str(key): val for key, val in params[None][k].items()}

    json_inputs = json.dumps(params, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))

    # Output json
    if output_file is None:
        output_file = 'inputs'

    with open('{}.json'.format(output_file), 'w') as outfile:
        json.dump(json_inputs, outfile, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))
    return json_inputs


def read_instance_json_str(instance_json_str):
    instance_json = json.loads(instance_json_str)
    for key, val in instance_json.items():
        if key in ['xgamma', 'xkappa', 'xp']:
            instance_json[key] = {eval(k): v for k, v in instance_json[key].items()}
    return instance_json


def update_instance_json(var_list, new_concrete_instance, instance_json):
    # Read in a json string to a dict object
    if type(instance_json) is str:
        instance_json = read_instance_json_str(instance_json)

    for var in var_list:
        var_instance = getattr(new_concrete_instance, var)
        for key, val in instance_json[var].items():
            var_instance[key] = val
        setattr(new_concrete_instance, var, var_instance)
    return new_concrete_instance


def convert_txt_instances_to_csv(instance, folder=f'{LOCAL_CONFIG.DIR_INSTANCES}/test_instances/evrptw_instances/',
                                 output_folder=f'{LOCAL_CONFIG.DIR_INSTANCES}/test_instances'):
    """Converts a Schneider txt test instance into a csv format for EVRPTWV2G starting point."""
    # Default parameters
    defaults = {'N': [1], 'cc': [1000], 'co': [1], 'cm': [0], 'cy': [3e-3], 'EMIN': [0],
                'cg': 0, 'SMIN': 0, 'instances': 1, 'tQ': 0, 'cq': 1, 't_S': 1, 't_H': 1, 'G': 1, 'ce': 0.1, 'eff': 0.9
    }  # Must use lists if table generated from dict. Use single value if updating a dataframe value

    fpath = folder + instance + '.txt'

    data = pd.read_csv(fpath, sep='\s+', skip_blank_lines=False)
    i_split = data.isnull().all(axis=1).argmax()  # blank row

    graph = data.iloc[:i_split].copy()

    col_map = {'StringID': 'node_id', 'Type': 'node_type', 'x': 'd_x', 'y': 'd_y', 'demand': 'q',
               'ReadyTime': 'tA', 'DueDate': 'tB', 'ServiceTime': 'tS'}

    graph.rename(columns=col_map, inplace=True)

    other = pd.read_csv(fpath, sep='/', skip_blank_lines=False, skiprows=i_split + 1)

    other_dict = {}
    for (k, v, nan) in other.index:
        other_dict[k[0]] = v

    # Make vehicle table
    W = pd.DataFrame(data={'vehicle_type_id': ['EV'], 'N': defaults['N'],
                           'r': [other_dict['r']],
                           'v': [other_dict['v']],
                           'cc': defaults['cc'], 'co': defaults['co'], 'cm': defaults['cm'], 'cy': defaults['cy'],
                           'eff': defaults['eff'],
                           'QMAX': [other_dict['C']],
                           'EMIN': defaults['EMIN'],
                           'EMAX': [other_dict['Q']],
                           'PMAX': [1 / other_dict['g']]})

    # Make depot table
    D = graph[graph['node_type'] == 'd'].copy()
    D = pd.DataFrame(np.repeat(D.values, 2, axis=0), columns=D.columns)
    D.loc[1]['node_id'] = 'D1'
    D['node_description'] = ['start', 'end']
    cols = D.columns.tolist()
    D = D[cols[:2] + [cols[-1]] + cols[2:4] + cols[5:-1]]

    # Make station table
    S = graph[graph['node_type'] == 'f'].copy()
    S['node_type'] = 's'
    S['node_description'] = S['node_type']
    cols = S.columns.tolist()
    S = S[cols[:2] + [cols[-1]] + cols[2:4] + cols[5:-1]]
    S['cg'] = defaults['cg']
    S['SMIN'] = defaults['SMIN']
    S['SMAX'] = W['PMAX'].max()
    S['instances'] = defaults['instances']

    # Make customer table
    M = graph[graph['node_type'] == 'c'].copy()
    M['node_type'] = 'm'
    M['node_description'] = M['node_type']
    cols = M.columns.tolist()
    M = M[cols[:2] + [cols[-1]] + cols[2:4] + cols[5:-1] + [cols[4]]]
    M['tQ'] = defaults['tQ']
    M['cq'] = defaults['cq']

    # Make parameters table
    P = pd.DataFrame(data={'parameter': ['t_T', 't_S', 't_H'],
                           'description': ['optimization time horizon', 'time step size', 'hours per time step unit'],
                           'value': [graph['tB'].max(), defaults['t_S'], defaults['t_H']]
                           })

    # Make time table
    T_dict = {'t': {'': 0}}
    val = {'G': defaults['G'], 'ce': defaults['ce']}
    for k, v in val.items():
        T_dict[k] = {s: v for s in S['node_id']}
    T = pd.DataFrame.from_dict(T_dict, orient='index')
    T = pd.DataFrame(T.stack()).T
    T = T.append(T, ignore_index=True)
    T.loc[1, 't'] = int(graph['tB'].max())

    # Write and output file
    dict_of_dfs = {'D': D, 'S': S, 'M': M, 'Parameters': P, 'W': W, 'T': T}
    csv_rows = max(len(d.columns) for k, d in dict_of_dfs.items())
    with open(f'{output_folder}/{instance}_.csv', 'w+') as f:
        for k, df in dict_of_dfs.items():
            f.write(k + ','*csv_rows +'\n')
            df.to_csv(f, index=False, sep=',')
            f.write(','*csv_rows+'\n')

def generate_stats(m):
    """
    Calculates output statistics
    :param m:
    :return:
    """
    stats = {
        # INPUTS
        'MQ_payload_constraints': m.instance.MQ.value,
        'MT_service_time_constraints': m.MT,
        'ME_energy_constraints': m.ME,
        'cc_capital_cost': m.cc,
        'co_operating_cost': m.co,
        'cm_maintenance_cost': m.cm,
        'cy_cycle_cost': m.cy,
        'QMAX_max_payload': m.QMAX,
        'EMAX_max_soe': m.EMAX,
        'EMIN_min_soe': m.EMIN,
        'PMAX_max_ev_power': m.PMAX,
        'N_max_evs': m.N,
        'r_ev_consumption_rate': m.r,
        'v_speed': m.v,
        't_T_time_horizon': m.t_T,
        't_S_time_step': m.t_S,
        't_H_hours_per_time_step': m.t_H,
        'eff': m.eff or 1.,
        # SET
        'V01__num_nodes': len(m.V01_),
        'E_num_edges': len(m.E),
        'S__num_stations_extended': len(m.S_),
        'S_num_stations': len(m.S),
        'M_num_customers': len(m.M),
        'T_num_time': len(m.T),
        # RESULTS
        'total_distance': m.total_distance(m.instance)(),
        'fleet_size': m.C_fleet_capital_cost(m.instance)() / m.cc,
        'C_fleet_capital_cost': m.C_fleet_capital_cost(m.instance)(),
        'O_delivery_operating_cost': m.O_delivery_operating_cost(m.instance)() + m.O_maintenance_operating_cost(m.instance)(),
        'R_delivery_revenue': m.R_delivery_revenue(m.instance)(),
        'R_energy_arbitrage_revenue': m.R_energy_arbitrage_revenue(m.instance)(),
        'R_peak_shaving_revenue': m.R_peak_shaving_revenue(m.instance)(),
        'cycle_cost': m.cycle_cost(m.instance)()
    }
    # averaes over parameter sets
    # average decision variables per vehicle
    return stats
