from evrp.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
import pandas as pd
import logging


def read_data(instance_filepath: str) -> dict:
    # TODO: probably move this to utilities

    # Read in csv
    logging.info('Reading CSV')
    data = parse_csv_tables(instance_filepath)

    # Create graph by concatenating D, S, M nodes
    logging.info('Creating graph')
    data['V'] = pd.concat(
        [data[k][['node_type', 'node_description', 'd_x', 'd_y']] for k in 'DSM'])
    data['num_nodes'] = len(data['V'])

    # Calculate distance matrix
    logging.info('Calculating distance matrix')
    data['d'] = calculate_distance_matrix(data['V'][['d_x', 'd_y']])

    # Generate index mappings
    i2v, v2i = generate_index_mapping(data['V'].index)

    return data


def create_params(instance_filepath: str) -> dict:
    data = read_data(instance_filepath)
    nodes = set(data['V'].index.values)
    edges_dist = data['d'].stack()
    edges = list(edges_dist.index)

    params = {None: {
        'V01': {None: nodes},  # Graph nodes
        'V': {None: nodes - {'d0', 'd-1'}},  # Graph nodes
        'V0': {None: nodes - {'d-1'}},  # Graph nodes without terminal
        'V1': {None: nodes - {'d0'}},  # Graph nodes without start
        'E': {None: edges},  # Graph edges
        'd': dict(zip(edges, edges_dist))  # Distances of edges (i;j) between nodes i;j
    }
    }

    return params
