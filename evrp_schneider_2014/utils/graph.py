from config.GLOBAL_CONFIG import node_colors_hex
from utils.utilities import create_flat_distance_matrix, create_coordinate_dict
import pandas as pd
import networkx as nx
import logging


def draw_plain_graph(v: 'pd.DataFrame', d: 'pd.DataFrame') -> 'fig':
    """
    Plots a plain, unlabeled undirected graph network given a df of nodes and coordinates.

    :param v: nodes and coordinates
    :type v: pd.DataFrame
    :param d: from and to node with edge distances
    :type d: pd.DataFrame
    :return: G
    :rtype: graph mode object
    """
    d_flat = create_flat_distance_matrix(d)
    G = nx.from_pandas_edgelist(d_flat, 'from', 'to', ['d'])
    coordinates = create_coordinate_dict(v)

    # Draw network graph G generated using pandas edge list with node positions corresponding to
    # coordinates
    nx.draw(G, pos=coordinates)

    logging.info('\n{}'.format(nx.info(G)))
    return G


def draw_graph(v: 'pd.DataFrame', d: 'pd.DataFrame') -> 'fig':
    """
    Plots a undirected graph network given a df of nodes and coordinates.

    :param v: nodes and coordinates
    :type v: pd.DataFrame
    :param d: from and to node with edge distances
    :return: G
    :rtype: graph mode object
    """
    d_flat = create_flat_distance_matrix(d)
    G = nx.from_pandas_edgelist(d_flat, 'from', 'to', ['d'])
    coordinates = create_coordinate_dict(v)

    # Map node type to node colors
    node_color = [node_colors_hex[n] for n in v['node_type']]

    # Draw network graph G generated using pandas edge list with node positions corresponding to
    # coordinates
    nx.draw(G, pos=coordinates, with_labels=True, node_color=node_color, node_size=1000, alpha=0.8, width=1, edge_color='grey')

    logging.info('\n{}'.format(nx.info(G)))
    return G
