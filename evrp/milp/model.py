from pyomo.environ import *
from evrp.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
import pandas as pd
import numpy as np


class MILP:
    """ Electric Vehicle Routing Problem with Time Windows, Vehicle-to-Grid, and Heterogeneous Fleet Design (E-VRP-TWV2GHFD).\n
    Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solutions.\n
    Author: Rami Ariss
    """

    def __init__(self, instance_filepath: str):

        # Read in csv
        self.data = parse_csv_tables(instance_filepath)

        # Create graph by concatenating D, S, M nodes
        self.data['V'] = pd.concat([self.data[k][['node_type', 'node_description', 'd_x', 'd_y']] for k in 'DSM'])

        # Calculate distance matrix
        self.data['d'] = calculate_distance_matrix(self.data['V'][['d_x','d_y']])

        # Generate index mappings
        self.i2v, self.v2i = generate_index_mapping(self.data['V'].index)

        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()
