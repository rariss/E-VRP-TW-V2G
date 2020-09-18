from pyomo.environ import *
from evrp.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
import pandas as pd
import numpy as np
import logging


class MILP:
    """ Electric Vehicle Routing Problem with Time Windows, Vehicle-to-Grid, and Heterogeneous Fleet Design (E-VRP-TWV2GHFD).\n
    Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solutions.\n
    Author: Rami Ariss
    """

    def __init__(self, instance_filepath: str):
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        logging.info('Building MILP for instance: {}'.format(self.instance_name))

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
        self.build_model()

    def build_model(self):
        self.define_parameters_and_sets()
        self.define_constraints()


    def define_parameters_and_sets(self):
        # Defining sets
        self.m.V = Set(initialize=self.data['V'], doc='Graph nodes')
        self.m.E = Set(initialize=self.m.E * self.m.E, within=self.m.E * self.m.E, doc='Graph edges')

        # Defining fixed parameters
        self.m.t_T = Param(within=NonNegativeIntegers, doc='Optimization time horizon')
        self.m.t_S = Param(within=NonNegativeIntegers, doc='Time step size')
        # self.m.d =


    # def define_constraints(self):
    #     # Defining routing constraints

