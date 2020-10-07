from pyomo.environ import *
from evrp.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping, create_flat_distance_matrix
import pandas as pd
import numpy as np
import logging


class TSP:
    """ Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solution of Traveling Salesman Problem(TSP).\n
    Author: Rami Ariss
    """

    def __init__(self):
        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()
        logging.info('Building abstract model')
        self.build_model()

    def build_model(self):
        logging.info('Defining parameters and sets')

        # Defining node sets
        self.m.V01 = Set(dimen=1, doc='Graph nodes')
        self.m.start_node = Set(within=self.m.V01, doc='Starting node')
        self.m.end_node = Set(within=self.m.V01, doc='Ending node')
        self.m.V = Set(initialize=self.m.V01 - self.m.start_node - self.m.end_node, doc='Graph nodes')
        self.m.V0 = Set(initialize=self.m.V01 - self.m.end_node, doc='Graph nodes without terminal')
        self.m.V1 = Set(initialize=self.m.V01 - self.m.start_node, doc='Graph nodes without start')

        # Define edges set
        self.m.E = Set(initialize=self.m.V01 * self.m.V01, within=self.m.V01 * self.m.V01,
                       doc='Graph edges')

        # Defining fixed parameters
        self.m.d = Param(self.m.E, doc='Distance matrix')
        self.m.num_nodes = Param(doc='Number of nodes')

        logging.info('Defining variables')
        # Defining variables
        self.m.xgamma = Var(self.m.E, within=Boolean, doc='Route decision of each edge')

        # Dummy variable ui
        self.m.u = Var(self.m.V1, within=NonNegativeIntegers,
                       bounds=(0, self.m.num_nodes - 1), doc='Dummy variable for breaking subtours')

        logging.info('Defining constraints')
        # Defining routing constraints
        def constraint_in_arcs(m, j):
            return sum([m.xgamma[i, j] for i in m.V01 - j]) == 1
        self.m.constraint_in_arcs = Constraint(self.m.V01, rule=constraint_in_arcs)

        def constraint_out_arcs(m, i):
            return sum([m.xgamma[i, j] for j in m.V01 - i]) == 1
        self.m.constraint_out_arcs = Constraint(self.m.V01, rule=constraint_out_arcs)

        def constraint_single_subtour(m, i, j):
            if i != j:
                return m.u[i] - m.u[j] + m.xgamma[i, j] * m.num_nodes <= m.num_nodes - 1
            else:
                return Constraint.Skip

        self.m.constraint_single_subtour = Constraint(self.m.V1, self.m.V1,
                                                      rule=constraint_single_subtour)

        logging.info('Defining objective')
        def objective_distance(m):
            """Objective: Calculate total distance traveled"""
            return sum(m.xgamma[e] * m.d[e] for e in m.E)
        self.m.obj = Objective(rule=objective_distance, sense=minimize)

        logging.info('Done building model')

    def create_data_dictionary(self):
        self.p = {
            None: {
                'V01': {None: self.data['V'].index.values},
                'start_node': {None: [self.data['start_node']]},
                'end_node': {None: [self.data['end_node']]},
                'd': self.data['d'].stack().to_dict(),
                'num_nodes': {None: len(self.data['V'])}
            }
        }

    def import_instance(self, instance_filepath: str):
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        logging.info('Importing TSP MILP instance: {}'.format(self.instance_name))

        # Read in csv
        logging.info('Reading CSV')
        self.data = parse_csv_tables(instance_filepath)

        # Create graph by concatenating D, S, M nodes
        logging.info('Creating graph')
        node_types = [k for k in self.data.keys() if k in 'DSM']
        self.data['V'] = pd.concat(
            [self.data[k][['node_type', 'node_description', 'd_x', 'd_y']] for k in node_types])
        # self.data['num_nodes'] = len(self.data['V'])

        # Calculate distance matrix
        logging.info('Calculating distance matrix')
        self.data['d'] = calculate_distance_matrix(self.data['V'][['d_x', 'd_y']])

        # TODO: Bring upstream for user passthrough
        # Define start and end nodes
        self.data['start_node'] = 'd0'
        self.data['end_node'] = 'd-1'

        # Generate index mappings
        self.i2v, self.v2i = generate_index_mapping(self.data['V'].index)

        # Create data dictionary for concrete model instance
        self.create_data_dictionary()

    def solve(self, instance_filepath: str):
        # Specify solver
        opt = SolverFactory('gurobi', solver_io="python")

        # Solver options
        solver_options = {'threads': 4}  # 'keep': True, 'LocRes': True, 'results': True,

        # Import instance
        self.import_instance(instance_filepath)

        # Create an instance of the AbstractModel passing in parameters
        logging.info('Creating instance')
        self.instance = self.m.create_instance(self.p)

        # Solve instance
        logging.info('Solving instance...')
        self.results = opt.solve(self.instance, tee=True, options=solver_options)
        logging.info('Done')
