import logging
import pandas as pd

from pyomo.environ import *  # Import last so no pyomo-provided math functions get overwritten

from evrptwv2g.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping

log = logging.getLogger('root')


class VRPTW:
    """ Vehicle Routing Problem with Time Windows and Homogeneous Fleet (VRPTW).\n
    Builds pyomo Mixed Integer Linear Program (MILP) model for exact-form solutions.\n
    Author: Rami Ariss
    """

    def __init__(self):
        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()
        log.info('Building abstract model')
        self.build_model()

    def build_model(self):
        log.info('Defining parameters and sets')

        # Defining fleet set
        self.m.NW = Param(doc='Maximum number of Vehicles in fleet')
        self.m.W = RangeSet(self.m.NW, doc='Set of vehicles')

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
        self.m.V01_type = Param(self.m.V01, within=Any, doc='Graph nodes type')  # TODO: Determine a better way to carry through node types
        self.m.d = Param(self.m.E, doc='Distance matrix')
        self.m.num_nodes = Param(doc='Number of nodes')
        self.m.tA = Param(self.m.V01, doc='Start time window')
        self.m.tB = Param(self.m.V01, doc='End time window')
        self.m.tS = Param(self.m.V01, doc='Fixed service time')
        self.m.tQ = Param(self.m.V01, doc='Delivery service time rate')
        self.m.t_T = Param(doc='Time horizon')
        self.m.t_S = Param(doc='Time step size')
        self.m.rW = Param(doc='Fuel consumption per unit distance')
        self.m.v = Param(doc='Average speed')
        self.m.cc = Param(doc='Amortized capital cost for purchasing a vehicle')
        self.m.c_F = Param(doc='Fuel cost')
        self.m.q = Param(self.m.V01, doc='Delivery demand of each node')
        self.m.QMAX = Param(doc='Maximum payload per vehicle')
        self.m.cq = Param(self.m.V01, doc='Delivery payment per unit at each node')

        log.info('Defining variables')
        # Defining variables
        self.m.xgamma = Var(self.m.W * self.m.E, initialize=0, within=Boolean, doc='Route decision of each edge for each vehicle')
        self.m.xw = Var(self.m.W * self.m.V01, initialize=0, within=NonNegativeReals, doc='Arrival time for each vehicle at each node')
        self.m.xq = Var(self.m.W * self.m.V01, initialize=self.m.QMAX, within=NonNegativeReals, doc='Payload of each vehicle arriving at each node')

        log.info('Defining constraints')
        # Defining routing constraints
        def constraint_visit_customers(m, i):
            return sum([m.xgamma[k, i, j] for k in m.W for j in m.V1 if j != i]) == 1
        self.m.constraint_visit_customers = Constraint(self.m.V, rule=constraint_visit_customers) # TODO: Changed V0 to V in constraints to try to get multiple assignments

        def constraint_in_and_out_arcs(m, k, i):
            return sum([m.xgamma[k, i, j] for j in m.V1 if j != i]) - sum([m.xgamma[k, j, i] for j in m.V0 if j != i]) == 0
        self.m.constraint_in_and_out_arcs = Constraint(self.m.W, self.m.V, rule=constraint_in_and_out_arcs)

        def constraint_vehicle_assignment(m, k):
            return sum([m.xgamma[k, s, j] for j in m.V for s in m.start_node]) <= 1
        self.m.constraint_vehicle_assignment = Constraint(self.m.W, rule=constraint_vehicle_assignment)

        # Defining time constraints
        def constraint_service_time(m, k, i, j):
            if i != j:
                # if m.V01_type[i] == 'd':
                #     return m.xw[k, i] + (m.tS[i] + m.d[i, j] / m.v + m.tQ[i] * (m.QMAX - m.q[i])) * m.xgamma[k, i, j] - m.t_T * (1 - m.xgamma[k, i, j]) <= m.xw[k, j]
                # else:
                return m.xw[k, i] + (m.tS[i] + m.d[i, j]/m.v + m.tQ[i] * m.q[i])*m.xgamma[k, i, j] - m.t_T*(1-m.xgamma[k, i, j]) <= m.xw[k, j]
            else:
                return Constraint.Skip
        self.m.constraint_service_time = Constraint(self.m.W, self.m.V0, self.m.V1, rule=constraint_service_time) # TODO: Changed V to V0 to capture start times

        def constraint_end_node_end_time(m, k):
            for i in m.end_node:
                return m.xw[k, i] + m.tS[i] + m.tQ[i] * (m.QMAX - m.q[i]) <= m.tB[i]
        self.m.constraint_end_node_end_time = Constraint(self.m.W, rule=constraint_end_node_end_time)

        def constraint_end_time(m, k, j):
            return m.xw[k, j] <= m.tB[j]
        self.m.constraint_end_time = Constraint(self.m.W, self.m.V1, rule=constraint_end_time)

        def constraint_start_time(m, k, i):
            return m.tA[i] <= m.xw[k, i]
        self.m.constraint_start_time = Constraint(self.m.W, self.m.V01, rule=constraint_start_time)

        # Defining payload constraints
        def constraint_payload(m, k, i, j):
            if i != j:
                # if m.V01_type[i] == 'd':
                #     return m.xq[k, i] + (m.QMAX - m.xq[k, i]) * m.xgamma[k, i, j] - m.QMAX * (1 - m.xgamma[k, i, j]) <= m.xq[k, j]
                # else:
                return m.xq[k, i] - m.q[i] * m.xgamma[k, i, j] + m.QMAX * (1 - m.xgamma[k, i, j]) >= m.xq[k, j]
            else:
                return Constraint.Skip
        self.m.constraint_payload = Constraint(self.m.W, self.m.V0, self.m.V1, rule=constraint_payload)

        def constraint_start_payload(m, k, i):
            return m.xq[k, i] == m.QMAX
        self.m.constraint_start_payload = Constraint(self.m.W, self.m.start_node, rule=constraint_start_payload)

        def constraint_max_payload(m, k, i):
            return m.xq[k, i] <= m.QMAX
        self.m.constraint_max_payload = Constraint(self.m.W, self.m.V01, rule=constraint_max_payload)


        log.info('Defining objective')
        def total_time_traveled(m, k):
            return sum(m.d[i, j] * m.xgamma[k, i, j] for i in m.V0 for j in m.V1 if j != i) / m.v

        def R_L(m):
            return sum(m.cq[i] * m.q[i] * m.xgamma[k, i, j] for k in m.W for i in m.V for j in m.V1 if j != i)

        def C(m):
            return m.cc * sum(m.xgamma[k, s, j] for k in m.W for j in m.V1 for s in m.start_node)

        def O_F(m):
            return m.c_F * m.rW * sum(total_time_traveled(m, k) for k in m.W) # TODO: keep getting NoneType error when multiplying * m.v

        def objective_net_amortized_profit(m):
            """Objective: Calculate net amortized profit across fleet"""
            return C(m) + O_F(m)
            # return R_L(m) - C(m) - O_F(m)
            # return sum(sum(sum(m.xgamma[k, i, j] * m.d[i, j] for k in m.W) for i in m.V0) for j in m.V1)
        self.m.obj = Objective(rule=objective_net_amortized_profit, sense=minimize)
        self.m.obj_C = Objective(rule=C, sense=minimize)
        self.m.obj_O_F = Objective(rule=O_F, sense=minimize)

        self.m.obj_C.deactivate()
        self.m.obj_O_F.deactivate()
        self.m.obj.activate()

        log.info('Done building model')

    def create_data_dictionary(self):
        self.p = {
            None: {
                'V01': {None: self.data['V'].index.values},
                'V01_type': self.data['V']['node_type'].to_dict(),
                'start_node': {None: [self.data['start_node']]},
                'end_node': {None: [self.data['end_node']]},
                'd': self.data['d'].stack().to_dict(),
                'num_nodes': {None: len(self.data['V'])},
                'q': self.data['V']['q'].to_dict(),
                'cq': self.data['V']['cq'].to_dict(),
                'tA': self.data['V']['tA'].to_dict(),
                'tB': self.data['V']['tB'].to_dict(),
                'tS': self.data['V']['tS'].to_dict(),
                'tQ': self.data['V']['tQ'].to_dict(),
                't_T': {None: self.data['Parameters'].loc['t_T', 'value']},
                't_S': {None: self.data['Parameters'].loc['t_S', 'value']},
                'NW': {None: self.data['W'].loc[:, 'NW'].sum()},
                'rW': {None: self.data['W'].loc[:, 'rW'].mean()},
                'v': {None: float(self.data['W'].loc[:, 'v'].mean())},
                'cc': {None: self.data['W'].loc[:, 'cc'].mean()},
                'c_F': {None: self.data['W'].loc[:, 'c_F'].mean()},
                'QMAX': {None: float(self.data['W'].loc[:, 'QMAX'].mean())}
            }
        }

    def import_instance(self, instance_filepath: str):
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        log.info('Importing VRPTW MILP instance: {}'.format(self.instance_name))

        # Read in csv
        log.info('Reading CSV')
        self.data = parse_csv_tables(instance_filepath)

        # Create graph by concatenating D, S, M nodes
        log.info('Creating graph')
        node_types = [k for k in self.data.keys() if k in 'DSM']
        self.data['V'] = pd.concat(
            [self.data[k] for k in node_types])
        # self.data['num_nodes'] = len(self.data['V'])

        # Calculate distance matrix
        log.info('Calculating distance matrix')
        self.data['d'] = calculate_distance_matrix(self.data['V'][['d_x', 'd_y']])

        # TODO: Bring upstream for user passthrough
        # Define start and end nodes
        self.data['start_node'] = self.data['D'].index[0]
        self.data['end_node'] = self.data['D'].index[1]

        # Generate index mappings
        self.i2v, self.v2i = generate_index_mapping(self.data['V'].index)

        # Create data dictionary for concrete model instance
        self.create_data_dictionary()

    def solve(self, instance_filepath: str):
        # Specify solver
        opt = SolverFactory('gurobi', solver_io="python")

        # Solver options
        solver_options = {'Threads': 8, 'MIPGap': 0.01, 'TimeLimit': 5*60}  # 'keep': True, 'LocRes': True, 'results': True,

        # Import instance
        self.import_instance(instance_filepath)

        # Create an instance of the AbstractModel passing in parameters
        log.info('Creating instance')
        self.instance = self.m.create_instance(self.p)

        # Solve instance
        log.info('Solving instance...')
        self.results = opt.solve(self.instance, tee=True, options=solver_options)
        log.info('Done')
