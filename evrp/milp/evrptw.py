from evrp.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
import pandas as pd
import numpy as np
import logging
from pyomo.environ import *

class EVRPTW:
    """
    Implementation of E-VRP-TW-V2G
    Authors: Leandre Berwa and Rami Ariss
    """

    def __init__(self):
        # Instantiate pyomo Abstract Model

        self.m = AbstractModel()
        logging.info('Building abstract model')
        self.build_model()

    def build_model(self):
        logging.info('Defining parameters and sets')

        # Defining fixed parameters
        self.m.MQ = Param(doc='Large value for big M payload constraints')
        self.m.MT = Param(doc='Large value for big M service time constraints')
        self.m.ME = Param(doc='Large value for big M energy constraints')
        self.m.cc = Param(doc='Amortized capital cost for purchasing a vehicle')
        self.m.co = Param(doc='Amortized operating cost for delivery (wages)')
        self.m.cm = Param(doc='Amortized operating cost for maintenance')
        self.m.QMAX = Param(doc='Maximum payload limit for all vehicles')
        self.m.EMAX = Param(doc='Maximum EV battery SOE limit for all EVs')
        self.m.EMIN = Param(doc='Minimum EV battery SOE limit for all EVs')
        self.m.PMAX = Param(doc='Maximum EV inverter limit for all EVs')
        self.m.r = Param(doc='Electric consumption per unit distance for EV')
        self.m.v = Param(doc='Average speed')
        self.m.t_T = Param(doc='Time horizon')
        self.m.t_S = Param(doc='Time step size')

        # Defining sets
        self.m.V01_ = Set(dimen=1, doc='All nodes extended')
        self.m.start_node = Set(within=self.m.V01_, doc='Starting node')
        self.m.end_node = Set(within=self.m.V01_, doc='Ending node')
        self.m.V0_ = Set(initialize=self.m.V01_ - self.m.end_node, doc='All nodes extended except ending depot node')
        self.m.V1_ = Set(initialize=self.m.V01_ - self.m.start_node, doc='All nodes extended except starting depot node')
        self.m.V_ = Set(initialize=self.m.V01_ - self.m.start_node - self.m.end_node,
                        doc='All nodes extended except start and end depot nodes')
        self.m.S_ = Set(dimen=1, doc='All charging station nodes extended')
        self.m.D_ = Set(dimen=1, doc='All depot nodes extended')
        self.m.S = Set(dimen=1, doc='All charging station nodes')
        self.m.M = Set(dimen=1, doc='All customer nodes')
        self.m.D = Set(dimen=1, doc='All depot nodes')
        # self.m.V0 = Set(initialize=self.m.V0_ - self.m.F, doc='All customer nodes including the starting node')
        # self.m.F0 = Set(initialize=self.m.V0_ - self.m.V, doc='All charging station nodes including the starting node')
        self.m.E = Set(initialize=[(i, j) for i in self.m.V0_ for j in self.m.V1_ if i !=j],
                       within=self.m.V01_ * self.m.V01_, doc='Graph edges')
        self.m.T = Set(initialize=list(range(0, self.m.t_T, self.m.t_S)), doc='Time')

        # Defining parameter sets
        self.m.d = Param(self.m.E, doc='Distance of edge (i;j) between nodes i;j (km)')
        self.m.q = Param(self.m.V01_, doc='Delivery demand at each customer')
        self.m.tS = Param(self.m.V01_, doc='Fixed service time for a vehicle at node i')
        self.m.tQ = Param(self.m.V01_, doc='Payload service time for a vehicle at node i')
        self.m.tA = Param(self.m.V01_, doc='Time window start time at node i')
        self.m.tB = Param(self.m.V01_, doc='Time window end time at node i')
        self.m.SMAX = Param(self.m.S_, doc='Maximum station inverter limit')
        self.m.SMIN = Param(self.m.S_, doc='Minimum station inverter limit')
        # self.m.G = Param(self.m.D, self.m.T, doc='Depot electric demand profile')
        # self.m.Dmap = Param(self.m.D, doc='Mapping from original depot to all its duplicates')
        # self.m.cg = Param(self.D, doc='Amortized operating cost for demand charges')
        self.m.ce = Param(self.D & self.m.S, self.T, doc='Amortized operating cost for energy charges')
        self.m.cq = Param(self.D & self.m.S, self.T, doc='Amortized revenue for deliveries')
        self.m.Smap = Param(self.m.S, doc='Mapping from original charging station to all its duplicates')

        logging.info('Defining variables')
        # Defining variables
        self.m.xgamma = Var(self.m.E, within=Boolean)  # Route decision of each edge for each EV
        self.m.xkappa = Var(self.m.D_ & self.m.S_, self.m.T, within=Boolean)
        self.m.xw = Var(self.m.V01_, within=NonNegativeReals)  # Arrival time for each vehicle at each node
        self.m.xq = Var(self.m.V01_, within=NonNegativeReals)  # Payload of each vehicle before visiting each node
        self.m.xa = Var(self.m.V01_, within=NonNegativeReals)  # Energy of each EV arriving at each node
        # self.m.xd = Var(self.m.D, within=NonNegativeReals)  # Each depot’s net peak electric demand
        self.m.xp = Var(self.m.D_ & self.S_, self.m.T, within=Reals)  # Charge or discharge rate of each EV

        # %% ROUTING CONSTRAINTS # TODO: consistency for ranged inequality expressions -
        #  WARNING:pyomo.core:DEPRECATED: Chained inequalities are deprecated.

        logging.info('Defining constraints')

        def constraint_visit_stations(m, i):
            """Depot and charging station nodes can be visited at most once in extended graph"""
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= 1
        self.m.constraint_visit_stations = Constraint(self.m.D_ & self.m.S_, rule=constraint_visit_stations)

        def constraint_visit_customers(m, i):
            """EVs must visit customer nodes once"""
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) == 1
        self.m.constraint_visit_customers = Constraint(self.m.M, rule=constraint_visit_customers)

        def constraint_single_route(m, i):
            """Vehicle incoming arcs equal outcoming arcs for any intermediate nodes"""
            route_out = sum(m.xgamma[i, j] for j in m.V1_ if i != j)
            route_in = sum(m.xgamma[j, i] for j in m.V0_ if i != j)
            return route_out - route_in == 0
        self.m.constraint_single_route = Constraint(self.m.V_, rule=constraint_single_route)

        # %% TIME CONSTRAINTS

        def constraint_time_depot(m, i, j):
            """Service time for each vehicle filling payload or doing V2G/G2V at each intermediate depot"""
            if i != j:
                return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v + m.tQ * (m.QMAX - m.xq[i]) +
                                  m.t_S * sum(m.xkappa[i, t] for t in m.T)) * m.xgamma[i, j] \
                       - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
            else:
                return Constraint.Skip
        self.m.constraint_time_depot = Constraint(self.m.D_, self.m.V1_, rule=constraint_time_depot)

        def constraint_time_start_depot(m, j):
            """Service time for each vehicle at start depot"""
            i = m.start_node
            if i != j:
                return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v) * m.xgamma[i, j] \
                       - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
            else:
                return Constraint.Skip
        self.m.constraint_time_start_depot = Constraint(self.m.V1_, rule=constraint_time_start_depot)

        def constraint_time_station(m, i, j):
            """Service time for each EV doing V2G/G2V at each charging station"""
            if i != j:
                return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v + m.t_S * sum(m.xkappa[i, t] for t in m.T)) \
                       * m.xgamma[i, j] - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
            else:
                return Constraint.Skip
        self.m.constraint_time_station = Constraint(self.m.S_, self.m.V1_, rule=constraint_time_station)

        def constraint_time_customer(m, i, j):
            """Service time for each vehicle doing delivery at each customer node"""
            if i != j:
                return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v + m.tQ * m.q[i]) \
                       * m.xgamma[i, j] - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
            else:
                return Constraint.Skip
        self.m.constraint_time_customer = Constraint(self.m.M, self.m.V1_, rule=constraint_time_customer)

        def constraint_node_time_window(m, i):
            """Arrival time must be after time window starts for each vehicle"""
            return m.tA[i] <= m.xw[i] <= m.tB[i]
        self.m.constraint_node_time_window = Constraint(self.m.V01_, rule=constraint_node_time_window)

        # %% ENERGY CONSTRAINTS

        def constraint_energy_station(m, i, j):
            """Energy transition for each EV while at an intermediate depot or charging station node i and traveling across edge (i, j)"""
            if i != j:
                return m.xa[j] <= m.xa[i] + m.t_S * sum(m.xkappa[i, t] * m.xp[i, t] for t in m.T) - \
                       (m.r * m.d[i, j]) * m.xgamma[i, j] + m.ME * (1 - m.xgamma[i, j])
            else:
                Constraint.Skip
        self.m.constraint_energy_station = Constraint(self.m.D_ & self.m.S_, self.m.V1_, rule=constraint_energy_station)

        def constraint_energy_customer(m, i, j):
            """Energy transition for each EV while at customer node i and traveling across edge (i, j)"""
            if i != j:
                return m.xa[j] <= m.xa[i] - (m.r * m.d[i, j]) * m.xgamma[i, j] + m.ME * (1 - m.xgamma[i, j])
            else:
                Constraint.Skip
        self.m.constraint_energy_customer = Constraint(self.m.M & self.m.start_node, self.m.V1_, rule=constraint_energy_customer)

        def constraint_energy_ev_limit(m, i, t):
            """Maximum charge and discharge limit for each EV at depot and charging stations"""
            return -m.PMAX * m.xkappa[i, t] <= m.xp[i, t] <= m.PMAX * m.xkappa[i, t]
        self.m.constraint_energy_ev_limit = Constraint(self.m.D_ & self.m.S_, self.m.T, rule=constraint_energy_ev_limit)

        def constraint_energy_station_limit(m, i, t):
            """Maximum charge and discharge limit for an EV at depot or charging station i ∈ D0 0,−1 ∪ S0"""
            return m.SMIN[i] * m.xkappa[i, t] <= m.xp[i, t] <= m.SMAX[i] * m.xkappa[i, t]
        self.m.constraint_energy_station_limit = Constraint(self.m.D_ & self.m.S_, self.m.T, rule=constraint_energy_station_limit)

        def constraint_energy_start_end_soe(m, i, j):
            """Start and end energy state must be equal for each EV"""
            return m.xa[i] == m.xa[j]
        self.m.constraint_energy_start_end_soe = Constraint(self.m.start_node, self.m.end_node, rule=constraint_energy_start_end_soe)

        def constraint_energy_soe(m, i, t):
            """Minimum and Maximum SOE limit for each EV"""
            return m.EMIN <= m.xa[i] <= m.EMAX
        self.m.constraint_energy_soe = Constraint(self.m.V01_, rule=constraint_energy_soe)

        # TODO: Need a depot D -> all duplicate depots i mapping
        # See this implementation example: https://stackoverflow.com/questions/53966482/how-to-map-different-indices-in-pyomo
        # def constraint_energy_peak(m, d, t):
        #     """Peak electric demand for each physical depot d(i) ∈ D"""
        #     return m.G[d, t] + sum(m.xkappa[i, t] * m.xp[i, t] for i in m.Dmap[d]) <= m.xd[d]
        # self.m.constraint_energy_peak = Constraint(self.m.D, self.m.T, rule=constraint_energy_peak)

        # %% PAYLOAD CONSTRAINTS

        # TODO: Likely don't need this constraint since there are no strict bounds on depot payload
        # def constraint_payload_depot(m, i, j):
        #     """Vehicles must fully fill payload when visiting a depot (constraint likely not needed)"""
        #     if i != j:
        #         return m.xq[i] + (m.QMAX - m.xq[i]) * m.xgamma[i, j] - m.MQ * (1 - m.xgamma[i, j]) <= m.xq[j]
        #     else:
        #         Constraint.Skip
        #     return
        # self.m.constraint_payload_depot = Constraint(self.m.D_ & self.m.start_node, self.m.V1_, rule=constraint_payload_depot)

        def constraint_payload(m, i, j):
            """Vehicles must unload payload for full customer demand when visiting a customer"""
            if i != j:
                return m.xq[j] <= m.xq[i] - (m.q[i] * m.xgamma[i, j]) + m.MQ * (1 - m.xgamma[i, j])
            else:
                Constraint.Skip
        self.m.constraint_payload = Constraint(self.m.M, self.m.V1_, rule=constraint_payload)

        def constraint_payload_limit(m, i):
            """Payload limits for each vehicle"""
            if i == m.start_node:  # Each vehicle must start (and end) with a full payload
                return m.xq[i] == m.QMAX
            else:
                return m.xq[i] <= m.QMAX  # xq is NonNegative
        self.m.constraint_payload_limit = Constraint(self.m.V01_ - self.m.S_, rule=constraint_payload_limit)

        # %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

        logging.info('Defining objective')

        def C_fleet_capital_cost(m):
            """Cost of total number vehicles"""
            return m.cW * sum(m.xgamma[m.start_node, j] for j in m.V_ if j != m.start_node)

        def O_delivery_operating_cost(m):
            """Amortized delivery operating cost for utilized vehicles (wages)"""
            return m.co * total_time(m)

        def O_maintenance_operating_cost(m):
            """Amortized delivery maintenance cost for utilized vehicles"""
            return m.cm * total_distance(m)

        # TODO: Implement GMAX properly by doing maximization in data preparation and passing through as parameters
        def R_peak_shaving_revenue(m):
            """Amortized V2G peak shaving demand charge savings (or net demand charge cost) over all depots"""
            return sum(sum(m.cg[d] * (max(m.G[d, :]) - m.xd[i]) for i in m.Dmap[d]) for d in m.D)

        def R_energy_arbitrage_revenue(m):
            """Amortized G2V/V2G energy arbitrage (or net cost of charging) over all depots and charging stations"""
            return sum(sum(sum(m.ce[d, t] * m.xp[i, t] * m.xkappa[i, t] for t in m.T) for i in m.Dmap[d]) for d in m.D) + \
                    sum(sum(sum(m.ce[s, t] * m.xp[i, t] * m.xkappa[i, t] for t in m.T) for i in m.Smap[s]) for s in m.S)

        def R_delivery_revenue(m):
            """Amortized delivery revenue for goods delivered to customer nodes by entire fleet"""
            return sum(m.cq[i] * m.q[i] for i in m.M)

        def total_distance(m):
            """Total traveled distance"""
            return sum(sum(m.d[i, j] * m.xgamma[i, j] for i in m.V0_) for j in m.V1_)

        def total_time(m):
            """Total time traveled by all EVs"""
            return total_distance(m) / m.v

        def obj_dist_fleet(m):
            """Objective: minimize the total traveled distance and the fleet size"""
            return total_distance(m)

        # Create objective function
        self.m.obj = Objective(rule=total_distance, sense=minimize)

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
        logging.info('Importing VRPTW MILP instance: {}'.format(self.instance_name))

        # Read in csv
        logging.info('Reading CSV')
        self.data = parse_csv_tables(instance_filepath)

        # Create graph by concatenating D, S, M nodes
        logging.info('Creating graph')
        node_types = [k for k in self.data.keys() if k in 'DSM']
        self.data['V'] = pd.concat([self.data[k] for k in node_types])
        # self.data['num_nodes'] = len(self.data['V'])

        # Calculate distance matrix
        logging.info('Calculating distance matrix')
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
        opt = SolverFactory('gurobi')

        # Import data
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        logging.info('Importing EVRPTW MILP instance: {}'.format(self.instance_name))

        self.data = import_instance(instance_filepath)

        # Create parameters
        logging.info('Creating parameters')
        self.p = create_data_dictionary(self.data)

        # Create an instance of the AbstractModel passing in parameters
        logging.info('Creating instance')
        self.instance = self.m.create_instance(self.p)

        # Solver options
        solv_options = {'MaxTime': 10e10}

        # Solve instance
        logging.info('Solving instance...')
        self.results = opt.solve(self.instance, tee=True, options=solv_options)
        logging.info('Done')
