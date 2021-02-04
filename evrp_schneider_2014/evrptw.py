from config.default_params import create_data_dictionary
from utils.utilities import import_instance
import logging
from pyomo.environ import *


class EVRPTW:
    """
    Implementation of Schneider, Stenger, and Goeke: The E-VRPTW and Recharging Stations, /n
    Transportation Science 48(4), pp. 500–520, ©2014 INFORMS /n
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
        self.m.Mq = Param(doc='Large value for big M payload constraints')
        self.m.Mt = Param(doc='Large value for big M service time constraints')
        self.m.Me = Param(doc='Large value for big M energy constraints')
        self.m.cW = Param(doc='Amortized capital cost for purchasing a vehicle')
        self.m.QMAX = Param(doc='Maximum payload limit for all vehicles')
        self.m.EMAX = Param(doc='Maximum EV battery SOE limit for all EVs')
        self.m.rE = Param(doc='Electric consumption per unit distance for EV')
        self.m.rC = Param(doc='Charging rate for EV')
        self.m.v = Param(doc='Average speed')

        # Defining sets
        self.m.V01_ = Set(dimen=1, doc='All nodes extended')
        self.m.start_node = Set(within=self.m.V01_, doc='Starting node')
        self.m.end_node = Set(within=self.m.V01_, doc='Ending node')
        self.m.V0_ = Set(initialize=self.m.V01_ - self.m.end_node, doc='All nodes extended except starting depot node')
        self.m.V1_ = Set(initialize=self.m.V01_ - self.m.start_node, doc='All nodes extended except ending depot node')
        self.m.V_ = Set(initialize=self.m.V01_ - self.m.start_node - self.m.end_node,
                        doc='All nodes extended except depot nodes')
        self.m.F_ = Set(dimen=1, doc='All charging station nodes')
        self.m.V = Set(dimen=1, doc='All customer nodes')
        self.m.V0 = Set(initialize=self.m.V0_ - self.m.F_, doc='All customer nodes including the starting node')
        self.m.F0_ = Set(initialize=self.m.V0_ - self.m.V, doc='All charging station nodes including the starting node')

        # Defining parameter sets
        self.m.d = Param(self.m.V01_, self.m.V01_, doc='Distance of edge (i;j) between nodes i;j (km)')
        self.m.q = Param(self.m.V01_, doc='Delivery demand at each customer')
        self.m.tS = Param(self.m.V01_, doc='Fixed service time for a vehicle at node i')
        self.m.tA = Param(self.m.V01_, doc='Time window start time at node i ')
        self.m.tB = Param(self.m.V01_, doc='Time window end time at node i ')

        logging.info('Defining variables')
        # Defining variables
        self.m.xgamma = Var(self.m.V01_, self.m.V01_, within=Boolean)  # Route decision of each edge for each EV
        self.m.xq = Var(self.m.V01_, within=NonNegativeReals)  # Payload of each vehicle before visiting each node
        self.m.xw = Var(self.m.V01_, within=NonNegativeReals)  # Arrival time for each vehicle at each node
        self.m.xa = Var(self.m.V01_, within=NonNegativeReals, initialize=self.m.EMAX)  # Energy of each EV arriving at each node

        # %% ROUTING CONSTRAINTS # TODO: consistency for ranged inequality expressions -
        #  WARNING:pyomo.core:DEPRECATED: Chained inequalities are deprecated.

        logging.info('Defining constraints')

        def constraint_visit_customers(m, i):
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) == 1

        # Create single visit constraint
        self.m.constraint_visit_customers = Constraint(self.m.V, rule=constraint_visit_customers)

        def constraint_visit_stations(m, i):
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= 1

        # Create single visit constraint
        self.m.constraint_visit_stations = Constraint(self.m.F_, rule=constraint_visit_stations)

        def constraint_single_route(m, j):
            route_in = sum(m.xgamma[i, j] for i in m.V0_ if i != j)
            route_out = sum(m.xgamma[j, i] for i in m.V1_ if i != j)

            return route_out - route_in == 0

        # Create single route constraint
        self.m.constraint_single_route = Constraint(self.m.V_, rule=constraint_single_route)

        # %% TIME CONSTRAINTS

        def constraint_time_customer(m, i, j):
            if i != j:
                # return m.xw[i] + (m.tS[i] + (m.d[i, j])) * m.xgamma[i, j] - m.Mt * (
                #         1 - m.xgamma[i, j]) <= m.xw[j]
                return inequality(None, m.xw[i] + (m.tS[i] + (m.d[i, j] / m.v)) * m.xgamma[i, j] - m.Mt * (
                        1 - m.xgamma[i, j]), m.xw[j])
            else:
                return Constraint.Skip

        self.m.constraint_time_customer = Constraint(self.m.V0, self.m.V1_, rule=constraint_time_customer)

        def constraint_time_station(m, i, j):
            if i != j:
                # return m.xw[i] + m.d[i, j] * m.xgamma[i, j] + m.rE * (m.EMAX - m.xa[j]) - (m.Mt + m.rE * m.EMAX) * (
                #         1 - m.xgamma[i, j]) <= m.xw[j]
                return inequality(None, m.xw[i] + (m.d[i, j] / m.v) * m.xgamma[i, j] + (m.rC * (m.EMAX - m.xa[i])) - (
                            m.Mt + (m.rC * m.EMAX)) * (
                                          1 - m.xgamma[i, j]), m.xw[j])
                # return inequality(None, m.xw[i] + (m.d[i, j] / m.v) * m.xgamma[i, j] + (m.rC * (m.QMAX - m.xa[i])) -
                #         m.Mt * (1 - m.xgamma[i, j]), m.xw[j])
            else:
                return Constraint.Skip

        self.m.constraint_time_station = Constraint(self.m.F_, self.m.V1_, rule=constraint_time_station)

        def constraint_node_time_window(m, i):
            # return m.tA[i] <= m.xw[i] <= m.tB[i]
            return inequality(m.tA[i], m.xw[i], m.tB[i])

        self.m.constraint_node_time_window = Constraint(self.m.V01_, rule=constraint_node_time_window)

        # %% PAYLOAD CONSTRAINTS

        def constraint_payload(m, i, j):
            if i != j:
                # return 0 <= m.xq[j] <= m.xq[i] - (m.q[i] * m.xgamma[i, j]) + m.Mq * (1 - m.xgamma[i, j]) # TODO: lower bound = 0 : ValueError: No value for uninitialized NumericValue object xq[C28]
                return inequality(None, m.xq[j], m.xq[i] - (m.q[i] * m.xgamma[i, j]) + m.Mq * (1 - m.xgamma[i, j]))
            else:
                return Constraint.Skip

        self.m.constraint_payload = Constraint(self.m.V0_, self.m.V1_, rule=constraint_payload)

        def constraint_payload_limit(m, i):
            # return 0 <= m.xq[i] <= m.QMAX
            return inequality(0, m.xq[i], m.QMAX)

        self.m.constraint_payload_limit = Constraint(self.m.start_node, rule=constraint_payload_limit)

        # %% ENERGY CONSTRAINTS

        def constraint_energy_customer(m, i, j):  # TODO: lower bound = 0
            if i != j:
                # return m.xa[j] <= m.xa[i] - (m.rE * m.d[i, j]) * m.xgamma[i, j] + m.Me * (1 - m.xgamma[i, j])
                return inequality(None, m.xa[j],
                                  m.xa[i] - m.rE * m.d[i, j] * m.xgamma[i, j] + m.Me * (1 - m.xgamma[i, j]))
            else:
                return Constraint.Skip

        self.m.constraint_energy_customer = Constraint(self.m.V, self.m.V1_, rule=constraint_energy_customer)

        def constraint_energy_station(m, i, j):  # TODO: lower bound = 0
            if i != j:
                return m.xa[j] <= m.EMAX - m.rE * m.d[i, j] * m.xgamma[i, j]
                # return inequality(None, m.xa[j], m.EMAX - (m.rE * m.d[i, j]) * m.xgamma[i, j])
            else:
                return Constraint.Skip

        self.m.constraint_energy_station = Constraint(self.m.F0_, self.m.V1_, rule=constraint_energy_station) #TODO: changed to F from F0


        # %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

        logging.info('Defining objective')

        def total_distance(m):
            """Total traveled distance"""
            return sum(sum(m.d[i, j] * m.xgamma[i, j] for i in m.V0_) for j in m.V1_)

        def fleet_cost(m):
            """Cost of total number vehicles"""
            return sum(sum(m.xgamma[i, j] * m.cW for i in self.m.start_node) for j in m.V0_)

        def obj_dist_fleet(m):
            """Objective: minimize the total traveled distance and the fleet size"""
            return total_distance(m) + fleet_cost(m)

        # Create objective function
        self.m.obj = Objective(rule=total_distance, sense=minimize)

    def solve(self, instance_filepath: str, duplicates: Boolean):
        # Specify solver
        opt = SolverFactory('gurobi')

        # Import data
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        logging.info('Importing VRPTW MILP instance: {}'.format(self.instance_name))

        self.data = import_instance(instance_filepath, duplicates)

        # Create parameters
        logging.info('Creating parameters')
        self.p = create_data_dictionary(self.data)

        # Create an instance of the AbstractModel passing in parameters
        logging.info('Creating instance')
        self.instance = self.m.create_instance(self.p)

        # Solver options
        # solv_options = {'MaxTime': 10e10}

        # Solve instance
        logging.info('Solving instance...')
        self.results = opt.solve(self.instance, tee=True) #, options=solv_options)
        logging.info('Done')
