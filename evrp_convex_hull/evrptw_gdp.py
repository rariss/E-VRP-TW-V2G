from config.default_params import create_data_dictionary
from utils.utilities import import_instance
import logging
from pyomo.environ import *
from pyomo.gdp import *


class EVRPTW:

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
        self.m.V0_ = Set(initialize=self.m.V01_ - self.m.end_node, doc='All nodes extended except ending depot node')
        self.m.V1_ = Set(initialize=self.m.V01_ - self.m.start_node, doc='All nodes extended except starting depot node')
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
        self.m.xq = Var(self.m.V01_, within=NonNegativeReals, bounds=(0, self.m.QMAX))  # Payload of each vehicle before visiting each node
        self.m.xw = Var(self.m.V01_, within=NonNegativeReals, bounds=(0, self.m.Mt))  # Arrival time for each vehicle at each node
        self.m.xa = Var(self.m.V01_, within=NonNegativeReals, bounds=(0, self.m.EMAX))  # Energy of each EV arriving at each node

        # Create Boolean variables associated with the disjuncts.
        self.m.Y = BooleanVar(self.m.V01_, self.m.V01_)

        logging.info('Defining disjuncts and disjunction')

        # CONSTRAINTS FOR THE ARC_ON DISJUNT
        # ==================================================================================

        def ArcOn(d, i, j):

            if i != j:
                m = d.model()
                d.constraint_xgamma = Constraint(expr=m.xgamma[i,j] == 1)
                d.constraint_node_time_window = Constraint(expr=inequality(m.tA[i], m.xw[i], m.tB[i]))
                d.constraint_payload = Constraint(expr=inequality(0, m.xq[j], (m.xq[i] - m.q[i])))
                if i in m.V:
                    d.constraint_time_customer = Constraint(expr=inequality(0, m.xw[i] + (m.tS[i] + (m.d[i, j] / m.v)), m.xw[j]))
                    d.constraint_energy_customer = Constraint(expr=inequality(0, m.xa[j], m.xa[i] - (m.rE * m.d[i, j])))
                if i in m.F0_:
                    d.constraint_time_station = Constraint(expr=inequality(0, m.xw[i] + (m.d[i, j] / m.v) + (m.rC * (m.EMAX - m.xa[i])), m.xw[j]))
                    d.constraint_energy_station = Constraint(expr=inequality(0, m.xa[j], m.EMAX - (m.rE * m.d[i, j])))
                # if i in m.start_node:
                #     d.contraint_payload_start = Constraint(expr=inequality(0, m.xq[i], m.QMAX)) #TODO: redundant and breaks bigm xfrm

            else:
                return Disjunct.Skip

        self.m.ArcOn = Disjunct(self.m.V0_, self.m.V1_, rule=ArcOn)

        def ArcOff(d, i, j):

            if i != j:
                m = d.model()
                d.constraint_xgamma = Constraint(expr=m.xgamma[i,j] == 0)
                d.constraint_node_time_window = Constraint(expr=inequality(m.tA[i], m.xw[i], m.tB[i]))
                # d.constraint_payload = Constraint(expr=inequality(0, m.xq[i], m.QMAX)) #TODO: redundant and breaks bigm xfrm
                # d.constraint_energy = Constraint(expr=inequality(0, m.xa[i], m.EMAX)) #TODO: redundant and breaks bigm xfrm

            else:
                return Disjunct.Skip


        self.m.ArcOff = Disjunct(self.m.V0_, self.m.V1_, rule=ArcOff)

        def ArcState(m, i, j):
            if i == j:
                return Disjunction.Skip
            else:
                return [m.ArcOn[i,j], m.ArcOff[i,j]]
        self.m.ArcState = Disjunction(self.m.V0_, self.m.V1_, rule=ArcState)

        def constraint_route_customer(m, i):
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) == 1
        self.m.constraint_route_customer = Constraint(self.m.V, rule=constraint_route_customer)

        def constraint_visit_stations(m, i):
            return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= 1
        self.m.constraint_visit_stations = Constraint(self.m.F_, rule=constraint_visit_stations)

        def constraint_single_route(m, j):
            route_in = sum(m.xgamma[i, j] for i in m.V0_ if i != j)
            route_out = sum(m.xgamma[j, i] for i in m.V1_ if i != j)

            return route_out - route_in == 0
        self.m.constraint_single_route = Constraint(self.m.V_, rule=constraint_single_route)

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

    def solve(self, instance_filepath: str, duplicates: Boolean, xfrm_key):

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

        # Apply convex hull or big-M transform
        xfrm = TransformationFactory('gdp.{}'.format(xfrm_key))
        xfrm.apply_to(self.instance)

        # Solver options
        # solv_options = {'MaxTime': 10e10}

        # Solve instance
        logging.info('Solving instance...')
        self.results = opt.solve(self.instance, tee=True)

        logging.info('Done')

        # Attempt to explicit disjuncts method from the documentation
# https://pyomo.readthedocs.io/en/latest/modeling_extensions/gdp/modeling.html#additional-examples
#         # CONSTRAINTS FOR THE ARC_ON DISJUNT
#         # ==================================================================================
#
#         self.m.arc_on = Disjunct(self.m.V0_, self.m.V1_)
#
#         def constraint_xgamma_on(m, i, j):
#             if i != j:
#                 return m.xgamma[i, j] == 1
#             else:
#                 return Constraint.Skip
#         self.m.arc_on.constraint_xgamma_on = Constraint(self.m.V0_, self.m.V1_, rule=constraint_xgamma_on)
#
#         def constraint_payload_on(m, i, j):
#             if i != j:
#                 # return inequality(None, m.xq1[j], (m.xq1[i] - m.q[i] * m.xgamma[i,j]) * m.y1)
#                 return inequality(None, m.xq[j], (m.xq[i] - m.q[i]))
#             else:
#                 return Constraint.Skip
#         self.m.arc_on.constraint_payload_on = Constraint(self.m.V0_, self.m.V1_, rule=constraint_payload_on)
#
#         def constraint_payload_limit_on(m, i):
#             # return 0 <= m.xq[i] <= m.QMAX
#             return inequality(0, m.xq[i], m.QMAX)
#         self.m.arc_on.constraint_payload_limit_on = Constraint(self.m.start_node, rule=constraint_payload_limit_on)
#
#         # CONSTRAINTS FOR THE ARC_OFF DISJUNT
#         # ==================================================================================
#
#         self.m.arc_off = Disjunct(self.m.V0_, self.m.V1_)
#
#         def constraint_xgamma_off(m, i, j):
#             if i != j:
#                 return m.xgamma[i, j] == 0
#             else:
#                 return Constraint.Skip
#         self.m.arc_off.constraint_xgamma_off = Constraint(self.m.V0_, self.m.V1_, rule=constraint_xgamma_off)
#
#         def constraint_payload_off(m, i):
#             return inequality(0, m.xq[i], None)
#         self.m.arc_off.constraint_payload_off = Constraint(self.m.V01_, rule=constraint_payload_off)
#
#         # Defining the disjunction
#         # ==================================================================================
#
#         self.m.disj = Disjunction(self.m.V01_, self.m.V01_, expr=[self.m.arc_on, self.m.arc_off])
#
#         for idx in self.m.Y:
# ...         self.m.Y[idx].associate_binary_var(self.m.disj.indicator_var)
