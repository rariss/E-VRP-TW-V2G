import logging
import datetime
import pandas as pd

from pyomo.environ import *

from evrptwv2g.utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
from evrptwv2g.config.GLOBAL_CONFIG import SOLVER_TYPE

log = logging.getLogger('root')


# Define functions
# Parameters and Sets
def E_init(m):  # Excludes edges for the same node (i.e. diagonal)
    return [(i, j) for i in m.V0_ for j in m.V1_ if i != j]


# Constraints
# Routing Constraints
def constraint_visit_stations(m, i):
    """Charging station nodes can be visited at most once in extended graph"""
    return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= 1


def constraint_visit_customers(m, i):
    """EVs must visit customer nodes once"""
    return sum(m.xgamma[i, j] for j in m.V1_ if i != j) == 1


def constraint_single_route(m, i):
    """Vehicle incoming arcs equal outcoming arcs for any intermediate nodes"""
    route_out = sum(m.xgamma[i, j] for j in m.V1_ if i != j)
    route_in = sum(m.xgamma[j, i] for j in m.V0_ if i != j)
    return route_out - route_in == 0


def constraint_min_vehicles(m, i):
    """Requires at least one vehicle assignment"""
    return sum(m.xgamma[i, j] for j in m.V1_ if i != j) >= 1


def constraint_max_vehicles(m, i):
    """Requires at most N vehicle assignments"""
    return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= m.N


def constraint_fixed_vehicles(m, i):
    """Requires exactly N vehicle assignments"""
    return sum(m.xgamma[i, j] for j in m.V1_ if i != j) == m.N


def constraint_station_symmetry(m, i):
    """Orders the stations with more than one instance"""
    duplicate_number = eval(i.split('_')[-1])
    if duplicate_number >= 1:  # check if it is a duplicate node
        previous_duplicate = i.split('_')[0] + '_' + str(duplicate_number - 1)
        return sum(m.xgamma[i, j] for j in m.V1_ if i != j) <= \
               sum(m.xgamma[previous_duplicate, j] for j in m.V1_ if previous_duplicate != j)
    else:
        return Constraint.Skip


def constraint_no_stationary_evs(m, i, j, s_):
    """Ensures no stationary vehicles staying at depot."""
    # if len(m.Smap[s]) > 1:
    #     return sum(m.xgamma[i, s_] + m.xgamma[s_, s__] + m.xgamma[s_, j] for s_ in m.Smap[s] for s__ in m.Smap[s] if s_ != s__) <= 0
    # else:
    #     return sum(m.xgamma[i, s_] + m.xgamma[s_, j] for s_ in m.Smap[s]) <= 0
    return m.xgamma[i, s_] + m.xgamma[s_, j] <= 1


def constraint_no_visit_duplicate_station(m, s):
    """Ensures no visit between duplicate nodes of the same charging station."""
    if len(m.Smap[s]) > 1:
        return sum(m.xgamma[s_, s__] for s_ in m.Smap[s] for s__ in m.Smap[s] if s_ != s__) <= 0
    else:
        return Constraint.Skip


# Time Constraints
def constraint_xgamma_xkappa(m, i, t):
    """Ensures charging can only happen when a vehicle is physically present at a node.
    IF $\sum_j{x_{ij}^\gamma}=0$, THEN $x^\kappa_{it}=0\ \forall t$, ELSE $x^\kappa_{it}=0\ \forall t<x^\omega_i$."""
    return m.xkappa[i, t] <= sum(m.xgamma[i, j] for j in m.V1_ if i != j)


def constraint_time_start_depot(m, i, j):  # TODO: Can we drop the constraint for the start depot if we drop fixed service time?
    """Service time for each vehicle at start depot"""
    if i != j:
        return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v) * m.xgamma[i, j] / m.t_H \
               - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
    else:
        return Constraint.Skip


def constraint_time_station(m, i, j):
    """Service time for each EV doing V2G/G2V at each charging station"""
    if i != j:
        return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v) * m.xgamma[i, j] / m.t_H + m.t_S * sum(m.xkappa[i, t] for t in m.T) \
               - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
    else:
        return Constraint.Skip


def constraint_time_customer(m, i, j):  # TODO: Should we get rid of "unload time"?
    """Service time for each vehicle doing delivery at each customer node"""
    if i != j:
        return m.xw[i] + (m.tS[i] + m.d[i, j] / m.v + m.tQ[i] * m.q[i]) \
               * m.xgamma[i, j] / m.t_H - m.MT * (1 - m.xgamma[i, j]) <= m.xw[j]
    else:
        return Constraint.Skip


def constraint_node_time_window(m, i):
    """Arrival time must be within time window for each node"""
    return inequality(m.tA[i], m.xw[i], m.tB[i])


def constraint_terminal_node_time(m, i):  # TODO: Could remove if m.tB[i] = m.t_T - m.tS[i]
    """Arrival time must be within time window for each node"""
    return m.xw[i] + m.tS[i] / m.t_H <= m.t_T


def constraint_time_xkappa_lb(m, i, t):
    """V2G decisions must be made after arrival at the node"""
    return (m.t_T - t) * m.xkappa[i, t] - m.MT * (1 - m.xkappa[i, t]) <= m.t_T - m.xw[i]


def constraint_time_xkappa_ub(m, i, j, t):
    """V2G decisions must be made before departure from the node"""
    if i != j:
        return (m.tS[i] + m.d[i, j] / m.v) * m.xgamma[i, j] / m.t_H + (t + m.t_S) * m.xkappa[i, t] - m.MT * (
                    1 - m.xgamma[i, j]) <= m.xw[j]
    else:
        return Constraint.Skip

# Energy Constraints
# def constraint_xp(m, i, t):
#     """Constructs the power variable from split charge and discharge variables."""
#     return m.xp[i, t] == m.xc[i, t] - m.xg[i, t]


def constraint_energy_zero_xg_splitxp(m, i, t):
    """Ensures no discharging."""
    return m.xg[i, t] == 0


def constraint_energy_station_splitxp(m, i, j):  # TODO: Check if number of xkappa variables can be reduced for same energy / depot periods
    """Energy transition for each EV while at an intermediate charging station node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] <= m.xa[i] + m.t_H * m.t_S * sum(m.eff * m.xc[i, t] - m.xg[i, t] / m.eff for t in m.T) - \
               (m.r * m.d[i, j]) * m.xgamma[i, j] + m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_station_exact_splitxp(m, i, j):  # TODO: Check if number of xkappa variables can be reduced for same energy / depot periods
    """Cap for exact energy transition for each EV while at an intermediate charging station node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] >= m.xa[i] + m.t_H * m.t_S * sum(m.eff * m.xc[i, t] - m.xg[i, t] / m.eff for t in m.T) - \
               (m.r * m.d[i, j]) * m.xgamma[i, j] - m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_station_exact(m, i, j):  # TODO: Check if number of xkappa variables can be reduced for same energy / depot periods
    """Cap for exact energy transition for each EV while at an intermediate charging station node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] >= m.xa[i] + m.t_H * m.t_S * sum(m.xp[i, t] for t in m.T) - \
               (m.r * m.d[i, j]) * m.xgamma[i, j] - m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_station(m, i, j):  # TODO: Check if number of xkappa variables can be reduced for same energy / depot periods
    """Energy transition for each EV while at an intermediate charging station node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] <= m.xa[i] + m.t_H * m.t_S * sum(m.xp[i, t] for t in m.T) - \
               (m.r * m.d[i, j]) * m.xgamma[i, j] + m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_customer(m, i, j):  # TODO: Can we drop this constraint for the start_node?
    """Energy transition for each EV while at customer node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] <= m.xa[i] - (m.r * m.d[i, j]) * m.xgamma[i, j] + m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_customer_exact(m, i, j):  # TODO: Can we drop this constraint for the start_node?
    """Energy transition for each EV while at customer node i and traveling across edge (i, j)"""
    if i != j:
        return m.xa[j] >= m.xa[i] - (m.r * m.d[i, j]) * m.xgamma[i, j] - m.ME * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_energy_ev_limit_lb_splitxp(m, i, t):
    """Charge limits for each EV at charging stations"""
    return -m.PMAX * m.xkappa[i, t] <= m.xc[i, t] - m.xg[i, t]


def constraint_energy_ev_limit_lb(m, i, t):
    """Charge limits for each EV at charging stations"""
    return -m.PMAX * m.xkappa[i, t] <= m.xp[i, t]


def constraint_energy_ev_limit_ub_splitxp(m, i, t):
    """Charge limits for each EV at charging stations"""
    return m.xc[i, t] - m.xg[i, t] <= m.PMAX * m.xkappa[i, t]


def constraint_energy_ev_limit_ub(m, i, t):
    return m.xp[i, t] <= m.PMAX * m.xkappa[i, t]


def constraint_energy_station_limit_lb_splitxp(m, i, t):  # TODO: Combine duplicates to be within limits at each time
    """Charge limits for an EV at charging station i"""
    return m.SMIN[i] * m.xkappa[i, t] <= m.xc[i, t] - m.xg[i, t]


def constraint_energy_station_limit_lb(m, i, t):  # TODO: Combine duplicates to be within limits at each time
    """Charge limits for an EV at charging station i"""
    return m.SMIN[i] * m.xkappa[i, t] <= m.xp[i, t]


def constraint_energy_station_limit_ub_splitxp(m, i, t):  # TODO: Combine duplicates to be within limits at each time
    """Charge limits for an EV at charging station i"""
    return m.xc[i, t] - m.xg[i, t] <= m.SMAX[i] * m.xkappa[i, t]


def constraint_energy_station_limit_ub(m, i, t):  # TODO: Combine duplicates to be within limits at each time
    """Charge limits for an EV at charging station i"""
    return m.xp[i, t] <= m.SMAX[i] * m.xkappa[i, t]


def constraint_energy_start_end_soe_ij(m, i, j):
    """Start and end energy state must be equal for each EV"""
    return m.xa[i] == m.xa[j]


def constraint_energy_start_end_soe(m, i):
    """Start and end energy state must be equal for each EV"""
    return m.xa[i] == m.EMAX


def constraint_energy_soe(m, i):
    """Minimum and Maximum SOE limit for each EV"""
    return inequality(m.EMIN, m.xa[i], m.EMAX)


def constraint_energy_soe_station_splitxp(m, i, t):
    """Minimum and Maximum SOE limit for each EV"""
    return inequality(m.EMIN,
                      m.xa[i] + m.t_H * m.t_S * sum(m.eff * m.xc[i, b] - m.xg[i, b] / m.eff for b in m.T if b <= t),
                      m.EMAX)


def constraint_energy_soe_station(m, i, t):
    """Minimum and Maximum SOE limit for each EV"""
    return inequality(m.EMIN, m.xa[i] + m.t_H * m.t_S * sum(m.xp[i, b] for b in m.T if b <= t), m.EMAX)


# See this implementation example: https://stackoverflow.com/questions/53966482/how-to-map-different-indices-in-pyomo
def constraint_energy_peak_splitxp(m, s, t):
    """Peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xc[i, t] - m.xg[i, t] for i in m.Smap[s]) <= m.xd[s]


def constraint_inverse_energy_peak_xomega(m, s):
    """xomega sums to 1 across time for each physical station s ∈ S"""
    return sum(m.xomega[s, t] for t in m.T) == 1


def constraint_inverse_energy_peak_splitxp_ub(m, s, t):
    """Inverts peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xc[i, t] - m.xg[i, t] for i in m.Smap[s]) + m.MG * (1 - m.xomega[s, t]) >= m.xd[s]


def constraint_inverse_energy_peak_splitxp_lb(m, s, t):
    """Inverts peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xc[i, t] - m.xg[i, t] for i in m.Smap[s]) - m.MG * m.xomega[s, t] <= m.xd[s]


def constraint_energy_peak(m, s, t):
    """Peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xp[i, t] for i in m.Smap[s]) <= m.xd[s]


def constraint_inverse_energy_peak_ub(m, s, t):
    """Inverts peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xp[i, t] for i in m.Smap[s]) + m.MG * (1 - m.xomega[s, t]) >= m.xd[s]


def constraint_inverse_energy_peak_lb(m, s, t):
    """Inverts peak electric demand for each physical station s(i) ∈ S"""
    return m.G[s, t] + sum(m.xp[i, t] for i in m.Smap[s]) - m.MG * m.xomega[s, t] <= m.xd[s]


def constraint_no_export_splitxp(m, s, t):
    """Limits energy discharge to coincident load profile so no energy is exported"""
    return m.G[s, t] + sum(m.xc[i, t] - m.xg[i, t] for i in m.Smap[s]) >= 0


def constraint_no_export(m, s, t):
    """Limits energy discharge to coincident load profile so no energy is exported"""
    return m.G[s, t] + sum(m.xp[i, t] for i in m.Smap[s]) >= 0


# Payload Constraints
def constraint_payload(m, i, j):
    """Vehicles must unload payload for full customer demand when visiting a customer"""
    if i != j:
        return m.xq[j] <= m.xq[i] - (m.q[i] * m.xgamma[i, j]) + m.MQ * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_payload_station(m, i, j):
    """EV payload must not decrease when visiting a charging station"""
    if i != j:
        return m.xq[j] <= m.xq[i] + m.MQ * (1 - m.xgamma[i, j])
    else:
        return Constraint.Skip


def constraint_payload_limit(m, i):
    """Payload limits for each vehicle"""
    if i in m.start_node:  # Each vehicle must start with a full payload
        return m.xq[i] == m.QMAX
    else:
        return m.xq[i] <= m.QMAX  # xq is NonNegative


class EVRPTWV2G:
    """
    Implementation of E-VRP-TW-V2G
    Authors: Rami Ariss
    """

    def __init__(self, problem_type: str, dist_type: str='scipy', save_dist_matrix: bool=False):
        """
        :param problem_type: Objective options include: {Schneider} OR {OpEx, CapEx, Cycle, EA, DCM, Delivery, InverseEA, InverseDCM}
         Constraint options include: {Start=End, FullStart=End, NoXkappaBounds, NoMinVehicles, MaxVehicles, FixedVehicles, NoSymmetry, NoXd, SplitXp, StationaryEVs, NoExport, NoG}
        """
        self.problem_type = problem_type
        self.problem_types = self.problem_type.lower().split()

        self.dist_type = dist_type

        self.save_dist_matrix = save_dist_matrix

        self.model_build_start_time = datetime.datetime.now()

        # Instantiate pyomo Abstract Model
        self.m = AbstractModel()

        log.info('Building abstract model')
        self.build_model()

    def build_model(self):
        log.info('Defining parameters and sets')

        # Defining fixed parameters
        self.m.MQ = Param(doc='Large value for big M payload constraints')
        self.m.MT = Param(doc='Large value for big M service time constraints')
        self.m.ME = Param(doc='Large value for big M energy constraints')
        if 'inversedcm' in self.problem_types:
            self.m.MG = Param(doc='Large value for big M demand constraints')
        self.m.cc = Param(mutable=True, doc='Amortized capital cost for purchasing a vehicle')
        self.m.co = Param(mutable=True, doc='Amortized operating cost for delivery (wages)')
        self.m.cm = Param(mutable=True, doc='Amortized operating cost for maintenance')
        self.m.cy = Param(mutable=True, doc='Penalty cycle cost')
        self.m.QMAX = Param(mutable=True, doc='Maximum payload limit for all vehicles')
        self.m.EMAX = Param(mutable=True, doc='Maximum EV battery SOE limit for all EVs')
        self.m.EMIN = Param(mutable=True, doc='Minimum EV battery SOE limit for all EVs')
        self.m.PMAX = Param(mutable=True, doc='Maximum EV inverter limit for all EVs')
        self.m.N = Param(mutable=True, doc='Maximum number of EVs possible in fleet')
        self.m.r = Param(mutable=True, doc='Electric consumption per unit distance for EV')
        self.m.v = Param(mutable=True, doc='Average speed')
        self.m.t_T = Param(doc='Time horizon')
        self.m.t_S = Param(doc='Time step size')
        self.m.t_H = Param(doc='Hours per time step unit')
        if 'splitxp' in self.problem_types:
            self.m.eff = Param(doc='One-way efficiency')

        # Defining sets
        self.m.V01_ = Set(dimen=1, doc='All nodes extended')
        self.m.start_node = Set(within=self.m.V01_, doc='Starting node')
        self.m.end_node = Set(within=self.m.V01_, doc='Ending node')
        self.m.V0_ = Set(initialize=self.m.V01_ - self.m.end_node, doc='All nodes extended except ending depot node')
        self.m.V1_ = Set(initialize=self.m.V01_ - self.m.start_node, doc='All nodes extended except starting depot node')
        self.m.V_ = Set(initialize=self.m.V01_ - self.m.start_node - self.m.end_node,
                        doc='All nodes extended except start and end depot nodes')
        self.m.S_ = Set(dimen=1, doc='All charging station nodes extended')
        self.m.S = Set(dimen=1, doc='All charging station nodes')
        self.m.M = Set(dimen=1, doc='All customer nodes')
        self.m.T = Set(dimen=1, doc='Time')
        # self.m.E = Set(initialize=self.m.V01 * self.m.V01, within=self.m.V01 * self.m.V01, doc='Graph edges')

        self.m.E = Set(initialize=E_init, within=self.m.V01_ * self.m.V01_, doc='Graph edges')

        # Defining parameter sets
        self.m.d = Param(self.m.E, doc='Distance of edge (i;j) between nodes i;j (km)')
        self.m.q = Param(self.m.M, mutable=True, doc='Delivery demand at each customer')
        self.m.tS = Param(self.m.V01_, mutable=True, doc='Fixed service time for a vehicle at node i')
        self.m.tQ = Param(self.m.V01_, mutable=True, doc='Payload service time for a vehicle at node i')
        self.m.tA = Param(self.m.V01_, mutable=True, doc='Time window start time at node i')
        self.m.tB = Param(self.m.V01_, mutable=True, doc='Time window end time at node i')
        self.m.SMAX = Param(self.m.S_, mutable=True, doc='Maximum station inverter limit')
        self.m.SMIN = Param(self.m.S_, mutable=True, doc='Minimum station inverter limit')
        if 'nog' not in self.problem_types:
            self.m.G = Param(self.m.S, self.m.T, mutable=True, doc='Station electric demand profile')
            self.m.GMAX = Param(self.m.S, mutable=True, doc='Maximum station electric demand')
            self.m.cg = Param(self.m.S, mutable=True, doc='Amortized operating cost for demand charges')
        self.m.ce = Param(self.m.S, self.m.T, mutable=True, doc='Amortized operating cost for energy charges')
        self.m.cq = Param(self.m.M, mutable=True, doc='Amortized revenue for deliveries')
        self.m.Smap = Param(self.m.S, domain=Any, doc='Mapping from original charging station to all its duplicates')

        log.info('Defining variables')
        # Defining variables
        self.m.xgamma = Var(self.m.E, within=Boolean, initialize=0)  # Route decision of each edge for each EV
        self.m.xkappa = Var(self.m.S_, self.m.T, within=Boolean, initialize=0)
        if 'inversedcm' in self.problem_types:
            self.m.xomega = Var(self.m.S, self.m.T, within=Boolean, initialize=0)
        self.m.xw = Var(self.m.V01_, within=NonNegativeReals)  # Arrival time for each vehicle at each node
        self.m.xq = Var(self.m.V01_, within=NonNegativeReals)  # Payload of each vehicle before visiting each node
        self.m.xa = Var(self.m.V01_, within=NonNegativeReals, initialize=self.m.EMAX)  # Energy of each EV arriving at each node
        self.m.xd = Var(self.m.S, within=NonNegativeReals, initialize=0)  # Each station’s net peak electric demand
        self.m.xp = Var(self.m.S_, self.m.T, within=Reals)  # AC Discharge or charge rate of each EV
        if 'splitxp' in self.problem_types:
            self.m.xc = Var(self.m.S_, self.m.T, within=NonNegativeReals, initialize=0)  # AC Charge rate of each EV
            self.m.xg = Var(self.m.S_, self.m.T, within=NonNegativeReals, initialize=0)  # AC Discharge rate of each EV


        # %% ROUTING CONSTRAINTS
        #  TODO: upper bound on maximum number of vehicles in fleet based off starting routes?

        log.info('Defining constraints')

        self.m.constraint_visit_stations = Constraint(self.m.S_, rule=constraint_visit_stations)

        self.m.constraint_visit_customers = Constraint(self.m.M, rule=constraint_visit_customers)

        self.m.constraint_single_route = Constraint(self.m.V_, rule=constraint_single_route)

        if 'nominvehicles' not in self.problem_types:
            self.m.constraint_min_vehicles = Constraint(self.m.start_node, rule=constraint_min_vehicles)

        if 'maxvehicles' in self.problem_types:
            self.m.constraint_max_vehicles = Constraint(self.m.start_node, rule=constraint_max_vehicles)

        if 'fixedvehicles' in self.problem_types:
            self.m.constraint_fixed_vehicles = Constraint(self.m.start_node, rule=constraint_fixed_vehicles)

        if 'nosymmetry' not in self.problem_types:
            self.m.constraint_station_symmetry = Constraint(self.m.S_, rule=constraint_station_symmetry)

        # TODO: Need to implement so that doesn't pass through station at depot (This currently prohibits tours of type depot>any station>depot)
        if 'stationaryevs' not in self.problem_types:
            self.m.constraint_no_stationary_evs = Constraint(self.m.start_node, self.m.end_node, self.m.S_, rule=constraint_no_stationary_evs)

        self.m.constraint_no_visit_duplicate_station = Constraint(self.m.S, rule=constraint_no_visit_duplicate_station)

        # %% TIME CONSTRAINTS
        self.m.constraint_xgamma_xkappa = Constraint(self.m.S_, self.m.T, rule=constraint_xgamma_xkappa)

        self.m.constraint_time_start_depot = Constraint(self.m.start_node, self.m.V1_, rule=constraint_time_start_depot)

        self.m.constraint_time_station = Constraint(self.m.S_, self.m.V1_, rule=constraint_time_station)

        self.m.constraint_time_customer = Constraint(self.m.M, self.m.V1_, rule=constraint_time_customer)

        self.m.constraint_node_time_window = Constraint(self.m.V01_, rule=constraint_node_time_window)

        self.m.constraint_terminal_node_time = Constraint(self.m.end_node, rule=constraint_terminal_node_time)

        if 'noxkappabounds' not in self.problem_types:
            self.m.constraint_time_xkappa_lb = Constraint(self.m.S_, self.m.T, rule=constraint_time_xkappa_lb)

            self.m.constraint_time_xkappa_ub = Constraint(self.m.S_, self.m.V1_, self.m.T, rule=constraint_time_xkappa_ub)

        # ENERGY CONSTRAINTS]
        if 'splitxp' in self.problem_types:
            # self.m.constraint_xp = Constraint(self.m.S_, self.m.T, rule=constraint_xp)
            self.m.constraint_energy_station = Constraint(self.m.S_, self.m.V1_, rule=constraint_energy_station_splitxp)
            if 'exactsoe' in self.problem_types:
                self.m.constraint_energy_station_exact = Constraint(self.m.S_, self.m.V1_, rule=constraint_energy_station_exact_splitxp)
            self.m.constraint_energy_ev_limit_lb = Constraint(self.m.S_, self.m.T, rule=constraint_energy_ev_limit_lb_splitxp)
            self.m.constraint_energy_ev_limit_ub = Constraint(self.m.S_, self.m.T, rule=constraint_energy_ev_limit_ub_splitxp)
            self.m.constraint_energy_station_limit_lb = Constraint(self.m.S_, self.m.T, rule=constraint_energy_station_limit_lb_splitxp)
            self.m.constraint_energy_station_limit_ub = Constraint(self.m.S_, self.m.T, rule=constraint_energy_station_limit_ub_splitxp)
            self.m.constraint_energy_soe_station = Constraint(self.m.S_, self.m.T, rule=constraint_energy_soe_station_splitxp)
            if 'noxd' not in self.problem_types:
                if 'inversedcm' in self.problem_types:
                    self.m.constraint_inverse_energy_peak_xomega = Constraint(self.m.S, rule=constraint_inverse_energy_peak_xomega)
                    self.m.constraint_inverse_energy_peak_ub = Constraint(self.m.S, self.m.T, rule=constraint_inverse_energy_peak_splitxp_ub)
                    self.m.constraint_inverse_energy_peak_lb = Constraint(self.m.S, self.m.T, rule=constraint_inverse_energy_peak_splitxp_lb)
                else:
                    self.m.constraint_energy_peak = Constraint(self.m.S, self.m.T, rule=constraint_energy_peak_splitxp)
            if 'zeroxg' in self.problem_types:
                self.m.constraint_energy_zero_xg_splitxp = Constraint(self.m.S_, self.m.T, rule=constraint_energy_zero_xg_splitxp)
            if 'noexport' in self.problem_types:
                self.m.constraint_no_export = Constraint(self.m.S, self.m.T, rule=constraint_no_export_splitxp)
        else:
            self.m.constraint_energy_station = Constraint(self.m.S_, self.m.V1_, rule=constraint_energy_station)
            if 'exactsoe' in self.problem_types:
                self.m.constraint_energy_station_exact = Constraint(self.m.S_, self.m.V1_, rule=constraint_energy_station_exact)
            self.m.constraint_energy_ev_limit_lb = Constraint(self.m.S_, self.m.T, rule=constraint_energy_ev_limit_lb)
            self.m.constraint_energy_ev_limit_ub = Constraint(self.m.S_, self.m.T, rule=constraint_energy_ev_limit_ub)
            self.m.constraint_energy_station_limit_lb = Constraint(self.m.S_, self.m.T, rule=constraint_energy_station_limit_lb)
            self.m.constraint_energy_station_limit_ub = Constraint(self.m.S_, self.m.T, rule=constraint_energy_station_limit_ub)
            self.m.constraint_energy_soe_station = Constraint(self.m.S_, self.m.T, rule=constraint_energy_soe_station)
            if 'noxd' not in self.problem_types:
                if 'inversedcm' in self.problem_types:
                    self.m.constraint_inverse_energy_peak_xomega = Constraint(self.m.S, rule=constraint_inverse_energy_peak_xomega)
                    self.m.constraint_inverse_energy_peak_ub = Constraint(self.m.S, self.m.T, rule=constraint_inverse_energy_peak_ub)
                    self.m.constraint_inverse_energy_peak_lb = Constraint(self.m.S, self.m.T, rule=constraint_inverse_energy_peak_lb)
                else:
                    self.m.constraint_energy_peak = Constraint(self.m.S, self.m.T, rule=constraint_energy_peak)
            if 'noexport' in self.problem_types:
                self.m.constraint_no_export = Constraint(self.m.S, self.m.T, rule=constraint_no_export)

        self.m.constraint_energy_customer = Constraint(self.m.M | self.m.start_node, self.m.V1_, rule=constraint_energy_customer)
        if 'exactsoe' in self.problem_types:
            self.m.constraint_energy_customer_exact = Constraint(self.m.M | self.m.start_node, self.m.V1_,  rule=constraint_energy_customer_exact)

        if 'start=end' in self.problem_types:
            self.m.constraint_energy_start_end_soe = Constraint(self.m.start_node, self.m.end_node, rule=constraint_energy_start_end_soe_ij)
        elif 'fullstart=end' in self.problem_types:
            self.m.constraint_energy_start_end_soe = Constraint(self.m.start_node | self.m.end_node, rule=constraint_energy_start_end_soe)
        else:
            self.m.constraint_energy_start_end_soe = Constraint(self.m.start_node, rule=constraint_energy_start_end_soe)

        self.m.constraint_energy_soe = Constraint(self.m.V01_, rule=constraint_energy_soe)

        # PAYLOAD CONSTRAINTS
        self.m.constraint_payload = Constraint(self.m.M, self.m.V1_, rule=constraint_payload)

        self.m.constraint_payload_station = Constraint(self.m.S_, self.m.V1_, rule=constraint_payload_station)

        self.m.constraint_payload_limit = Constraint(self.m.V01_, rule=constraint_payload_limit)

        # OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS
        self.m.obj = Objective(rule=self.obj, sense=minimize)

    # OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS
    def obj(self, m):
        """Objective: maximize net profits"""
        log.info('Problem type: {}'.format(self.problem_type))

        # Construct objective function
        if 'schneider' in self.problem_types:
            obj_result = self.total_distance(m) + self.C_fleet_capital_cost(m)
        else:
            # Initialize
            obj_result = 0

            if 'distance' in self.problem_types:
                obj_result += self.total_distance(m)
            if 'opex' in self.problem_types:
                obj_result += self.O_maintenance_operating_cost(m)
                obj_result += self.O_delivery_operating_cost(m)
            if 'capex' in self.problem_types:
                obj_result += self.C_fleet_capital_cost(m)
            if 'cycle' in self.problem_types:
                obj_result += self.cycle_cost(m)
            if 'ea' in self.problem_types:
                obj_result -= self.R_energy_arbitrage_revenue(m)
            if 'dcm' in self.problem_types:
                obj_result -= self.R_peak_shaving_revenue(m)
            if 'delivery' in self.problem_types:
                obj_result -= self.R_delivery_revenue(m)
            if 'inverseea':
                obj_result += self.R_energy_arbitrage_revenue(m)
            if 'inversedcm' in self.problem_types:
                obj_result += self.R_peak_shaving_revenue(m)

        return obj_result

    def C_fleet_capital_cost(self, m):
        """Cost of total number vehicles"""
        return m.cc * sum(sum(m.xgamma[i, j] for j in m.V1_ if j != i) for i in m.start_node)

    def O_delivery_operating_cost(self, m):
        """Amortized delivery operating cost for utilized vehicles (wages)"""
        return m.co * self.total_time(m)

    def O_maintenance_operating_cost(self, m):
        """Amortized delivery maintenance cost for utilized vehicles"""
        return m.cm * self.total_distance(m)

    def R_peak_shaving_revenue(self, m):
        """Amortized V2G peak shaving demand charge savings (or net demand charge cost) over all stations"""
        return sum(m.cg[s] * (m.GMAX[s] - m.xd[s]) for s in m.S)

    def R_energy_arbitrage_revenue(self, m):
        """Amortized G2V/V2G energy arbitrage (or net cost of charging) over all charging stations"""
        if 'splitxp' in self.problem_types:
            return -m.t_H * m.t_S * sum(sum(sum(m.ce[s, t] * (m.xc[i, t] - m.xg[i, t]) for t in m.T) for i in m.Smap[s]) for s in m.S)
        else:
            return -m.t_H * m.t_S * sum(sum(sum(m.ce[s, t] * m.xp[i, t] for t in m.T) for i in m.Smap[s]) for s in m.S)

    def R_delivery_revenue(self, m):
        """Amortized delivery revenue for goods delivered to customer nodes by entire fleet"""
        return sum(m.cq[i] * m.q[i] for i in m.M)

    def total_distance(self, m):
        """Total traveled distance"""
        return sum(sum(m.d[i, j] * m.xgamma[i, j] for i in m.V0_ if i != j) for j in m.V1_)

    def total_time(self, m):
        """Total time traveled by all EVs"""
        return self.total_distance(m) / m.v

    def cycle_cost(self, m):
        """Adds a penalty for any battery actions."""
        # return m.cy * m.t_S * sum(sum(sum(m.xkappa[i, t] * m.xgamma[i, j] for t in m.T) for j in m.V1_ if i != j) for i in m.S_)
        if 'splitxp' in self.problem_types:
            return m.cy * m.t_H * m.t_S * sum(sum(m.xc[i, t] for t in m.T) for i in m.S_)
        else:
            return m.cy * m.t_H * m.t_S * sum(sum(m.xkappa[i, t] for t in m.T) for i in m.S_)

    # TODO: FIX t_S unit size so no decimals are needed
    def create_data_dictionary(self):
        self.p = {
            None: {
                'MQ': {None: float(self.data['W'].loc[:, 'QMAX'].max())},  # Parameters
                'MT': {None: self.data['Parameters'].loc['t_T', 'value']},
                'ME': {None: float(self.data['W'].loc[:, 'EMAX'].max())},
                'cc': {None: self.data['W'].loc[:, 'cc'].mean()},
                'co': {None: self.data['W'].loc[:, 'co'].mean()},
                'cm': {None: self.data['W'].loc[:, 'cm'].mean()},
                'cy': {None: self.data['W'].loc[:, 'cy'].mean()},
                'eff': {None: self.data['W'].loc[:, 'eff'].mean()},
                'QMAX': {None: float(self.data['W'].loc[:, 'QMAX'].mean())},
                'EMIN': {None: float(self.data['W'].loc[:, 'EMIN'].mean())},
                'EMAX': {None: float(self.data['W'].loc[:, 'EMAX'].mean())},
                'PMAX': {None: float(self.data['W'].loc[:, 'PMAX'].mean())},
                'N': {None: self.data['W'].loc[:, 'N'].sum()},
                'r': {None: self.data['W'].loc[:, 'r'].mean()},
                'v': {None: float(self.data['W'].loc[:, 'v'].mean())},
                'V01_': {None: self.data['V_'].index.values},  # Sets
                'start_node': {None: [self.data['start_node']]},
                'end_node': {None: [self.data['end_node']]},
                'S_': {None: self.data['S_'].index.values},  # Sets
                'S': {None: self.data['S'].index.values},  # Sets
                'M': {None: self.data['M'].index.values},  # Sets
                'd': {(k[0], k[1]): d for k, d in self.data['d'].stack().iteritems()
                      if k[0] != k[1] if k[0] != self.data['end_node'] if k[1] != self.data['start_node']},  # Parameter Sets  # self.data['d'].stack().to_dict()
                'q': self.data['M']['q'].to_dict(),
                'tS': self.data['V_']['tS'].to_dict(),
                'tQ': self.data['V_']['tQ'].dropna().to_dict(),
                'tA': self.data['V_']['tA'].to_dict(),
                'tB': self.data['V_']['tB'].to_dict(),
                'SMIN': self.data['S_']['SMIN'].to_dict(),
                'SMAX': self.data['S_']['SMAX'].to_dict(),
                'ce': self.data['ce'].T.stack().to_dict(),
                'cq': self.data['M']['cq'].to_dict(),
                'Smap': self.s2s_,
                't_T': {None: self.t_T},
                't_S': {None: self.t_S},
                't_H': {None: self.t_H},
                'T': {None: list(range(0, self.t_T, self.t_S))}
            }
        }

        if 'G' in self.data['T'].columns.get_level_values(0).drop_duplicates():
            self.p[None].update({
                'G':  self.data['G'].T.stack().to_dict(),
                'GMAX': self.data['G'].max().to_dict(),
                'cg': self.data['S']['cg'].to_dict()})

        if 'inversedcm' in self.problem_types:
            self.p[None].update({
                'MG': {None: max(self.p[None]['GMAX'].values()) + self.p[None]['N'][None] * self.p[None]['PMAX'][None]},
            })

    def import_instance(self, instance_filepath: str):
        self.instance_filepath = instance_filepath
        self.instance_name = instance_filepath.split('/')[-1].split('.')[0]
        log.info('Importing EVRPTW MILP instance: {}'.format(self.instance_name))

        # Read in csv
        log.info('Reading CSV')
        self.data = parse_csv_tables(instance_filepath)

        # Create graph by concatenating D, S, M nodes
        log.info('Creating graph')
        node_types = [k for k in self.data.keys() if k in 'DSM']
        self.data['V'] = pd.concat([self.data[k] for k in node_types])

        # TODO: Bring upstream for user passthrough
        # Define start and end nodes
        self.data['start_node'] = self.data['D'].index[0]
        self.data['end_node'] = self.data['D'].index[1]

        # TODO: Need to define two different timestep variables (1) for dividing up timeseries data and (2) for converting into hour units
        # Create timeseries data
        log.info('Creating timeseries data')
        self.t_T = self.data['Parameters'].loc['t_T', 'value'].astype(int)
        self.t_S = self.data['Parameters'].loc['t_S', 'value'].astype(int)
        self.t_H = self.data['Parameters'].loc['t_H', 'value'].astype(float)

        for col in self.data['T'].columns.get_level_values(0).drop_duplicates():  # Produces energy price (ce) and demand (G) timeseries
            self.data[col] = self.data['T'][col].reindex(range(0, self.t_T, self.t_S), method='ffill')

        # Create duplicates of charging stations
        log.info('Creating duplicates and extended graph')
        self.data['S_'] = self.data['S'].reindex(self.data['S'].index.repeat(self.data['S']['instances']))
        self.data['S_'].set_index(pd.Index(self.data['S_'].index + '_' + self.data['S_'].groupby(['node_id']).cumcount().astype(str)), inplace=True)

        # Create extended graph
        self.data['V_'] = pd.concat([self.data[k] for k in ['D', 'S_', 'M']])

        # Calculate distance matrix
        log.info('Calculating distance matrix')
        self.data['d'] = calculate_distance_matrix(self.data['V_'][['d_x', 'd_y']], dist_type=self.dist_type,
                                                   instance=self.instance_name, save=self.save_dist_matrix)

        # Create duplicates mappings
        self.s2s_ = {s: [s_ for s_ in self.data['S_'].index if s+'_' in s_] for s in self.data['S'].index}  # S -> S_
        self.s_2s = {s_: s for s in self.data['S'].index for s_ in self.data['S_'].index if s+'_' in s_}  # S_ -> S

        # Generate index mappings
        self.i2v, self.v2i = generate_index_mapping(self.data['V'].index)

        # Create data dictionary for concrete model instance (parameters)
        log.info('Creating parameters')
        self.create_data_dictionary()

    def make_instance(self, add_to_instance_name=''):
        # Create a ConcreteModel instance of the AbstractModel passing in parameters
        log.info('Creating instance')
        self.instance = self.m.create_instance(data=self.p)
        if len(add_to_instance_name) > 0:
            add_to_instance_name = ' ' + add_to_instance_name

        self.instance.problem_type = self.problem_type
        self.instance.dist_type = self.dist_type
        self.instance.instance_name = f'{self.instance_name}{add_to_instance_name}'
        self.instance.name = '{} {}{}'.format(self.instance_name, self.problem_type, add_to_instance_name)
        self.instance.instance_filepath = self.instance_filepath if hasattr(self, 'instance_filepath') else None
        self.instance.s2s = self.s2s if hasattr(self, 's2s') else None
        self.instance.s_2s = self.s_2s if hasattr(self, 's_2s') else None
        self.instance.i2v = self.i2v if hasattr(self, 'i2v') else None
        self.instance.v2i = self.v2i if hasattr(self, 'v2i') else None

        self.model_build_end_time = datetime.datetime.now()

    # For Gurobi solver options, see: https://www.gurobi.com/documentation/9.5/refman/parameters.html
    def make_solver(self, solve_options={'TimeLimit': 60 * 5}):  #'MIPGap': 1e-2, 'MIPFocus': 3, 'Cuts': 3
        # Specify solver
        self.opt = SolverFactory(SOLVER_TYPE, io_format='python')

        # Solver options
        self.solve_options = solve_options

    def full_solve(self, instance_filepath: str):
        # Import data
        self.import_instance(instance_filepath)

        # Create an instance of the AbstractModel passing in parameters
        self.make_instance()

        # Create solver
        self.make_solver()

        # Solve instance
        log.info('Solving instance...')
        self.results = self.opt.solve(self.instance, tee=True, options=self.solve_options)
        if 'splitxp' in self.problem_types:
            self.set_xp()
        log.info('Done')

        self.model_solve_end_time = datetime.datetime.now()

        self.model_build_duration = (self.model_build_end_time - self.model_build_start_time).total_seconds()
        self.model_solve_duration = (self.model_solve_end_time - self.model_build_end_time).total_seconds()

    def solve(self):
        if not (hasattr(self, 'opt') | hasattr(self, 'solve_options')):
            log.info('Making solver...')
            self.make_solver()

        # Solve instance
        log.info('Solving instance...')
        self.results = self.opt.solve(self.instance, tee=True, options=self.solve_options)
        if 'splitxp' in self.problem_types:
            self.set_xp()
        log.info('Done')

        self.model_solve_end_time = datetime.datetime.now()

        self.model_build_duration = (self.model_build_end_time - self.model_build_start_time).total_seconds()
        self.model_solve_duration = (self.model_solve_end_time - self.model_build_end_time).total_seconds()

    def warmstart_solve(self):
        if not (hasattr(self, 'opt') | hasattr(self, 'solve_options')):
            log.info('Making solver...')
            self.make_solver()

        # Solve instance
        log.info('Solving instance with warmstart...')
        self.results = self.opt.solve(self.instance, tee=True, warmstart=True, options=self.solve_options)
        if 'splitxp' in self.problem_types:
            self.set_xp()
        log.info('Done')

    def archive_instance_result(self, add_to_key_name=''):
        # Initialize
        if not hasattr(self, 'instance_archive'):
            self.instance_archive = {}
        if not hasattr(self, 'results_archive'):
            self.results_archive = {}

        # Archive current instance and results
        key = '{}{}'.format(self.instance.name, add_to_key_name)
        self.instance_archive[key] = self.instance.clone()
        self.results_archive[key] = self.results.copy()

    def delete_instance_result(self):
        del self.instance, self.results

    def delete_instance_result_archive(self):
        del self.instance_archive, self.results_archive

    def fix_variables(self, var_list: 'list of str'):
        for v in var_list:
            getattr(self.instance, v).fix()

    def free_variables(self, var_list: 'list of str'):
        for v in var_list:
            getattr(self.instance, v).free()

    def remake_objective(self, instance, new_problem_type: str):
        """Creates a new objective function given a new problem type."""
        self.problem_type = new_problem_type
        instance.name = new_problem_type

        instance.del_component('obj')
        instance.add_component('obj', Objective(rule=self.obj, sense=minimize))

    def remake_model(self, new_problem_type: str, fpath: str, add_to_instance_name=''):
        """Remakes a fresh AbstractModel and a new instance. Must do this if constraint options are changed."""
        self.problem_type = new_problem_type

        self.m = AbstractModel()
        self.build_model()

        self.import_instance(fpath)

        self.make_instance(add_to_instance_name)

    def set_xgamma(self, archived_instance_name: str):
        """Set's the current instance's xgamma values to the xgamma of an archived instance."""
        self.instance.xgamma.set_values(
            {k: round(v) for k, v in self.instance_archive[archived_instance_name].xgamma.extract_values().items()})

    def set_xp(self):
        """Set's the current instance's xp values to xc - xg."""
        xc = self.instance.xc.extract_values()
        xg = self.instance.xg.extract_values()
        self.instance.xp.set_values(
            {k: xc[k]-xg[k] for k in self.instance.xp.extract_values().keys()})

    def set_xd(self):
        """Set's the current instance's xd values to max of G + (xc - xg)."""
        # Get the pre-fleet load
        G = pd.Series(self.instance.G.extract_values())
        G = G.reset_index().pivot(index='level_1', columns='level_0', values=0)

        # Get the fleet load
        xp = self.instance.xp.extract_values()

        # Calculate the post-fleet load
        G_xd = G.copy()
        for s in G:
            for t in G[s].index:
                G_xd.loc[t, s] = G[s][t] + sum(xp[(i, t)] for i in self.instance.Smap[s])

        # Set xd to the max of the pre-fleet and post-fleet load
        xd = G_xd.max()
        self.instance.xd.set_values({k: xd[k] for k in self.instance.xd.extract_values().keys()})
