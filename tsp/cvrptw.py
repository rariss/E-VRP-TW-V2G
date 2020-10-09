"""
Implementation of a capacitated vehicle routing problem with time windows
By: Leandre Berwa
"""

from pyomo.environ import *
from config.default_params1 import *
from utils.utilities import *
import time

# Track execution time
start_time = time.time()

# DEFINING THE MODEL, PARAMETERS AND SETS

m = AbstractModel()

# Defining fixed parameters
m.Mq = Param(doc='Large value for big M payload constraints')
m.Mt = Param(doc='Large value for big M service time constraints')

# Defining sets
m.V01_ = Set(doc='All nodes extended')
m.V0_ = Set(doc='All nodes extended except starting depot node')
m.V1_ = Set(doc='All nodes extended except ending depot node')
m.V_ = Set(doc='All nodes extended except depot nodes')
m.V = Set(doc='All customer nodes')
m.F = Set(doc='All charging station nodes')

# Defining parameter sets
m.d = Param(m.V01_, m.V01_, doc='Distance of edge (i;j) between nodes i;j (km)')
m.Arcs = Set(initialize=arcs, doc='All possible arcs') #TODO: move set to param dict
m.q = Param(m.V01_, doc='Delivery demand at each customer')
m.tS = Param(m.V01_, doc='Fixed service time for a vehicle at node i')
m.tA = Param(m.V01_, doc='Time window start time at node i ')
m.tB = Param(m.V01_, doc='Time window end time at node i ')

# Defining variables
m.x = Var(m.V01_, m.V01_, within=Boolean)  # Route decision of each edge for each EV
m.xq = Var(m.V01_, within=NonNegativeIntegers) # Payload of each vehicle before visiting each node
m.xw = Var(m.V01_, within=NonNegativeIntegers) # Arrival time for each vehicle at each node

# %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

def obj_total_distance(m):
    """Objective: total traveled distance"""
    return sum(sum(m.d[i,j] * m.x[i, j]  for i in m.V0_) for j in m.V1_)
    # return sum(sum((m.d[i,j] - m.xq[j]) * m.x[i, j]  for i in m.V01_) for j in m.V01_)


# Create objective function
m.obj = Objective(rule=obj_total_distance, sense=minimize)


# %% ROUTING CONSTRAINTS

def constraint_single_visit(m, i):
    """Nodes can be visited at most once in the graph"""
    return sum(m.x[i, j] for j in m.V1_ if i != j) == 1 

# Create single visit constraint
m.constraint_single_visit = Constraint(m.V, rule=constraint_single_visit) 

def constraint_single_visit_station(m, i):
    """Nodes can be visited at most once in the graph"""
    return sum(m.x[i, j] for j in m.V1_ if i != j) == 1

# Create single visit constraint
m.constraint_single_visit_station = Constraint(m.F, rule=constraint_single_visit_station)


def constraint_single_route(m, j):
    """At most one route assigned"""

    route_in = sum(m.x[i, j] for i in m.V0_)
    route_out = sum(m.x[j, i] for i in m.V1_)

    return route_out - route_in == 0

# Create single route constraint
m.constraint_single_route = Constraint(m.V_, rule=constraint_single_route)


def constraint_route_number(m, i):
    return sum(m.x[i, j] for j in m.V1_) <= 1

# m.constraint_route_number = Constraint({'D0'}, rule=constraint_route_number)


# %% TIME CONSTRAINTS

def constraint_time(m, i, j):
    return m.xw[i] + (m.tS[i] + m.d[i,j]) * m.x[i,j] - 1236 * (1 - m.x[i,j]) <= m.xw[j]

m.contraint_time = Constraint(m.V0_, m.V1_, rule=constraint_time)

def constraint_node_time_window(m, i):
    return m.tA[i] <= m.xw[i] <= m.tB[i]

m.constraint_node_time_window = Constraint(m.V01_, rule=constraint_node_time_window)


# %% ENERGY CONSTRAINTS


# %% PAYLOAD CONSTRAINTS 

def constraint_payload(m, i, j):
    '''Vehicle must unload payload or full customer demand when visiting a customer'''
    
    return m.xq[j] <= m.xq[i] - (m.q[i] * m.x[i,j]) + 200 * (1 - m.x[i,j]) 

m.constraint_payload = Constraint(m.V0_, m.V1_, rule=constraint_payload)

def constraint_payload_limit(m, i):
    ''' Payload limits for each vehicle'''
    
    return 0 <= m.xq[i] <= 200

m.constraint_payload_limit = Constraint({'D0'}, rule=constraint_payload_limit)

def constraint_initial_payload(m, i):
    ''''Each vehicle must start with a full payload'''
    
    return m.xq[i] == 200

# m.constraint_initial_payload = Constraint({'D0'}, rule=constraint_initial_payload)

# %% SOLUTION

def main():
    # Specify solver
    opt = SolverFactory('gurobi')

    # Create an instance of the AbstractModel passing in parameters
    instance = m.create_instance(p)

    # Solve instance
    opt.solve(instance, tee=True)

    # Print, save and plot results
    instance.pprint()
    routes = solution_saver(instance, instance_name)
    solution_plotter(nodes, nodes_coords, routes, instance, node_start_time, node_end_time) #TODO
    
    xq = getattr(instance, 'xq').extract_values()
    xq_sorted = {k: v for k, v in sorted(xq.items(), key=lambda item: item[1], reverse=True)}
    print('\nxq:', xq_sorted)
    

    xw = getattr(instance, 'xw').extract_values()
    xw_sorted = {k: v for k, v in sorted(xw.items(), key=lambda item: item[1])}
    print('\nxw:', xw_sorted)
    

    print('\nExecution time: %s seconds' % (time.time() - start_time))

    return m


if __name__ == "__main__":
    main()
