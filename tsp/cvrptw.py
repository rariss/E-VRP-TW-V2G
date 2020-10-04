"""
Implementation of a vehicle routing problem with time windows
By: Leandre Berwa
"""

from pyomo.environ import *
from config.default_params import *
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
m.V01 = Set(doc='All nodes extended')
m.V0 = Set(doc='All nodes extended')
m.V1 = Set(doc='All nodes extended')
m.V = Set(doc='All nodes extended')

# Defining parameter sets
m.d = Param(m.V01, m.V01, doc='Distance of edge (i;j) between nodes i;j (km)')
m.Arcs = Set(initialize=arcs, doc='All possible arcs') #TODO: move set to param dict
m.q = Param(m.V01, doc='Delivery demand at each customer')
m.tS = Param(m.V01, doc='Fixed service time for a vehicle at node i')
m.tA = Param(m.V01, doc='Time window start time at node i ')
m.tB = Param(m.V01, doc='Time window end time at node i ')

# Defining variables
m.x = Var(m.V01, m.V01, within=Boolean)  # Route decision of each edge for each EV
m.xq = Var(m.V01, within=NonNegativeIntegers)
m.xw = Var(m.V01, within=NonNegativeIntegers)

# %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

def obj_total_distance(m):
    """Objective: total traveled distance"""
    return sum(sum(m.d[i, j] * m.x[i, j] for i in m.V01) for j in m.V01)


# Create objective function
m.obj = Objective(rule=obj_total_distance, sense=minimize)


# %% ROUTING CONSTRAINTS

def constraint_single_visit(m, j):
    """Nodes can be visited at most once in the graph"""
    return sum(m.x[i, j] for i in m.V0 if i != j) == 1


# Create single visit constraint
m.constraint_single_visit = Constraint(m.V01, rule=constraint_single_visit)


def constraint_single_route(m, i):
    """At most one route assigned"""

    route_in = sum(m.x[j, i] for j in m.V01)
    route_out = sum(m.x[i, j] for j in m.V01)

    return route_in - route_out == 0


# Create single route constraint
m.constraint_single_route = Constraint(m.V0, rule=constraint_single_route)


def constraint_route_number(m, i):
    return sum(m.x[i, j] for j in m.V0) <= 1

m.constraint_route_number = Constraint({'D0'}, rule=constraint_route_number)


# %% TIME CONSTRAINTS
# def constraint_time(m, i, j):
#     return m.xw[j] >= m.xw[i] + (m.tS[i] + m.t[i,j]) * m.x[i,j] - m.Mt * (1 - m.x[i,j])
#
# m.contraint_time = Constraint(m.V0, m.V01, rule=constraint_time)
#
# def constraint_node_time_window(m, i):
#     return m.tA[i] <= m.xw[i] <= m.tB[i]
#
# m.constraint_node_time_window = Constraint(m.V01, rule=constraint_node_time_window)


# %% ENERGY CONSTRAINTS


# %% PAYLOAD CONSTRAINTS

#TODO: payload constraints are supposed to instill a single subtour, or subtours including the starting depot 
def constraint_capacity(m, i, j):
    if i!=j:
        return m.xq[i] + m.q[j] * m.x[i,j] - m.Mq(1 - m.x[i,j]) <= m.xq[j]
    else:
        return Constraint.Skip

m.constraint_capacity = Constraint(m.V01, m.V01, rule=constraint_capacity)

def constraint_vehicle_capacity(m, i):
    return m.xq[i] <= m.Mq

m.constraint_vehicle_capacity = Constraint(m.V01, rule=constraint_vehicle_capacity)

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
    # routes = solution_saver(instance, instance_name)
    # solution_plotter(nodes, nodes_coords, routes)

    print('\nExecution time: %s seconds' % (time.time() - start_time))

    return m


if __name__ == "__main__":
    main()
