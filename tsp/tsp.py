"""
Implementation of a traveling salesman problem
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

# Defining sets
m.V01 = Set(doc='All nodes extended')
m.V0 = Set(doc='All nodes extended')
m.V1 = Set(doc='All nodes extended')
m.V = Set(doc='All nodes extended')

# Defining parameter sets
m.d = Param(m.V01, m.V01, doc='Distance of edge (i;j) between nodes i;j (km)')
m.Arcs = Set(initialize=arcs, doc='All possible arcs')

# Defining variables
m.x = Var(m.V01, m.V01, within=Boolean)  # Route decision of each edge for each EV

# Dummy variable ui
m.u = Var(m.V01, within=NonNegativeIntegers, bounds=(0, 9 - 1))
#TODO: read node number from instance file


# %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

def obj_total_distance(m):
    """Objective: total traveled distance"""
    return sum(sum(m.d[i, j] * m.x[i, j] for i in m.V01) for j in m.V01)


# Create objective function in Abstractm
m.obj = Objective(rule=obj_total_distance, sense=minimize)


# %% ROUTING CONSTRAINTS

def constraint_single_visit(m, j):
    """Nodes can be visited at most once in the graph"""
    return sum(m.x[i, j] for i in m.V01 if i != j) == 1

# Create single visit constraint
m.constraint_single_visit = Constraint(m.V01, rule=constraint_single_visit)

def constraint_single_route(m, i):
    """At most one route assigned"""

    route_in = sum(m.x[j, i] for j in m.V01)
    route_out = sum(m.x[i, j] for j in m.V01)

    return route_in - route_out == 0

# Create single route constraint
m.constraint_single_route = Constraint(m.V01, rule=constraint_single_route)


def constraint_single_subtour(m, i, j):

    if i!=j:
        return m.u[i] - m.u[j] + m.x[i,j] * 9 <= 9-1
    else:
        return Constraint.Skip

m.constraint_single_subtour = Constraint(m.V, m.V01, rule=constraint_single_subtour)


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
    solution_plotter(nodes, nodes_coords, routes)

    print('\nExecution time: %s seconds' % (time.time() - start_time))

    return m


if __name__ == "__main__":
    main()
