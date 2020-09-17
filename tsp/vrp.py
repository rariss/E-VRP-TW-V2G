"""
ELECTRIC VEHICLE ROUTING PROBLEM WITH VEHICLE-TO-GRID AND HETEROGENEOUS FLEET DESIGN
(E-VRP-V2GHFD)
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
m.V_ = Set(doc='All nodes extended')

# Defining parameter sets
m.d = Param(m.V_, m.V_, doc='Distance of edge (i;j) between nodes i;j (km)') 
m.Arcs = Set(initialize=arcs, doc='All possible arcs')

# Defining variables
m.x = Var(m.V_, m.V_, within=Boolean)  # Route decision of each edge for each EV

# %% OBJECTIVE FUNCTION AND DEPENDENT FUNCTIONS

def obj_total_distance(m):
    """Objective: total traveled distance"""
    return sum (sum (m.d[i,j] * m.x[i,j] for i in m.V_) for j in m.V_)

# Create objective function in Abstractm
m.obj = Objective(rule=obj_total_distance, sense=minimize)


# %% ROUTING CONSTRAINTS

def constraint_single_visit(m, j):
    """Nodes can be visited at most once in the graph"""
    # return sum(x[u] for u in m.V_*m.V_) <= 1
    # return sum(sum(m.x[i,j] for i in m.V_) for j in m.V_) == 1
    # return sum(m.x[i,j] for i in m.V_ if (i,j) in m.Arcs) == 1
    return sum(m.x[i,j] for i in m.V_ if i!=j) == 1


# Create single visit constraint
m.constraint_single_visit = Constraint(m.V_, rule=constraint_single_visit)


def constraint_single_route(m, i):
    """At most one route assigned"""
    
    # route_in = sum(m.x[j,i] for j in m.V_ if (j,i) in m.Arcs)
    
    route_in = sum(m.x[j,i] for j in m.V_)

    route_out = sum(m.x[i,j] for j in m.V_)

    return route_in - route_out == 0 


# def constraint_single_route(m, i):
#     """At most one route assigned"""
    
    
    
#     print('===================================================================', i)
#     a = arcs
#     print('\n\n\n\n\n', a)
    
#     b = [c for c in arcs if c[1] != i]
#     print('\n\n\n\n\n', b)
    
    
#     route_in = sum(m.x[j,i] for j in set(nodes) if (j,i) in b)

     
#     # if i != 'D0':
#     route_out = sum(m.x[i,j] for j in set(nodes) if (i,j) in a)
#     # route_out = sum(m.x[j,i] for j in set(nodes)-{i}) #if (j,i) in b)
        
#     # else:
#     #     route_in = sum(m.x[i,j] for j in set(nodes)) # if (i,j) in a)
#     #     route_out = sum(m.x[j,i] for j in set(nodes)) #if (j,i) in b)

#     return route_in - route_out == 0 

# Create single route constraint
m.constraint_single_route = Constraint(m.V_, rule=constraint_single_route)

def constraint_try(m,i,j):

    return sum(m.x[j,i] for j in [j]) - sum(m.x[i,j] for j in [j]) == 1

# m.constraint_try = Constraint(m.V_,m.V_, rule=constraint_try)


# %% TIME CONSTRAINTS


# %% ENERGY CONSTRAINTS


# %% PAYLOAD CONSTRAINTS

    
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

if __name__=="__main__":
    main()
