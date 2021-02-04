# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 08:03:08 2021

@author: berwa

Created this file only to test stuff with Spyder, 
the environment I'm most confortable in.
Working code will be moved to the notebook or the main file.

"""

# ------
# temporary imports

import logging
from pyomo.environ import *
logging.root.setLevel(logging.NOTSET) # or logging.basicConfig(level=logging.NOTSET)

from config.default_params import create_data_dictionary
from utils.utilities import import_instance
# ------

import sys
sys.path.append("../") # go to parent dir

%load_ext autoreload
%autoreload 2

from evrptw import EVRPTW

from utils.graph import draw_plain_graph, draw_graph
from utils.plot import plot_interactive_graph
from utils.utilities import create_optimal_edges, merge_variable_results
from utils.utilities import parse_csv_tables, calculate_distance_matrix, generate_index_mapping
from utils.utilities import trace_routes
import pandas as pd

# instance = 'r104C5'
# instance = 'r105C5'
# instance = 'r201C10'
# instance = 'rc108C10'
instance = 'c103C15'
# instance = 'c202C15'
# instance = 'rc103C15'
# instance = 'r105C15'
# instance = 'r202C15'
# instance = 'r102C15'

fpath = 'data/' + instance + '.csv'

m_run = EVRPTW()

duplicates = False
# m_run.solve(fpath, duplicates)

#%% testing initializing instances, solve function

# Specify solver
opt = SolverFactory('gurobi')

# Import data
instance_filepath = fpath
m_run.instance_name = instance_filepath.split('/')[-1].split('.')[0]
logging.info('Importing VRPTW MILP instance: {}'.format(m_run.instance_name))

m_run.data = import_instance(instance_filepath, duplicates)

# Create parameters
logging.info('Creating parameters')
p = create_data_dictionary(m_run.data)

# Create an instance of the AbstractModel passing in parameters
logging.info('Creating instance')
m_run.instance = m_run.m.create_instance(p)

# Solver options
solv_options = {'timelimit': 5} # fix a time limit to create suboptimal solution
    
# Solve instance
logging.info('Solving instance...')
# results = opt.solve(m_run.instance, tee=True)
results = opt.solve(m_run.instance, tee=True, options=solv_options)
logging.info('Done')

# Clone the suboptimal solution
instance_subopt =  m_run.instance.clone()

# Solve the suboptimal solution to optimality
opt.solve(instance_subopt, tee=True)

#%% testing pickling instances

import cloudpickle

with open('test.pkl', mode='wb') as file:
   cloudpickle.dump(m_run.instance, file)

with open('test.pkl', mode='rb') as file:
   instance_pickle = cloudpickle.load(file)


#%% testing fixing instance variables

m_run.instance = m_run.m.create_instance(p)

var_list = ['xgamma', 'xw', 'xq', 'xa']

for var in var_list:
    
    for (key, val) in getattr(instance_subopt, var).get_values().items():
        
        var_instance = getattr(m_run.instance, 'xw')
        var_instannce[key] = val
        m_run.instance.xw[key] = val
    
m_run.instance.xgamma.pprint()
    
# m_run.instance.obj = instance_subopt.obj()

opt.solve(m_run.instance, tee=True)

#%% plotting

e, e_flat = create_optimal_edges(m_run)

plot_interactive_graph(m_run.data['V'], e=e_flat, obj=m_run.instance.obj(), instance_name=m_run.instance_name)

var_list = ['xgamma', 'xw', 'xq', 'xa']
x = merge_variable_results(m_run, var_list)
print(x[x['state']>0].sort_values(['route', 'xw']))

# m_run.instance.xw.pprint()


# import sys
# f = open('instance_not_opt.txt', 'w')
# sys.stdout = f
# instance_not_opt.display()
# f.close()

