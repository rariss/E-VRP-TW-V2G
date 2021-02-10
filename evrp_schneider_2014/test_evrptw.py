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
opt = SolverFactory('baron')

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
# solv_options = {'timelimit': 5} # fix a time limit to create suboptimal solution
solv_options = {'MaxTime': 60*60*2} # for baron while my gurobi license has expired
    
# Solve instance
logging.info('Solving instance...')
# results = opt.solve(m_run.instance, tee=True)
results = opt.solve(m_run.instance, tee=True, options=solv_options)
logging.info('Done')

# Clone the suboptimal solution
# instance_subopt =  m_run.instance.clone()

# Solve the suboptimal solution to optimality
# opt.solve(instance_subopt, tee=True)

#%% testing pickling instances

import cloudpickle

with open('test.pkl', mode='wb') as file:
   cloudpickle.dump(m_run.instance, file)

with open('test.pkl', mode='rb') as file:
   instance_pickle = cloudpickle.load(file)


#%% testing updating instance variables

def update_instance(var_list, new_concrete_instance, instance_subopt):
    
    for var in var_list:
        var_instance = getattr(new_concrete_instance, var)
        
        for (key, val) in getattr(instance_subopt, var).get_values().items():
            var_instance[key] = val
        
        setattr(new_concrete_instance, var, var_instance)
        
    return new_concrete_instance

var_list = ['xgamma', 'xw', 'xq', 'xa']
m_run.instance = m_run.m.create_instance(p)
m_run.instance = update_instance(var_list, m_run.instance, instance_subopt)

# m_run.instance.xw.pprint()
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

#%% creating json output

import json
import numpy as np

# JSON can't handle np.arrays so convert to list
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, tuple):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super(NpEncoder, self).default(obj)

data_out = {}
for v in m_run.instance.component_objects(Var):
    data_out[v.name] = {}
    for index in v:
        try:        
            data_out[v.name][str(index)] = value(v[index])
        except:
            data_out[v.name][str(index)] = None

json_outputs = json.dumps(data_out, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))

with open('output.json', 'w') as outfile:
    json.dump(data_out, outfile, cls=NpEncoder, sort_keys=True, indent=4, separators=(',', ': '))

# print(json_outputs)

#%% loading json output and solving

with  open('output.json', 'r') as inputfile:
    instance_subopt2 = json.load(inputfile)

def update_instance_json(var_list, new_concrete_instance, instance_subopt):
    
    for var in var_list:
        
        var_instance = getattr(new_concrete_instance, var)
        
        for (key, val) in instance_subopt2[var].items():
            
            if var is 'xgamma':
                var_instance[eval(key)] = val 
            else:
                var_instance[key] = val
        
        setattr(new_concrete_instance, var, var_instance)
        
    return new_concrete_instance


var_list = ['xgamma', 'xw', 'xq', 'xa']
m_run.instance = m_run.m.create_instance(p)
m_run.instance = update_instance_json(var_list, m_run.instance, instance_subopt2)


#%% debugging json list error, not needed by kept for reference

from json import dumps, loads, JSONEncoder, JSONDecoder
import pickle

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return JSONEncoder.default(self, obj)
        return {'_python_object': pickle.dumps(obj)}
    
def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(str(dct['_python_object']))
    return dct

data = [1,2,3, set(['knights', 'who', 'say', 'ni']), {'key':'value'}, Decimal('3.14')]

j = dumps(data, cls=PythonObjectEncoder)

loads(j, object_hook=as_python_object)
[1, 2, 3, set(['knights', 'say', 'who', 'ni']), {u'key': u'value'}, Decimal('3.14')]