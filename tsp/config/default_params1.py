from utils.utilities import *

# inst_5 = 'c101C5-D1-Cap.txt' # 5 customers instance file 'c101C5-10.txt'
inst_5 = 'c101C5-10-S0.txt'
inst_15 = 'c103C15-D1.txt' # 15 customer instance file
inst_100 = 'rc103_21.txt' # 100 customer instance file

instance_name = inst_15

# TODO: returns should be in a table
instance_data = instance_reader('data/' + instance_name)
nodes = instance_data[0]
nodes_pairs = instance_data[1]
nodes_pairs_dist = instance_data[2]
arcs = instance_data[1]
nodes_coords = instance_data[4]
demand = instance_data[5].astype(float)
node_start_time = instance_data[6].astype(float)
node_end_time = instance_data[7].astype(float)
service_time = instance_data[8].astype(float)
station_nodes = instance_data[9]
customer_nodes = instance_data[10]

p = {None: {    
    'Mq': {None: 200000}, # Large value for big M payload constraints
    'Mt': {None: 1236}, # Large value for big M service time constraints 
    'V01_': {None: set(nodes)},  # All nodes extended np.arange(len(nodes)))
    'V0_': {None: set(nodes)-{'D1'}},  # All nodes extended np.arange(len(nodes)))
    'V1_': {None: set(nodes)-{'D0'}},  # All nodes extended np.arange(len(nodes)))
    'V_': {None: set(nodes)-{'D0','D1'}},  # All nodes extended np.arange(len(nodes)))
    'V': {None: set(customer_nodes)},  # All nodes extended np.arange(len(nodes)))
    'F': {None: set(station_nodes)},
    'd': dict(zip(nodes_pairs, nodes_pairs_dist)), # Distance of edge (i;j) between nodes i;j (km)
    'q': dict(zip(nodes, demand)), # Delivery demand at each customer
    'tS': dict(zip(nodes, service_time)), # Fixed service time for a vehicle at node i
    'tA': dict(zip(nodes, node_start_time)), # Delivery demand at each customer
    'tB': dict(zip(nodes, node_end_time)), # Delivery demand at each customer
        
}}

