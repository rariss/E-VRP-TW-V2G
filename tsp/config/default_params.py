from utils.utilities import *

inst_5 = 'c101C5-D1.txt' # 5 customers instance file
inst_15 = 'c103C15.txt' # 15 customer instance file
inst_100 = 'rc103_21.txt' # 100 customer instance file

instance_name = inst_5

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

p = {None: {
    'Mq': {None: 1000}, # Large value for big M payload constraints
    'Mt': {None: 100000}, # Large value for big M service time constraints
    'V01': {None: set(nodes)},  # All nodes extended np.arange(len(nodes)))
    'V0': {None: set(nodes)-{'D1'}},  # All nodes extended np.arange(len(nodes)))
    'V1': {None: set(nodes)-{'D0'}},  # All nodes extended np.arange(len(nodes)))
    'V': {None: set(nodes)-{'D0','D1'}},  # All nodes extended np.arange(len(nodes)))
    'd': dict(zip(nodes_pairs, nodes_pairs_dist)), # Distance of edge (i;j) between nodes i;j (km)
    'q': dict(zip(nodes, demand)), # Delivery demand at each customer
    'tS': dict(zip(nodes, service_time)), # Fixed service time for a vehicle at node i
    'tA': dict(zip(nodes, node_start_time)), # Delivery demand at each customer
    'tB': dict(zip(nodes, node_end_time)), # Delivery demand at each customer

}}
