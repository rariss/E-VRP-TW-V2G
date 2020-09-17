from utils.utilities import *

inst_5 = 'c101C5.txt' # 5 customers instance file
inst_15 = 'c103C15.txt' # 15 customer instance file
inst_100 = 'rc103_21.txt' # 100 customer instance file

instance_name = inst_5

instance_data = instance_reader('data/' + instance_name)
nodes = instance_data[0]
nodes_pairs = instance_data[1]
nodes_pairs_dist = instance_data[2]
arcs = instance_data[3]
nodes_coords = instance_data[4]

p = {None: {    
    'V_': {None: set(nodes)},  # All nodes extended np.arange(len(nodes)))
    'd': dict(zip(nodes_pairs, nodes_pairs_dist)), # Distance of edge (i;j) between nodes i;j (km)
    # 'Arcs': dict(zip(arcs, range(len(arcs))))
}}

