import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as dist
import itertools as itt
import matplotlib.pyplot as plt
from pyomo.environ import value

def instance_reader(filename):

    instance_data = pd.read_csv(filename, delimiter= '\s+').dropna()

    depot_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'd']
    station_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'f']
    customer_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'c']

    nodes = instance_data['StringID']
    demand = instance_data['demand']
    node_start_time = instance_data['ReadyTime']
    node_end_time = instance_data['DueDate']
    service_time = instance_data['ServiceTime']

    depot_nodes = nodes[depot_nodes_idx]
    station_nodes = nodes[station_nodes_idx]
    customer_nodes = nodes[customer_nodes_idx]

    nodes_coords = list(zip(instance_data['x'].astype(float), instance_data['y'].astype(float)))

    nodes_dist = dist(nodes_coords,nodes_coords) # distance between nodes

    # nodes_pair = [i for i  in itt.permutations(nodes,2)]

    nodes_pairs = []
    nodes_pairs_idx = []
    nodes_pairs_dist = []
    arcs = []

    for i, j in itt.product(range(len(nodes)), range(len(nodes))):
        if i != j and (j,i) not in nodes_pairs_idx or j == 0 :
            arcs.append((nodes[i], nodes[j]))

        nodes_pairs.append((nodes[i], nodes[j]))
        nodes_pairs_idx.append((i,j))
        nodes_pairs_dist.append(nodes_dist[(i,j)])

    #TODO: return table
    return nodes, nodes_pairs, nodes_pairs_dist, arcs, nodes_coords, demand, node_start_time, node_end_time, service_time


def solution_saver(inst, inst_name):

    solution_file = open('solutions/' + inst_name[:-4] + '_sol.txt', 'w')
    solution_file.write('# solution for ' + inst_name[:-4] +
                        '\n' + format(value(inst.obj)))

    solution = getattr(inst, 'x').extract_values()
    active_arcs = [n for n in list(solution.items()) if n[1] == 1.0]
    active_arcs = list(list(zip(*active_arcs))[0])

    start_nodes, end_nodes = list(list(zip(*active_arcs)))

    routes = []
    route = []

    for n,m in enumerate(start_nodes):

        curent_node = m

        if curent_node == 'D0':
            route.append(curent_node)
            next_node = end_nodes[n]
            solution_file.write('\n' + curent_node + ', ')

            while next_node in start_nodes:

                if next_node == 'D0':
                    route.append(next_node)
                    solution_file.write(next_node)
                    break


                route.append(next_node)
                idx = start_nodes.index(next_node)

                solution_file.write(next_node + ', ')

                next_node = end_nodes[idx]

    routes.append(route)

    solution_file.close()

    print('\n****************SOLUTION SUMMARY****************')
    print('\n Objective function value: ', value(inst.obj))
    print('\n Routes: \n', routes)
    print('\n Active arcs: \n', active_arcs)

    return routes


def solution_plotter(nodes, nodes_coords, routes):

    xs, ys = list(zip(*nodes_coords))

    # plot nodes
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)

    # label nodes
    plotted = []

    for i, node in enumerate(nodes):
        x = xs[i]
        y = ys[i]
        if ys.count(y) > 1 and xs.count(x) > 1 and y not in plotted:
            plotted.append(y)
            y = y - 6
            plotted.append(y)
        ax.annotate(node, (x-0.5, y+2))

    # plot routes
    for route in routes:
        for i, node in enumerate(route):
            coords = nodes_coords[(nodes==node).argmax()]
            if i < len(route)-1:
                coords_next = nodes_coords[(nodes==route[i+1]).argmax()]
            x = coords[0]
            y = coords[1]
            dx = coords_next[0] - x
            dy = coords_next[1] - y
            ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.5, head_length=1, color = 'Black')


import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as dist
import itertools as itt
import matplotlib.pyplot as plt
from pyomo.environ import value

def instance_reader(filename):

    instance_data = pd.read_csv(filename, delimiter= '\s+').dropna()

    depot_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'd']
    station_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'f']
    customer_nodes_idx = [n for (n,m) in enumerate(instance_data['Type']) if m is 'c']

    nodes = instance_data['StringID']

    depot_nodes = nodes[depot_nodes_idx]
    station_nodes = nodes[station_nodes_idx]
    customer_nodes = nodes[customer_nodes_idx]

    nodes_coords = list(zip(instance_data['x'].astype(float), instance_data['y'].astype(float)))

    nodes_dist = dist(nodes_coords,nodes_coords) # distance between nodes

    # nodes_pair = [i for i  in itt.permutations(nodes,2)]

    nodes_pairs = []
    nodes_pairs_idx = []
    nodes_pairs_dist = []
    arcs = []

    for i, j in itt.product(range(len(nodes)), range(len(nodes))):
        if i != j and (j,i) not in nodes_pairs_idx or j == 0 :
            arcs.append((nodes[i], nodes[j]))

        nodes_pairs.append((nodes[i], nodes[j]))
        nodes_pairs_idx.append((i,j))
        nodes_pairs_dist.append(nodes_dist[(i,j)])

    return nodes, nodes_pairs, nodes_pairs_dist, arcs, nodes_coords


def solution_saver(inst, inst_name):

    solution_file = open('solutions/' + inst_name[:-4] + '_sol.txt', 'w')
    solution_file.write('# solution for ' + inst_name[:-4] +
                        '\n' + format(value(inst.obj)))

    solution = getattr(inst, 'x').extract_values()
    active_arcs = [n for n in list(solution.items()) if n[1] == 1.0]
    active_arcs = list(list(zip(*active_arcs))[0])

    start_nodes, end_nodes = list(list(zip(*active_arcs)))

    routes = []
    route = []

    for n,m in enumerate(start_nodes):

        curent_node = m

        if curent_node == 'D0':
            route.append(curent_node)
            next_node = end_nodes[n]
            solution_file.write('\n' + curent_node + ', ')

            while next_node in start_nodes:

                if next_node == 'D0':
                    route.append(next_node)
                    solution_file.write(next_node)
                    break


                route.append(next_node)
                idx = start_nodes.index(next_node)

                solution_file.write(next_node + ', ')

                next_node = end_nodes[idx]

    routes.append(route)

    solution_file.close()

    print('\n****************SOLUTION SUMMARY****************')
    print('\n Objective function value: ', value(inst.obj))
    print('\n Routes: \n', routes)
    print(active_arcs)

    return routes


def solution_plotter(nodes, nodes_coords, routes):

    xs, ys = list(zip(*nodes_coords))

    # plot nodes
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)

    # label nodes
    plotted = []

    for i, node in enumerate(nodes):
        x = xs[i]
        y = ys[i]
        if ys.count(y) > 1 and xs.count(x) > 1 and y not in plotted:
            plotted.append(y)
            y = y - 6
            plotted.append(y)
        ax.annotate(node, (x-0.5, y+2))

    # plot routes
    for route in routes:
        for i, node in enumerate(route):
            coords = nodes_coords[(nodes==node).argmax()]
            if i < len(route)-1:
                coords_next = nodes_coords[(nodes==route[i+1]).argmax()]
            x = coords[0]
            y = coords[1]
            dx = coords_next[0] - x
            dy = coords_next[1] - y
            ax.arrow(x, y, dx, dy, length_includes_head=True, head_width=0.5, head_length=1, color = 'Black')
