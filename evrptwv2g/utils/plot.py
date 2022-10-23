import logging
import pathlib
import folium
import folium.plugins
import matplotlib.cm
import matplotlib.colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import evrptwv2g.config.LOCAL_CONFIG as LOCAL_CONFIG
import evrptwv2g.config.GLOBAL_CONFIG as GLOBAL_CONFIG

from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.basemap import Basemap
from plotly.subplots import make_subplots
from evrptwv2g.utils.utilities import create_plotting_edges, results_to_dfs, generate_stats, get_route

log = logging.getLogger('root')


def plot_evrptwv2g(m: 'obj', **kwargs):
    # MODEL RESULTS AND PARAMETERS
    # Merge model results for plots
    x, xp, traces, routes = results_to_dfs(m)

    # Pull time-related parameters
    t_S = m.instance.t_S.value  # timestep size
    t_T = m.instance.t_T.value  # time horizon
    time = np.arange(0, t_T, t_S)   # time indices
    t_major = (t_T / 10 - (t_T / 10) % t_S)  # major ticks for time-based decisions grid

    tA = m.instance.tA.extract_values()  # start time window
    tB = m.instance.tB.extract_values()  # end time window

    # Vehicle and route parameters
    nt = len(traces)  # number of vehicles

    # Graph parameters
    stations = np.sort(xp['node'].unique())  # stations
    ns = len(stations)

    unique_stations = np.unique([m.s_2s[s] for s in stations])
    nus = len(unique_stations)  # number of unique stations

    # Optimality gap
    problem_info = m.results['Problem'][0]
    if problem_info['Upper bound'] == 0:
        gap = float("nan")
    else:
        gap = np.round(np.float64(problem_info['Upper bound'] - problem_info['Lower bound']) /
                       np.float64(problem_info['Upper bound']) * 100., 2)

    # Load Profiles
    if hasattr(m.instance, 'G'):
        G = pd.Series(m.instance.G.extract_values())
        G = G.reset_index().pivot(index='level_1', columns='level_0', values=0)

    # Energy Price
    ce = pd.Series(m.instance.ce.extract_values())
    ce = ce.reset_index().pivot(index='level_1', columns='level_0', values=0)

    # PLOTTING PARAMETERS
    # General
    if hasattr(m.instance, 'G'):
        n_plots = 7
    else:
        n_plots = 6
    style_label = u'seaborn-white'
    vehicle_cmap = 'Dark2'
    station_cmap = 'viridis'

    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fweight = 2  # title fontweights
    fdict = {'fontsize': MEDIUM_SIZE, 'fontweight': fweight}  # subplot titles

    a = 0.5 # general opacity for consistency

    tw_height = 1 # height of time window bars

    # Get colors for each vehicle
    c = plt.get_cmap(vehicle_cmap)
    colors = [c(tc) for tc in np.linspace(0, 1, len(traces))]

    # Get colors for each station
    c2 = plt.get_cmap(station_cmap)
    colors2 = [c2(tc) for tc in np.linspace(0, 1, nus)]

    us_color = {s: colors2[i] for i, s in enumerate(unique_stations)}   # map from station to color
    s_color = {s_: us_color[m.s_2s[s_]] for s_ in stations} # map from duplicate station to color
    s_alpha = {s: 0 for s in unique_stations}   # opacity for duplicate stations
    for s_, c in s_color.items():   # generating tuples for duplicate station shading
        s = m.s_2s[s_]
        s_alpha[s] += 1 / m.data['S']['instances'][s]
        c = list(c)
        c[-1] = s_alpha[s]
        s_color[s_] = tuple(c)

    # PLOTTING
    # Create the plot canvas
    plt.style.use(style_label)
    fig = plt.figure(figsize=(10, 20), num=style_label)

    # Create the subplot grid with ratios dependent on number of embedded plots in each subplot
    height_ratios = [np.ceil(n_plots/2), 1, nt, nt, 1, 1, ns]
    gs = GridSpec(n_plots, 1, width_ratios=[1], height_ratios=height_ratios[:n_plots])
    axs_graph = fig.add_subplot(gs[0, 0])
    axs_q = fig.add_subplot(gs[2, 0])
    axs_soe = fig.add_subplot(gs[3, 0], sharex=axs_q)
    axs_p = fig.add_subplot(gs[4, 0], sharex=axs_q)
    axs_ce = fig.add_subplot(gs[5, 0], sharex=axs_q)
    if hasattr(m.instance, 'G'):
        axs_g = fig.add_subplot(gs[6, 0], sharex=axs_q)
    axs_tw = fig.add_subplot(gs[1, 0], sharex=axs_q)

    # Generate objective results for plot title
    obj_breakdown = generate_stats(m)

    # Generate plot title
    title = ''
    title += '{}: {}\n'.format(m.instance_name, m.problem_type)
    title += 'Objective: {:.2f}, Gap: {:.2f}%, {:.1f}s\n'.format(
        m.instance.obj.expr(),
        gap,
        m.results['Solver'][0]['Time'] if hasattr(m.results['Solver'][0], 'Time') else m.results['Solver'][0]['Wallclock time']  # SOLVER_TYPE = 'gurobi' or 'gurobi_direct'
    )
    title += 'Dist.: {:.2f}, CapEx: {:.2f}, OpEx: {:.2f}, Delivery: {:.2f}, EA: {:.2f}, DCM: {:.2f}, Cycle: {:.2f}'.format(
        obj_breakdown['total_distance'],
        obj_breakdown['C_fleet_capital_cost'],
        obj_breakdown['O_delivery_operating_cost'],
        obj_breakdown['R_delivery_revenue'],
        obj_breakdown['R_energy_arbitrage_revenue'],
        obj_breakdown['R_peak_shaving_revenue'],
        obj_breakdown['cycle_cost']
    )
    fig.suptitle(title, x=0.5, y=1.03, horizontalalignment='center', verticalalignment='top', size=SMALL_SIZE, weight=fweight)

    # Plot graph
    # add basemap
    if kwargs.get('add_basemap', False):
        zoom_frac = [height_ratios[0]*0.01, 0.01]  # x, y
        min_lon, min_lat = m.data['V_'][['d_x', 'd_y']].min()
        max_lon, max_lat = m.data['V_'][['d_x', 'd_y']].max()
        llcrnrlon = min_lon * (1 - np.sign(min_lon) * zoom_frac[0])
        llcrnrlat = min_lat * (1 - np.sign(min_lat) * zoom_frac[1])
        urcrnrlon = max_lon * (1 + np.sign(max_lon) * zoom_frac[0])
        urcrnrlat = max_lat * (1 + np.sign(max_lat) * zoom_frac[1])
        basemap = Basemap(llcrnrlon=llcrnrlon,
                          llcrnrlat=llcrnrlat,
                          urcrnrlon=urcrnrlon,
                          urcrnrlat=urcrnrlat,
                          projection='merc',
                          resolution='h',
                          ax=axs_graph)
        basemap.drawmapboundary(fill_color='white', linewidth=0, ax=axs_graph)
        basemap.fillcontinents(color=(0.3, 0.3, 0.3), alpha=0.2, lake_color='lightblue', ax=axs_graph, zorder=1)
        basemap.drawrivers(linewidth=0.25, color='lightblue', ax=axs_graph, zorder=2)
        basemap.drawcoastlines(linewidth=0.25, color="k", ax=axs_graph)
        basemap.drawcountries(linewidth=0.25, linestyle='solid', color='k', ax=axs_graph)
        basemap.drawstates(linewidth=0.25, linestyle='solid', color='k', ax=axs_graph)
        counties = basemap.drawcounties(linewidth=0.25, linestyle='solid', color='k', ax=axs_graph, zorder=0)  # facecolor=(0.3, 0.3, 0.3, 0.2),
        counties.set_label('_nolegend_')
        basemap.drawparallels(np.arange(np.floor(llcrnrlat), np.ceil(urcrnrlat), 1), labels=[1, 0, 1, 0], linewidth=0, ax=axs_graph)
        basemap.drawmeridians(np.arange(np.floor(llcrnrlon), np.ceil(urcrnrlon), 2), labels=[1, 0, 1, 0], linewidth=0, ax=axs_graph)
        basemap.drawmapscale(urcrnrlon * (1 - np.sign(max_lon) * zoom_frac[0]/10), llcrnrlat, max_lon, llcrnrlat, length=100, units='km', ax=axs_graph)
    else:
        basemap = lambda d_x, d_y: (d_x, d_y)

    for i, n in enumerate('SDM'):
        # Plot the nodes
        axs_graph.scatter(*basemap(m.data[n]['d_x'], m.data[n]['d_y']),
                     c=[GLOBAL_CONFIG.node_colors_rgba_tuple[n]] * len(m.data[n]), s=100)

        # Annotate the nodes
        for j, txt in enumerate(m.data[n].index):
            if (m.dist_type == 'googlemaps') or (pathlib.Path(m.dist_type).suffix == '.csv'):
                scale = .4
            else:
                scale = 2
            if n == 'D':
                offset = [-scale, -scale]
            elif n == 'S':
                offset = [-scale, 0]
            else:
                offset = [0, 0]
            if txt == 'D1':
                offset = [0, -scale]

            axs_graph.annotate(txt, basemap(m.data[n]['d_x'][j] + offset[0], m.data[n]['d_y'][j] + offset[1]))

    # Plot time windows
    i = 0
    tw_labels = {}
    # Depots
    for n in tA.keys():
        if 'D' in n:
            axs_tw.broken_barh([(tA[n], tB[n] - tA[n])], (i, tw_height), facecolors=GLOBAL_CONFIG.node_colors_rgba_tuple[n[0]])
            tw_labels[n] = i
            i += 1

    # Stations
    for n in tA.keys():
        if n in s_color.keys():
            axs_tw.broken_barh([(tA[n], tB[n] - tA[n])], (i, tw_height), facecolors=s_color[n])
            tw_labels[n] = i
            i += 1
        elif 'S' in n:
            axs_tw.broken_barh([(tA[n], tB[n] - tA[n])], (i, tw_height), facecolors=GLOBAL_CONFIG.node_colors_rgba_tuple['S'])
            tw_labels[n] = i
            i += 1


    # Customers
    for n in tA.keys():
        if ('C' in n) | ('M' in n):
            axs_tw.broken_barh([(tA[n], tB[n] - tA[n])], (i, tw_height), facecolors=GLOBAL_CONFIG.node_colors_rgba_tuple['M'])
            tw_labels[n] = i
            i += 1

    axs_tw.set_yticks(np.arange(tw_height/2, i + .5*tw_height, tw_height))
    axs_tw.set_yticklabels(tw_labels.keys(), fontsize=np.ceil(2*SMALL_SIZE/3))
    axs_tw.grid(True, which='both', axis='x', alpha=a, linestyle=':')
    axs_tw.set_title('Arrival Times', fontdict=fdict)

    # Plot the vehicle routes
    for ti, t in enumerate(traces):
        t = (t[0],) + t  # Create starting point (D0, D0)

        # Plot the route edges for each vehicle on the graph
        for ki, k in enumerate(t[:-1]):
            if kwargs.get('add_basemap', False):
                arrow_x, arrow_y = basemap(m.data['V_'].loc[t[ki], 'd_x'], m.data['V_'].loc[t[ki], 'd_y'])
                end_arrow_x, end_arrow_y = basemap(m.data['V_'].loc[t[ki + 1], 'd_x'], m.data['V_'].loc[t[ki + 1], 'd_y'])
                arrow_dx = end_arrow_x - arrow_x
                arrow_dy = end_arrow_y - arrow_y

                axs_graph.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, alpha=a, linewidth=2,
                                length_includes_head=True, head_width=1.e5 * scale / 2, head_length=1.e5 * scale / 2, overhang=0,
                                color=colors[ti])

                # arrow_x, arrow_y = (m.data['V_'].loc[t[ki], 'd_x'], m.data['V_'].loc[t[ki], 'd_y'])
                # end_arrow_x, end_arrow_y = (m.data['V_'].loc[t[ki + 1], 'd_x'], m.data['V_'].loc[t[ki + 1], 'd_y'])
                #
                # basemap.drawgreatcircle(arrow_x, arrow_y, end_arrow_x, end_arrow_y, linewidth=2, color=colors[ti])
            else:
                arrow_x = m.data['V_'].loc[t[ki], 'd_x']
                arrow_y = m.data['V_'].loc[t[ki], 'd_y']
                arrow_dx = m.data['V_'].loc[t[ki + 1], 'd_x'] - arrow_x
                arrow_dy = m.data['V_'].loc[t[ki + 1], 'd_y'] - arrow_y

                axs_graph.arrow(arrow_x, arrow_y, arrow_dx, arrow_dy, alpha=a, linewidth=2,
                           length_includes_head=True, head_width=scale/2, head_length=scale/2, overhang=0,
                           color=colors[ti])

        axs_graph.set_aspect('equal', adjustable='datalim')

        # Plot the vehicle states
        r = routes.loc[list(zip(t[:-1], t[1:]))]

        # Plot the vehicle arrivals over the time windows
        w_nodes = [tw_labels[v[1]] + tw_height/2 for v in r.index] # Map the arrival nodes to tw y-value
        axs_tw.scatter(r['xw'], w_nodes, color=colors[ti], alpha=a)

        # Plot arrival capacity
        # create an inset axe in the current axe:
        ax_q = axs_q.inset_axes([0, ti * 1 / nt, 1, .8 * 1 / nt])
        ax_q.plot(r['xw'], r['xq'], color=colors[ti], alpha=a)
        for row in r.iterrows():
            ax_q.annotate('{}: {:.1f}'.format(row[0][1], row[1]['xq']),
                         xy=(row[1]['xw'], row[1]['xq']), color=colors[ti])
        ax_q.get_shared_x_axes().join(ax_q, axs_q)
        ax_q.get_shared_y_axes().join(ax_q, axs_q)
        ax_q.set_xticklabels([])
        ax_q.xaxis.set_major_locator(MultipleLocator(t_major))
        ax_q.xaxis.set_minor_locator(MultipleLocator(t_S))
        ax_q.grid(True, which='both', axis='x', alpha=a, linestyle=':')
        axs_q.set_yticklabels([])
        axs_q.set_xlim([0, t_T])
        #     axs_q.axis('off')
        #     axs_q.get_xaxis().set_visible(True)
        #     axs_q.grid(False)
        axs_q.spines['right'].set_visible(False)
        axs_q.spines['top'].set_visible(False)
        axs_q.spines['bottom'].set_visible(False)
        axs_q.spines['left'].set_visible(False)
        axs_q.set_title('Arrival capacity', fontdict=fdict)

        # Plot arrival SOE
        # create an inset axe in the current axe:
        ax_soe = axs_soe.inset_axes([0, ti * 1 / nt, 1, .8 * 1 / nt])
        ax_soe.plot(r['xw'], r['xa'], color=colors[ti], alpha=a)
        for row in r.iterrows():
            ax_soe.annotate('{}: {:.1f}'.format(row[0][1], row[1]['xa']),
                         xy=(row[1]['xw'], row[1]['xa']), color=colors[ti])
        ax_soe.get_shared_x_axes().join(ax_soe, axs_soe)
        ax_soe.get_shared_y_axes().join(ax_soe, axs_soe)
        ax_soe.set_xticklabels([])
        ax_soe.xaxis.set_major_locator(MultipleLocator(t_major))
        ax_soe.xaxis.set_minor_locator(MultipleLocator(t_S))
        ax_soe.grid(True, which='both', axis='x', alpha=a, linestyle=':')
        axs_soe.set_yticklabels([])
        axs_soe.set_xlim([0, t_T])
        axs_soe.spines['right'].set_visible(False)
        axs_soe.spines['top'].set_visible(False)
        axs_soe.spines['bottom'].set_visible(False)
        axs_soe.spines['left'].set_visible(False)
        axs_soe.set_title('Arrival SOE', fontdict=fdict)

    # TODO: If no stations are visited, need to exclude this...
    # Plot power
    s = stations[0]
    bott = pd.DataFrame(index=time)
    bott['xp'] = 0
    ind = xp[xp['node'] == s]['t']
    axs_p.bar(ind, xp[xp['node'] == s]['xp'],
             label=s, width=t_S, align='edge', color=s_color[s])
    for s2 in stations[1:]:
        ind = xp[xp['node'] == s]['t']
        bott['xp'].loc[ind] += xp[xp['node'] == s]['xp'].values
        axs_p.bar(xp[xp['node'] == s2]['t'], xp[xp['node'] == s2]['xp'],
                 bottom=bott.loc[xp[xp['node'] == s2]['t']].values.reshape(-1),
                 label=s2, width=t_S, align='edge', color=s_color[s2])
        s = s2
    ind = xp[xp['node'] == s]['t']
    bott['xp'].loc[ind] += xp[xp['node'] == s]['xp'].values

    # axs_p.hlines(bott['xp'], bott.index, bott.index+t_S)
    bott.loc[t_T, 'xp'] = bott.loc[t_T - t_S, 'xp']
    axs_p.step(bott.index, bott['xp'], where='post', alpha=a, color='black')
    axs_p.set_title('Power', fontdict=fdict)

    axs_p.xaxis.set_major_locator(MultipleLocator(t_major))
    axs_p.xaxis.set_minor_locator(MultipleLocator(t_S))
    axs_p.grid(True, which='both', axis='x', alpha=a, linestyle=':')
    # axs_p.legend(loc="upper left", ncol=round(ns/2), bbox_to_anchor=(0, -.25));

    # Plot energy price
    ce.loc[t_T] = ce.loc[t_T - t_S]
    for s in stations:
        axs_ce.step(ce[m.s_2s[s]].index, ce[m.s_2s[s]], where='post', label='_nolegend_', alpha=a,
                  color=s_color[s])
    axs_ce.set_title('Energy Price', fontdict={'fontsize': MEDIUM_SIZE, 'fontweight': fweight})

    axs_ce.xaxis.set_major_locator(MultipleLocator(t_major))
    axs_ce.xaxis.set_minor_locator(MultipleLocator(t_S))
    axs_ce.grid(True, which='both', axis='x', alpha=a, linestyle=':')

    if hasattr(m.instance, 'G'):
        # Plot load profiles
        G.loc[t_T] = G.loc[t_T - t_S]

        # calculate total power for each station
        temp = xp.copy()
        temp['s'] = temp['node'].apply(lambda x: m.s_2s[x])
        total_power = temp.groupby(['s', 't']).sum()

        for si, s in enumerate(unique_stations):
            # create an inset axis in the current axes:
            ax_g = axs_g.inset_axes([0, si * 1 / nus, 1, .8 * 1 / nus])
            ax_g.step(G[s].index, G[s], where='post', label='_nolegend_', alpha=a, color=us_color[s])

            net_load = G[s].copy()
            net_load[total_power.loc[s].index] += total_power.loc[s]['xp']
            ax_g.fill_between(net_load.index, net_load, step="post", alpha=a, color='black')

            ax_g.get_shared_x_axes().join(ax_g, axs_g)
            ax_g.get_shared_y_axes().join(ax_g, axs_g)
            ax_g.set_xticklabels([])
            axs_g.set_yticklabels([])
            axs_g.set_xlim([0, t_T])
            ax_g.title.set_text('{}: ${}/kW'.format(s, m.instance.cg.extract_values()[s]))

            ax_g.xaxis.set_major_locator(MultipleLocator(t_major))
            ax_g.xaxis.set_minor_locator(MultipleLocator(t_S))
            ax_g.grid(True, which='both', axis='x', alpha=a, linestyle=':')

        axs_g.spines['right'].set_visible(False)
        axs_g.spines['top'].set_visible(False)
        axs_g.spines['bottom'].set_visible(False)
        axs_g.spines['left'].set_visible(False)
        axs_g.set_title('Load Profile', fontdict=fdict)

    fig.tight_layout()

    fig.legend(
        loc="lower right",
        bbox_to_anchor=(1.08, 0.01),
        bbox_transform=plt.gcf().transFigure
    );  # ncol=round(nus), , bbox_to_anchor=(1.05, .4)

    if kwargs.get('save', False):
        if kwargs.get('save_folder', False):  # skips if None
            save_folder = kwargs.get('save_folder')
        else:
            save_folder = LOCAL_CONFIG.DIR_OUTPUT

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path = f'{save_folder}/{timestamp}_{m.instance_name}_{m.problem_type}.png'
        # fig.show()
        fig.savefig(save_path, bbox_inches='tight')
        log.info(f'Plot saved: {save_path}')

        plt.close(fig)
        fig.clear()

    return x, xp, traces, routes


def plot_interactive_graph(v: 'pd.DataFrame', **kwargs) -> 'fig':
    """
    Plots an interactive undirected network graph with labeled and colored nodes and edges.

    :param v: dataframe of nodes and their coordinates
    :type v: pd.DataFrame
    :param d: square distance matrix
    :type d: pd.DataFrame
    :return: plotly figure
    """
    if 'd' in kwargs.keys():
        edges = create_plotting_edges(v, kwargs['d'])
    elif 'e' in kwargs.keys():
        edges = kwargs['e']
    else:
        log.error('Must provide distance matrix "d" or optimal edge matrix "e".')

    title = ''
    if 'instance_name' in kwargs.keys():
        title += '{}<br>'.format(kwargs['instance_name'])

    if 'obj' in kwargs.keys():
        title += 'Objective: {}'.format(round(kwargs['obj'], 2))

    # Initialize plotly figure
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.02,
                        specs=[[{"secondary_y": False}]] * 1
                        )

    # Add trace for edges
    if 'd' in kwargs.keys():
        fig.add_trace(
            go.Scatter(
                x=edges[:, 0],
                y=edges[:, 1],
                name='Edges',
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')
        )
    elif 'e' in kwargs.keys():
        if 'vehicle' in edges.columns:
            c = plt.get_cmap('Blues')
            vehicles = edges['vehicle'].unique()
            for vehicle in vehicles:
                fig['layout']['annotations'] += tuple(
                    [dict(ax=e['from_d_x'], ay=e['from_d_y'], x=e['to_d_x'], y=e['to_d_y'],
                          xref='x', yref='y', axref='x', ayref='y',
                          text='', showarrow=True, arrowcolor='rgba{}'.format(c(vehicle/len(vehicles))), arrowhead=2, arrowsize=1,
                          arrowwidth=2)
                     for i, e in edges[edges['vehicle'] == vehicle].iterrows()])
        else:
            fig['layout']['annotations'] += tuple(
                [dict(ax=e['from_d_x'], ay=e['from_d_y'], x=e['to_d_x'], y=e['to_d_y'],
                      xref='x', yref='y', axref='x', ayref='y',
                      text='', showarrow=True, arrowcolor='#888', arrowhead=2, arrowsize=1, arrowwidth=2)
                 for i, e in edges.iterrows()])

    # Add trace for nodes
    if 'e' in kwargs.keys():
        # TODO: Modify plot by type of VRP problem
        try:
            customdata = [[v.loc[i, ['node_description']],
                           v.loc[i, ['q']],
                           edges[edges['to']==i]['vehicle'],
                           edges[edges['to']==i]['xw'],
                           edges[edges['to']==i]['xq']] for i in v.index]
            hovertemplate = '<b>%{text}</b>: %{customdata[0]}' + \
                            '<br>Vehicle: %{customdata[2]}' + \
                            '<br>Arrival Time: %{customdata[3]}' + \
                            '<br>Payload: %{customdata[4]}' + \
                            '<br>Demand: %{customdata[1]}' + \
                            '<br>X: %{x}' + \
                            '<br>Y: %{y}'
        except:
            customdata = [[v.loc[i, ['node_description']]] for i in v.index]
            hovertemplate = '<b>%{text}</b>: %{customdata[0]}' + \
                            '<br>X: %{x}' + \
                            '<br>Y: %{y}'

    else:
        customdata = v['node_description']
        hovertemplate = '<b>%{text}</b>: %{customdata}' + \
                        '<br>X: %{x}' + \
                        '<br>Y: %{y}'
    fig.add_trace(
        go.Scatter(
            x=v['d_x'],
            y=v['d_y'],
            text=v.index, #['{}: {}'.format(n, v['node_description'][n]) for n in v.index],
            name='Nodes',
            customdata=customdata,
            hovertemplate=hovertemplate,
            hoverinfo='none',
            mode='markers+text',
            marker=dict(size=25, color=[GLOBAL_CONFIG.node_colors_rgba[n] for n in v['node_type']]),
            showlegend=True
        )
    )

    # Update template and subplot heights
    fig.update_layout(
        template='simple_white',
        height=600,
        title=title,
        xaxis=dict(
            title='x',
            range=[0, v['d_x'].max() * 1.1],
            constrain='domain',
            mirror=True,
            ticks='outside',
            showline=True
        ),
        yaxis=dict(
            title='y',
            range=[0, v['d_y'].max() * 1.1],
            mirror=True,
            ticks='outside',
            showline=True,
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=True
    )

    fig.show()


def plot_folium_map(m, traces):
    """
    requires running osrm-backend for routing. See: https://github.com/Project-OSRM/osrm-backend
    """
    min_lon, min_lat = m.data['V_'][['d_x', 'd_y']].min()
    max_lon, max_lat = m.data['V_'][['d_x', 'd_y']].max()
    fmap = folium.Map(
        location=[(min_lat + max_lat) / 2,  # lat
                  (min_lon + max_lon) / 2],  # lon
        zoom_start=7)
    folium.TileLayer('cartodbpositron').add_to(fmap)  # cartodbpositron, stamentoner

    cmap = matplotlib.cm.get_cmap('Dark2')
    i_route_colors = list(np.linspace(0., 1., len(traces)))
    route_colors = [matplotlib.colors.to_hex(cmap(i), keep_alpha=True) for i in i_route_colors]

    osrm_routes = {}
    for t in traces:
        osrm_route = {}
        route_color = route_colors.pop()
        for i in range(1, len(t)):
            origin_lon, origin_lat = m.data['V_'].loc[t[i - 1]][['d_x', 'd_y']]
            dest_lon, dest_lat = m.data['V_'].loc[t[i]][['d_x', 'd_y']]
            osrm_route[(t[i - 1], t[i])] = get_route(origin_lon, origin_lat, dest_lon, dest_lat)
        osrm_routes[t] = osrm_route

        # Print routes
        opacity = list(np.linspace(0.6, 0.1, len(osrm_route)))
        for od, route in osrm_route.items():
            folium.plugins.AntPath(
                route['route'],
                weight=8,
                color=route_color,
                opacity=opacity.pop()
            ).add_to(fmap)

    node_size = {
        'D': 6,
        'S': 4,
        'M': 2
    }
    for name, row in m.data['V'].iterrows():
        node_loc = [row['d_y'], row['d_x']]
        node_type = row['node_type'].upper()
        node_color = matplotlib.colors.to_hex(GLOBAL_CONFIG.node_colors_rgba_tuple[node_type])
        folium.CircleMarker(
            location=node_loc,
            radius=node_size[node_type],
            popup=row['node_description'],
            color=node_color,
            fill=True,
            fill_color=node_color,
        ).add_to(fmap)

        scale = .4
        if node_type == 'D':
            offset = [scale, scale]
        elif node_type == 'S':
            offset = [0, -scale]
        else:
            offset = [0, 0]
        if row.name == 'D1':
            offset = [0, scale]

        folium.map.Marker(
            [sum(tup) for tup in zip(node_loc, offset)],
            icon=folium.DivIcon(
                icon_size=(250, 36),
                icon_anchor=(0, 0),
                html=f'<div style="font-size: 16pt; color: {node_color}">{name}</div>',
            )
        ).add_to(fmap)

    fmap.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
    return fmap
