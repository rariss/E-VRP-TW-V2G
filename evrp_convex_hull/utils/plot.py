# from config.GLOBAL_CONFIG import node_colors_rgba
# from utils.utilities import create_plotting_edges
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import plot # for Spyder
import logging

from utils.utilities import create_flat_distance_matrix, create_coordinate_dict  # TODO: NEW
import numpy as np  # TODO: NEW

node_colors_rgba = {'m': 'rgba(240, 169, 171, 0.8)',  # alpha=0.4
                    'd': 'rgba(191, 243, 180, 0.8)',
                    's': 'rgba(167, 215, 241, 0.8)'
                    }


def create_plotting_edges(v: 'pd.DataFrame', d: 'pd.DataFrame') -> 'np.array':
    """
    Creates a 2D np.array of edges betwee provided from and to nodes, with (None, None) indices
    creating the disconnection between edges in the array.

    :param v: dataframe of nodes and their coordinates
    :type v: pd.DataFrame
    :param d: square distance matrix
    :type d: pd.DataFrame
    :return: 2*Vx2 np.array of edge coordinates
    :rtype: np.array
    """
    d_flat = create_flat_distance_matrix(d)
    coordinates = create_coordinate_dict(v)

    edges = np.concatenate(
        [np.vstack((coordinates[a], coordinates[b], [None, None])) for a, b in d_flat[['from', 'to']].values])
    return edges


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
        # edges = create_plotting_edges(v, kwargs['e'])
        edges = kwargs['e']
    else:
        logging.error('Must provide distance matrix "d" or optimal edge matrix "e".')

    if 'obj' in kwargs.keys():
        title = '{} - Objective: {}'.format(kwargs['instance_name'], round(kwargs['obj'], 2))
    else:
        title = ''

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
        fig['layout']['annotations'] += tuple(
            [dict(x=e['from_d_x'], y=e['from_d_y'], ax=e['to_d_x'], ay=e['to_d_y'],
                  xref='x', yref='y', axref='x', ayref='y',
                  text='', showarrow=True, arrowcolor='#888', arrowhead=5, arrowsize=1, arrowwidth=2)
             for i, e in edges.iterrows()])

    # Add trace for nodes
    fig.add_trace(
        go.Scatter(
            x=v['d_x'],
            y=v['d_y'],
            text=v.index,  # ['{}: {}'.format(n, v['node_description'][n]) for n in v.index],
            name='Nodes',
            customdata=v['node_description'],
            hovertemplate=
            '<b>%{text}</b>: %{customdata}' +
            '<br>X: %{x}' +
            '<br>Y: %{y}',
            hoverinfo='none',
            mode='markers+text',
            marker=dict(size=25, color=[node_colors_rgba[n] for n in v['node_type']]),
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
    plot(fig) # for Spyder
