from evrp.config.GLOBAL_CONFIG import node_colors_rgba
from evrp.utils.utilities import create_plotting_edges
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_interactive_graph(v: 'pd.DataFrame', d: 'pd.DataFrame') -> 'fig':
    """
    Plots an interactive undirected network graph with labeled and colored nodes and edges.

    :param v: dataframe of nodes and their coordinates
    :type v: pd.DataFrame
    :param d: square distance matrix
    :type d: pd.DataFrame
    :return: plotly figure
    """
    edges = create_plotting_edges(v, d)

    # Initialize plotly figure
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.02,
                        specs=[[{"secondary_y": False}]] * 1
                        )

    # Add trace for edges
    fig.add_trace(
        go.Scatter(
            x=edges[:, 0],
            y=edges[:, 1],
            name='Edges',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
    )

    # Add trace for nodes
    fig.add_trace(
        go.Scatter(
            x=v['d_x'],
            y=v['d_y'],
            text=v.index, #['{}: {}'.format(n, v['node_description'][n]) for n in v.index],
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
