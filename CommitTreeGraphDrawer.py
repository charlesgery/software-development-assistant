import networkx as nx
from pyvis.network import Network
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool, TapTool, BoxSelectTool,
                          MultiLine, Plot, Range1d, ResetTool, GraphRenderer, StaticLayoutProvider,
                          EdgesAndLinkedNodes, NodesAndLinkedEdges, WheelZoomTool, PanTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, figure

import compute_layout


class CommitTreeGraphDrawer:

    def __init__(self, graph):

        self.graph = graph

    def draw_bokeh_commit_treegraph(self):
        """ Draw commit TreeGraph using networkx.
        """

        def draw_bokeh_commit_treegraph_iteration(treegraph, center=(0,0), node_radius=0.5):

            graph_renderer = GraphRenderer()

            graph_renderer.node_renderer.glyph = Circle(radius="nodes_radius", fill_color="fill_color")
            graph_renderer.node_renderer.selection_glyph = Circle(size=node_radius, fill_color=Spectral4[2])
            graph_renderer.node_renderer.hover_glyph = Circle(size=node_radius, fill_color=Spectral4[1])
            graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)
            graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
            graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

            graph_renderer.selection_policy = NodesAndLinkedEdges()
            graph_renderer.inspection_policy = EdgesAndLinkedNodes()

            index, color, nodes_radius = [], [], []
            for node in treegraph.graph.nodes:
                index.append(node)
                if treegraph.kids[node].is_file:
                    color.append(Spectral4[0])
                    nodes_radius.append(0.05)

                else:
                    color.append(Spectral4[1])
                    nodes_radius.append(node_radius)
            
            graph_renderer.node_renderer.data_source.data = dict(
                index=index,
                fill_color=color,
                nodes_radius=nodes_radius)

            start, end = [], []
            for (node1, node2) in treegraph.graph.edges:
                start.append(node1)
                end.append(node2)

            graph_renderer.edge_renderer.data_source.data = dict(
                start=start,
                end=end)

            node_size = dict(zip(treegraph.graph.nodes, nodes_radius))
            pos = compute_layout.get_fruchterman_reingold_layout(list(zip(start, end)), scale=(2,2), origin=(-1, -1), node_size=node_size)
            #pos = nx.spring_layout(treegraph.graph, scale=1, center=center, k=node_radius*5)

            graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=pos)

            plot.renderers.append(graph_renderer)


        plot = figure(title="Graph layout demonstration", x_range=(-1.1,1.1),
                    y_range=(-1.1,1.1), tools="", toolbar_location=None, sizing_mode="scale_height")

        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), BoxSelectTool())

        draw_bokeh_commit_treegraph_iteration(self.graph)

        # specify the name of the output file
        output_file('graph.html')

        # display the plot
        show(plot)


    @staticmethod
    def rgb_to_hex(rgb):
        """ Converts rgb tuple to hex.
        """
        return '#%02x%02x%02x' % rgb