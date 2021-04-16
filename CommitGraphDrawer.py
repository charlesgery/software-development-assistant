import networkx as nx
import matplotlib.pyplot as plt


from pyvis.network import Network
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool, TapTool, BoxSelectTool,
                          MultiLine, Plot, Range1d, ResetTool, GraphRenderer, StaticLayoutProvider,
                          EdgesAndLinkedNodes, NodesAndLinkedEdges, WheelZoomTool, PanTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, figure

import compute_layout


class CommitGraphDrawer:

    def __init__(self, graph):

        self.graph = graph

    def draw_networkx(self):
        """ Draw commit graph using networkx.
        """

        # Layout
        pos = nx.spring_layout(self.graph, weight='number_modifications_same_commit')

        # Edge Width
        edges = self.graph.edges()
        number_time_modified_together = [self.graph[u][v]['number_modifications_same_commit'] for u,v in edges]
        max_number_time_modified_together = max(number_time_modified_together)
        width = [num / max_number_time_modified_together for num in number_time_modified_together]

        nx.draw(self.graph, pos=pos, with_labels=True, width=width)
        plt.show()

    def draw_pyvis(self):
        """ Draw commit graph using Pyvis.
        """

        # Edge Width
        edges = self.graph.edges()
        number_time_modified_together = [self.graph[u][v]['number_modifications_same_commit'] for u,v in edges]
        max_number_time_modified_together = max(number_time_modified_together)

        # Draw
        nt = Network(height='100%', width='70%')
        nt.from_nx(self.graph)
        

        for edge in nt.get_edges():
            edge['value'] = self.graph[edge['from']][edge['to']]['number_modifications_same_commit'] / max_number_time_modified_together

        for node_id in nt.get_nodes():
            node = nt.get_node(node_id)
            node['color'] = self.rgb_to_hex((self.graph.nodes[node_id]['number_modifications'], 0, 0))
            print(node['color'])

        nt.show_buttons(filter_=['physics'])
        nt.show('nx.html')

    def draw_bokeh(self):
        """ Draw commit graph using Bokeh.
        """

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.5,1.5), y_range=Range1d(-1.5,1.5))
        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), WheelZoomTool(), ResetTool(), PanTool())

        graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)

    @staticmethod
    def rgb_to_hex(rgb):
        """ Converts rgb tuple to hex.
        """

        return '#%02x%02x%02x' % rgb