import networkx as nx
import matplotlib.pyplot as plt


from pyvis.network import Network
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool, TapTool, BoxSelectTool,
                          MultiLine, Plot, Range1d, ResetTool, GraphRenderer, StaticLayoutProvider,
                          EdgesAndLinkedNodes, NodesAndLinkedEdges, WheelZoomTool, PanTool)
from bokeh.palettes import Spectral4, magma, turbo
from bokeh.plotting import from_networkx, figure
from bokeh.colors.groups import black
from bokeh.transform import linear_cmap

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

    def draw_bokeh(self, layout=None):
        """ Draw commit graph using Bokeh.
        """

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.5,1.5), y_range=Range1d(-1.5,1.5))
        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), WheelZoomTool(), ResetTool(), PanTool())

        if layout is None:
            graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)
        else:
            graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        if layout:
            graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=layout)

        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)

    def draw_bokeh_software_as_cities(self, layout=None, routes=None):
        """ Draw commit graph using Bokeh.
        """

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.5,1.5), y_range=Range1d(-1.5,1.5))
        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), WheelZoomTool(), ResetTool(), PanTool())

        if layout is None:
            graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)
        else:
            graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        data = graph_renderer.edge_renderer.data_source.data
        normalization_value = max(list(routes.values()))
        data["line_width"] = [routes[edge] * 10 / normalization_value for edge in zip(data["start"], data["end"])]
        graph_renderer.edge_renderer.glyph.line_width = {'field': 'line_width'}

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        if layout:
            graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=layout)

        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)

    def draw_commit_missing_files_bokeh(self, files_scores):

        def translate(value, leftMin, leftMax, rightMin, rightMax):
            # Figure out how 'wide' each range is
            leftSpan = leftMax - leftMin
            rightSpan = rightMax - rightMin

            # Convert the left range into a 0-1 range (float)
            valueScaled = float(value - leftMin) / float(leftSpan)

            # Convert the 0-1 range into a value in the right range.
            return rightMin + (valueScaled * rightSpan)

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.5,1.5), y_range=Range1d(-1.5,1.5))
        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), WheelZoomTool(), ResetTool(), PanTool())

        graph_renderer = from_networkx(self.graph, nx.spring_layout, scale=1, center=(0,0), k=1)

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color='fill_color')
        graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
        graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)

        graph_renderer.selection_policy = NodesAndLinkedEdges()

        color = []
        score = []
        for node in graph_renderer.node_renderer.data_source.data['index']:
                if files_scores[node] == 0:
                    color.append("#ffffff")
                elif files_scores[node] == 1:
                    color.append("#ff0000")
                else:
                    color.append(turbo(256)[int(files_scores[node]*256)])
        graph_renderer.node_renderer.data_source.data['fill_color'] = color

        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
        show(plot)

    @staticmethod
    def rgb_to_hex(rgb):
        """ Converts rgb tuple to hex.
        """

        return '#%02x%02x%02x' % rgb