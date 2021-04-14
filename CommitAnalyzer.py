import pydriller
import git
import tempfile
import atexit
import shutil
import os
import networkx as nx
import matplotlib.pyplot as plt
import tqdm

from pyvis.network import Network
from bokeh.io import output_file, show
from bokeh.models import (BoxZoomTool, Circle, HoverTool, TapTool, BoxSelectTool,
                          MultiLine, Plot, Range1d, ResetTool, GraphRenderer, StaticLayoutProvider,
                          EdgesAndLinkedNodes, NodesAndLinkedEdges, WheelZoomTool, PanTool)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, figure

import TreeGraph
import compute_layout

class CommitAnalyzer:

    def __init__(self, url):
        """ Downloads the repo in a temp folder if it is not stored locally.
        Create a repository mining object to later analyze the commits.
        Registers a function to supress the temp folder at the end of the execution
        if the repo was stored remotely.

        Attributes :
            url : url of the repo (either remote or local)
            repo_folder : folder where repo is stored (same as url if local repo)
            repository_mining : RepositoryMining object to analyze the repo
            git_repo : GitRepository object
            repo_files : list of files contained in the repo
            _tmp_dir : location of temp directory
        """

        self.url = url

        # Clone repo if necessary
        if self._is_remote_repository(url):
            self.repo_folder = self._clone_remote_repository(self._clone_folder(), url)
        else:
            self.repo_folder = url

        # Get a RepositoryMining object
        self.repository_mining = pydriller.RepositoryMining(self.repo_folder)

        # Get a GitRepository object
        self.git_repo = pydriller.GitRepository(self.repo_folder)

        # Get list of files
        self.repo_files_paths = self.git_repo.files()
        self.repo_files = [os.path.basename(file) for file in self.repo_files_paths]
        self.total_commits = self.git_repo.total_commits()

        # Create graph of all commits
        self.commit_graph = nx.Graph()
        self.commit_graph.add_nodes_from([(filename, {'number_modifications': 0, 'index': filename}) for filename in self.repo_files])
        print([self.commit_graph.nodes[node]["index"] for node in self.commit_graph.nodes])

        # Create a dict mapping filename to path in repo
        self.filename_to_path = {}

        # Create TreeGraph
        self.commit_tree_graph = TreeGraph.TreeGraph(self._get_repo_name_from_url(self.url), False)
        self.path_prefix = os.path.commonpath(self.repo_files_paths)
        for file_path in self.repo_files_paths:
            file_path = file_path[len(self.path_prefix)+1:]
            self.filename_to_path[os.path.basename(file_path)] = file_path
            split_path = file_path.split('\\')
            self.commit_tree_graph.add_children(split_path)

        # Remove temp folder at end of execution
        atexit.register(self._cleanup)

    
    @staticmethod
    def _is_remote_repository(repo: str) -> bool:
        """ Checks wether or not repo is a local or remote path
        to a repo.
        """

        return repo.startswith("git@") or repo.startswith("https://")

    def _clone_remote_repository(self, path_to_folder: str, repo: str) -> str:
        """ Clones the remote repo to path_to_folder.
        """

        repo_folder = os.path.join(path_to_folder, self._get_repo_name_from_url(repo))
        git.Repo.clone_from(url=repo, to_path=repo_folder)

        return repo_folder

    def _clone_folder(self) -> str:
        """ Create and returns a temporary folder.
        """

        self._tmp_dir = tempfile.TemporaryDirectory()
        clone_folder = self._tmp_dir.name

        return clone_folder

    @staticmethod
    def _get_repo_name_from_url(url: str) -> str:
        """ Parses repo url to get its name.
        """

        last_slash_index = url.rfind("/")
        last_suffix_index = url.rfind(".git")
        if last_suffix_index < 0:
            last_suffix_index = len(url)

        if last_slash_index < 0 or last_suffix_index <= last_slash_index:
            raise Exception("Badly formatted url {}".format(url))

        return url[last_slash_index + 1:last_suffix_index]

    def _cleanup(self):
        """ Cleanup temporary folder at the end of execution.
        """

        if self._is_remote_repository(self.url):
            assert self._tmp_dir is not None
            try:
                self._tmp_dir.cleanup()
            except PermissionError:
                # on Windows, Python 3.5, 3.6, 3.7 are not able to delete
                # git directories because of read-only files.
                # In this case, just ignore the errors.
                shutil.rmtree(self._tmp_dir.name, ignore_errors=True)

    def analyze_correlation(self):
        """ Find files that are modified together (ie. in same commit).
        Create an edge between them, and update its value based.
        """

        pbar = tqdm.tqdm(total=self.total_commits)
        for commit in self.repository_mining.traverse_commits():

            modified_files =  [modification.filename for modification in commit.modifications]

            pairs_of_modified_files = []
            for i in range(len(modified_files)):
                for j in range(i+1, len(modified_files)):
                    pairs_of_modified_files.append((modified_files[i], modified_files[j]))

            # TreeCommit Graph
            self.analyze_correlation_treecommit_graph(pairs_of_modified_files)

            # Commit Graph
            self.analyze_correlation_commit_graph(modified_files, pairs_of_modified_files)

            pbar.update(1)
        pbar.close()

    def analyze_correlation_commit_graph(self, modified_files, pairs_of_modified_files):

        for modified_file in modified_files:
            if modified_file in self.commit_graph.nodes:
                self.commit_graph.nodes[modified_file]['number_modifications'] += 1

        for edge in pairs_of_modified_files:

            if edge[0] in self.commit_graph.nodes and edge[1] in self.commit_graph.nodes:
                if self.commit_graph.has_edge(edge[0], edge[1]):
                    self.commit_graph.edges[edge[0], edge[1]]['number_modifications_same_commit'] += 1
                else:
                    self.commit_graph.add_edge(edge[0], edge[1], number_modifications_same_commit=1)

    def analyze_correlation_treecommit_graph(self, pairs_of_modified_files):

        for (node1, node2) in pairs_of_modified_files:

            if node1 in self.filename_to_path and node2 in self.filename_to_path:

                # Find common prefix
                path_node1 = self.filename_to_path[node1]
                path_node2 = self.filename_to_path[node2]
                path_prefix = os.path.commonpath([path_node1, path_node2])
                
                if len(path_prefix) > 0:
                    path_prefix_split = path_prefix.split('\\')
                    tree_commit_node_name1 = path_node1[len(path_prefix)+1:].split('\\')[0]
                    tree_commit_node_name2 = path_node2[len(path_prefix)+1:].split('\\')[0]
                else:
                    path_prefix_split = []
                    tree_commit_node_name1 = path_node1[len(path_prefix):].split('\\')[0]
                    tree_commit_node_name2 = path_node2[len(path_prefix):].split('\\')[0]

                # Create or update edge in TreeCommit graph
                self.commit_tree_graph.add_edge(path_prefix_split, tree_commit_node_name1, tree_commit_node_name2)

    def compute_correlation(self, node_name):

        number_modifications = self.commit_graph.nodes[node_name]["number_modifications"]
        neighbors_correlation = []

        for neighbor in self.commit_graph.neighbors(node_name):

            number_modifications_same_commit = self.commit_graph.edges[node_name, neighbor]["number_modifications_same_commit"]
            neighbors_correlation.append((neighbor, 100*number_modifications_same_commit/number_modifications, number_modifications_same_commit))
        
        neighbors_correlation.sort(key=lambda x: x[1], reverse=True)

        print(f'Correlation of {node_name} (modified in {number_modifications} commits) with :')
        for neighbor in neighbors_correlation:
            print(f'{neighbor[0]} : {neighbor[1]}% (modified {neighbor[2]} times)')

    def draw_networkx(self):

        # Layout
        pos = nx.spring_layout(self.commit_graph, weight='number_modifications_same_commit')

        # Edge Width
        edges = self.commit_graph.edges()
        number_time_modified_together = [self.commit_graph[u][v]['number_modifications_same_commit'] for u,v in edges]
        max_number_time_modified_together = max(number_time_modified_together)
        width = [num / max_number_time_modified_together for num in number_time_modified_together]

        nx.draw(self.commit_graph, pos=pos, with_labels=True, width=width)
        plt.show()

    def draw_pyvis(self):

        # Edge Width
        edges = self.commit_graph.edges()
        number_time_modified_together = [self.commit_graph[u][v]['number_modifications_same_commit'] for u,v in edges]
        max_number_time_modified_together = max(number_time_modified_together)

        # Draw
        nt = Network(height='100%', width='70%')
        nt.from_nx(self.commit_graph)
        

        for edge in nt.get_edges():
            edge['value'] = self.commit_graph[edge['from']][edge['to']]['number_modifications_same_commit'] / max_number_time_modified_together

        for node_id in nt.get_nodes():
            node = nt.get_node(node_id)
            node['color'] = self.rgb_to_hex((self.commit_graph.nodes[node_id]['number_modifications'], 0, 0))
            print(node['color'])

        nt.show_buttons(filter_=['physics'])
        nt.show('nx.html')

    def draw_bokeh(self):

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.5,1.5), y_range=Range1d(-1.5,1.5))
        plot.add_tools(HoverTool(tooltips=[("index", "@index")]), TapTool(), WheelZoomTool(), ResetTool(), PanTool())

        graph_renderer = from_networkx(self.commit_graph, nx.spring_layout, scale=1, center=(0,0), k=1)

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

    def draw_bokeh_commit_treegraph(self):

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

        draw_bokeh_commit_treegraph_iteration(self.commit_tree_graph)

        # specify the name of the output file
        output_file('graph.html')

        # display the plot
        show(plot)

        


    @staticmethod
    def draw_bokeh_test():


        plot = figure(title="Graph layout demonstration", x_range=(-1.1,1.1),
                    y_range=(-1.1,1.1), tools="", toolbar_location=None, sizing_mode="scale_height")

        # Graph 1
        N = 4
        node_indices = list(range(4))

        graph_renderer = GraphRenderer()

        graph_renderer.node_renderer.glyph = Circle(radius=0.3, fill_color=Spectral4[0])
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)

        graph_renderer.node_renderer.data_source.data = dict(
            index=node_indices,
            fill_color=Spectral4)

        graph_renderer.edge_renderer.data_source.data = dict(
            start=[0]*N,
            end=node_indices)

        x = [0, 0.5, -0.5, 1]
        y = [0.5, 0, -0.5, -1]

        graph_layout = dict(zip(node_indices, zip(x, y)))

        graph_renderer.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        # render the graph
        plot.renderers.append(graph_renderer)


        ##################################################
        # Graph 2
        N = 4
        node_indices = list(range(4))

        graph_renderer2 = GraphRenderer()

        graph_renderer2.node_renderer.glyph = Circle(radius=0.03, fill_color=Spectral4[1])
        graph_renderer2.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)

        graph_renderer2.node_renderer.data_source.data = dict(
            index=node_indices,
            fill_color=Spectral4)

        graph_renderer2.edge_renderer.data_source.data = dict(
            start=[0]*N,
            end=node_indices)

        x = [0.05, 0, -0.05, 0]
        y = [0.5, 0.48, 0.5, 0.7]

        graph_layout = dict(zip(node_indices, zip(x, y)))

        graph_renderer2.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

        # render the graph
        plot.renderers.append(graph_renderer2)

        # specify the name of the output file
        output_file('graph.html')

        # display the plot
        show(plot)

    @staticmethod
    def rgb_to_hex(rgb):

        return '#%02x%02x%02x' % rgb



if __name__ == "__main__":

    url = "https://github.com/ishepard/pydriller.git"

    # CommitAnalyzer.draw_bokeh_test()
    
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)
    
    print("Running analysis")
    ca.analyze_correlation()

    ca.compute_correlation('.flake8')
    
    """
    print("Drawing results")
    ca.draw_bokeh()
    """