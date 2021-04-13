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
from bokeh.models import (BoxZoomTool, Circle, HoverTool,
                          MultiLine, Plot, Range1d, ResetTool, GraphRenderer, StaticLayoutProvider,)
from bokeh.palettes import Spectral4
from bokeh.plotting import from_networkx, figure

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

        if self._is_remote_repository(url):
            self.repo_folder = self._clone_remote_repository(self._clone_folder(), url)
        else:
            self.repo_folder = url

        self.repository_mining = pydriller.RepositoryMining(self.repo_folder)

        self.git_repo = pydriller.GitRepository(self.repo_folder)

        self.repo_files = [os.path.basename(file) for file in self.git_repo.files()]
        self.total_commits = self.git_repo.total_commits()

        self.commit_graph = nx.Graph()
        self.commit_graph.add_nodes_from([(filename, {'number_modifications': 0}) for filename in self.repo_files])

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
        for commit in ca.repository_mining.traverse_commits():

            modified_files =  [modification.filename for modification in commit.modifications]

            for modified_file in modified_files:
                if modified_file in self.commit_graph.nodes:
                    self.commit_graph.nodes[modified_file]['number_modifications'] += 1

            pairs_of_modified_files = []
            for i in range(len(modified_files)):
                for j in range(i+1, len(modified_files)):
                    pairs_of_modified_files.append((modified_files[i], modified_files[j]))

            for edge in pairs_of_modified_files:

                if edge[0] in self.commit_graph.nodes and edge[1] in self.commit_graph.nodes:
                    if self.commit_graph.has_edge(edge[0], edge[1]):
                        self.commit_graph.edges[edge[0], edge[1]]['number_modifications_same_commit'] += 1
                    else:
                        self.commit_graph.add_edge(edge[0], edge[1], number_modifications_same_commit=0)

            pbar.update(1)
        pbar.close()

    def draw_networkx(self):

        # Layout
        pos = nx.spring_layout(self.commit_graph, weight='number_modifications_same_commit')

        # Edge Width
        edges = ca.commit_graph.edges()
        number_time_modified_together = [ca.commit_graph[u][v]['number_modifications_same_commit'] for u,v in edges]
        max_number_time_modified_together = max(number_time_modified_together)
        width = [num / max_number_time_modified_together for num in number_time_modified_together]

        nx.draw(self.commit_graph, pos=pos, with_labels=True, width=width)
        pls.show()

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

        plot = Plot(sizing_mode="scale_height", x_range=Range1d(-1.1,1.1), y_range=Range1d(-1.1,1.1))
        graph_renderer = from_networkx(self.commit_graph, nx.circular_layout, scale=1, center=(0,0))

        graph_renderer.node_renderer.glyph = Circle(size=15, fill_color=Spectral4[0])
        graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=5)

        plot.renderers.append(graph_renderer)

        output_file("interactive_graphs.html")
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

    CommitAnalyzer.draw_bokeh_test()
    """
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)

    print("Running analysis")
    ca.analyze_correlation()
    
    print("Drawing results")
    ca.draw_bokeh()
    """
    
    

    
