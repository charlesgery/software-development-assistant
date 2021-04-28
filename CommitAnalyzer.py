import pydriller
import git
import tempfile
import atexit
import shutil
import os
import networkx as nx
import matplotlib.pyplot as plt
import tqdm
import zipfile
import pickle
import pandas as pd
import hdbscan


import TreeGraph
import compute_layout
import CommitGraphDrawer
import CommitTreeGraphDrawer
import Correlation

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
            git_repo2 : PyGit Repo object
            repo_files_path : list of paths to the files contained in the repo
            repo_files : list of files contained in the repo
            total_commits : total number of commits
            commit_graph : networkx graph object of files in the repo
            filename_to_path : dict to get path of file in repo given its name
            commit_tree_graph : TreeGraph object of correlation between files
            path_prefix : path prefix specific to the computer you are using
            _tmp_dir : location of temp directory
        """

        self.url = url

        # Clone repo if necessary
        if self._is_remote_repository(url):
            self.repo_folder = self._clone_remote_repository(self._clone_folder(), url)
        else:
            self.repo_folder = url

        with open(self.repo_folder + '\\.gitattributes', 'a') as f:
            f.write('*.py   diff=python')

        # Get a RepositoryMining object
        self.repository_mining = pydriller.RepositoryMining(self.repo_folder)

        # Get a GitRepository object
        self.git_repo = pydriller.GitRepository(self.repo_folder)
        self.git_repo2 = git.Repo(self.repo_folder)
        self.total_commits = self.git_repo.total_commits()


        # Create graph of all commits
        self.commit_graph = nx.Graph()

        # Create graph of all commits lines where involved in
        # Create graph of all commits
        self.commit_graph_lines = nx.Graph()


        # Create TreeGraph
        self.commit_tree_graph = TreeGraph.TreeGraph(self._get_repo_name_from_url(self.url), False)

        # Get list of files
        repo_files_paths = self.git_repo.files()
        self.path_prefix = os.path.commonpath(repo_files_paths)
        self.repo_files_path = []
        for file_path in repo_files_paths:
            _, file_extension = os.path.splitext(file_path)
            if file_extension not in ['.zip', '.gif', '.png']:
                file_path = file_path[len(self.path_prefix)+1:]
                self.repo_files_path.append(file_path)
                split_path = file_path.split('\\')
                self.commit_tree_graph.add_children(split_path)
        self.commit_graph.add_nodes_from([(file_path, {'number_modifications': 0, 'index': file_path}) for file_path in self.repo_files_path])
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
        print(clone_folder)

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

    def save_graph(self, G, path):

        nx.readwrite.gpickle.write_gpickle(G, path)

    def load_commit_graph(self, path):

        self.commit_graph = nx.readwrite.gpickle.read_gpickle(path)

    def load_commit_graph_lines(self, path):

        self.commit_graph_lines = nx.readwrite.gpickle.read_gpickle(path)
        
    def find_lines_related_to_function(self, function_name, path):

        modified_in_commits = self.get_commits_that_modified_function(function_name, path)
        self.find_related_lines(path, modified_in_commits)

    def find_lines_related_to_lines(self, start_line, end_line, path):
        """ Find lines in other files that are related to line in a given file,
        based on commit history.
        """

        modified_in_commits = self.get_commits_that_modified_line(start_line, end_line, path)
        self.find_related_lines(path, modified_in_commits)
        
    def find_related_lines(self, path, modified_in_commits):

        related_lines = {}
        line_history = {}

        for commit in pydriller.RepositoryMining(self.repo_folder, only_commits=modified_in_commits).traverse_commits():

            for modification in tqdm.tqdm(commit.modifications):

                path = path.replace("/", "\\")
                if modification.new_path in self.repo_files_path and not modification.new_path[-4:] == '.zip' and modification.new_path != path:

                    # Get path to file to count number of lines
                    filepath = self.repo_folder + '\\' + modification.new_path
                    if os.path.getsize(filepath):
                        with open(filepath) as f:
                            for i, _ in enumerate(f):
                                pass
                            linenumber = i + 1
                    else:
                        linenumber = 0
                    
                    # Split file in group of 10 lines and check of they are linked to the modified line
                    if linenumber > 0:
                        self.get_related_lines_precise(related_lines, linenumber, modification.new_path, commit.hash, line_history)

        self.display_related_lines(related_lines, len(modified_in_commits))
      

    def get_related_lines_fast(self, related_lines, linenumber, file_path, commit_hash):

        for i in range(1, linenumber, 10):
            if i + 10 > linenumber:
                modified_in_commits2 = self.get_commits_that_modified_line(i, linenumber, file_path)
            else:
                modified_in_commits2 = self.get_commits_that_modified_line(i, i+9, file_path)
        
            if commit_hash in modified_in_commits2:
                if file_path in related_lines:
                    if i not in related_lines[file_path]:
                        for j in range(10):
                            related_lines[file_path][i+j] += 1
                    else:
                        for j in range(10):
                            related_lines[file_path][i+j] = 1
                    if not self.interval_contained_in_list(related_lines[file_path], (i, i+9)):
                        self.insert_interval_in_list(related_lines[file_path], (i, i+9))
                else:
                    related_lines[file_path] = {i:1}
                    for j in range(1, 10):
                            related_lines[file_path][i+j] = 1
                    


    def get_related_lines_precise(self, related_lines, linenumber, file_path, commit_hash, line_history):

        if file_path not in line_history:
            line_history[file_path] = {}
            for i in range(1, linenumber):
                modified_in_commits2 = self.get_commits_that_modified_line(i, i, file_path)
                line_history[file_path][i] = modified_in_commits2

        for i in range(1, linenumber):
            if commit_hash in line_history[file_path][i]:
                if file_path in related_lines:
                    if i in related_lines[file_path]:
                        related_lines[file_path][i] += 1
                    else:
                        related_lines[file_path][i] = 1
                    
                else:
                    related_lines[file_path] = {i:1}

    @staticmethod
    def display_related_lines(related_lines, num_modifications):

        most_correlated_lines = []

        for file_path in related_lines:

            file_correlation_string = ''
            file_correlation_string += f'File {file_path}'
            lines = []
            for key in related_lines[file_path]:
                lines.append(key)
                most_correlated_lines.append((key, file_path, related_lines[file_path][key], f'{100*related_lines[file_path][key]/num_modifications}%'))
            lines.sort()
            start, end = lines[0], lines[0]
            for i in range(1, len(lines)):
                if lines[i] == end + 1:
                    end += 1
                else:
                    file_correlation_string += f' {start}-{end}'
                    start, end = lines[i], lines[i]
            file_correlation_string += f' {start}-{end}'
            print(file_correlation_string)

        most_correlated_lines.sort(key=lambda x: x[2], reverse=True)
        for (line, file_path, num_modifications_line, correlation) in most_correlated_lines:
            print(f'Line {line} of {file_path} is {correlation} correlated ({num_modifications_line} modifs)')
                    


    def get_commits_that_modified_line(self, start_line, end_line, path):

        history = self.git_repo2.git.log('-L', f'{start_line},{end_line}:{path}').split('\n')
        modified_in_commits = []

        for line in history:
            if line[0:6] == 'commit':
                modified_in_commits.append(line[7:])
        
        return modified_in_commits

    def get_commits_that_modified_function(self, function_name, path):

        history = self.git_repo2.git.log('-L', f':{function_name}:{path}').split('\n')
        modified_in_commits = []

        for line in history:
            if line[0:6] == 'commit':
                modified_in_commits.append(line[7:])
        
        return modified_in_commits
                    
    @staticmethod
    def interval_contained_in_list(list_intervals, interval):

        for (a, b) in list_intervals:

            if a <= interval[0] and interval[1] <= b:
                return True
        
        return False

    @staticmethod
    def insert_interval_in_list(list_intervals, interval):

        merge_left, merge_right = False, False
        for (a, b) in list_intervals:
            if b == interval[0] - 1:
                merge_left = True
                merge_left_pair = (a, b)
            if a == interval[1] + 1:
                merge_right = True
                merge_right_pair = (a, b)
        if merge_left and merge_right:
            list_intervals.remove(merge_left_pair)
            list_intervals.remove(merge_right_pair)
            list_intervals.append((merge_left_pair[0], merge_right_pair[1]))
        elif merge_left:
            list_intervals.remove(merge_left_pair)
            list_intervals.append((merge_left_pair[0], interval[1]))
        elif merge_right:
            list_intervals.remove(merge_right_pair)
            list_intervals.append((interval[0], merge_right_pair[1]))
        else:
            list_intervals.append(interval)


    def analyze_correlation(self, treecommit_analysis=False, commit_analysis=False, commit_lines_analysis=False):
        """ Find files/folders that are modified together (ie. in same commit).
        Update commit and TreeCommit graphs accordingly.
        """

        if treecommit_analysis or commit_analysis:
            pbar = tqdm.tqdm(total=self.total_commits)
            for commit in self.repository_mining.traverse_commits():

                modified_files =  [modification.new_path for modification in commit.modifications]

                pairs_of_modified_files = []
                for i in range(len(modified_files)):
                    for j in range(i+1, len(modified_files)):
                        pairs_of_modified_files.append((modified_files[i], modified_files[j]))

                # TreeCommit Graph
                if treecommit_analysis:
                    self.analyze_correlation_treecommit_graph(pairs_of_modified_files)

                # Commit Graph
                if commit_analysis:
                    self.analyze_correlation_commit_graph(modified_files, pairs_of_modified_files)

                pbar.update(1)
            pbar.close()

        # Commit Graph lines
        if commit_lines_analysis:
            self.analyze_correlation_commit_lines_graph()

    def analyze_correlation_commit_graph(self, modified_files, pairs_of_modified_files):
        """ Find files that are modified together (ie. in same commit).
        Create an edge between them, and update its value.
        """

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
        """ Find files/folders that are modified together (ie. in same commit).
        Create an edge between them, and update its value.
        """

        for (node1, node2) in pairs_of_modified_files:

            if node1 in self.repo_files_path and node2 in self.repo_files_path:

                # Find common prefix
                path_prefix = os.path.commonpath([node1, node2])
                
                if len(path_prefix) > 0:
                    path_prefix_split = path_prefix.split('\\')
                    tree_commit_node_name1 = node1[len(path_prefix)+1:].split('\\')[0]
                    tree_commit_node_name2 = node2[len(path_prefix)+1:].split('\\')[0]
                else:
                    path_prefix_split = []
                    tree_commit_node_name1 = node1[len(path_prefix):].split('\\')[0]
                    tree_commit_node_name2 = node2[len(path_prefix):].split('\\')[0]

                # Create or update edge in TreeCommit graph
                self.commit_tree_graph.add_edge(path_prefix_split, tree_commit_node_name1, tree_commit_node_name2)

    def analyze_correlation_commit_lines_graph(self):

        commit_to_lines = {}

        # Print analyzing all the lines of the repo
        print('Print analyzing all the lines of the repo')
        for file_path in tqdm.tqdm(self.repo_files_path):

            print(file_path)
            # Get path to file and count number of lines
            complete_file_path = self.repo_folder + '\\' + file_path
            if os.path.getsize(complete_file_path):
                with open(complete_file_path, 'rb') as f:
                    for i, _ in enumerate(f):
                        pass
                    linenumber = i + 1
            else:
                linenumber = 0

            for line in range(1, linenumber):

                modified_in_commits = self.get_commits_that_modified_line(line, line, file_path)
                self.commit_graph_lines.add_node(f'{file_path}:{line}', number_modifications=len(modified_in_commits))

                for commit in modified_in_commits:

                    if commit in commit_to_lines:
                        commit_to_lines[commit].append(f'{file_path}:{line}')
                    else:
                        commit_to_lines[commit] = [f'{file_path}:{line}']

        # Building the graph
        print('\n\nBuilding the graph')
        for (commit, list_lines) in tqdm.tqdm(commit_to_lines.items()):

            pairs_of_modified_lines = []
            for i in range(len(list_lines)):
                for j in range(i+1, len(list_lines)):
                    pairs_of_modified_lines.append((list_lines[i], list_lines[j]))

            for edge in pairs_of_modified_lines:

                if edge[0] in self.commit_graph_lines.nodes and edge[1] in self.commit_graph_lines.nodes:
                    if self.commit_graph_lines.has_edge(edge[0], edge[1]):
                        self.commit_graph_lines.edges[edge[0], edge[1]]['number_modifications_same_commit'] += 1
                    else:
                        self.commit_graph_lines.add_edge(edge[0], edge[1], number_modifications_same_commit=1)



    @staticmethod
    def compute_correlation(node_name, commit_graph, method='basic', alpha=0.5):
        """ Compute correlation between a file and another one in commit graph based on value of edge.
        Correlation = Value of edge / max value of edge for this node
        """

        number_modifications = commit_graph.nodes[node_name]["number_modifications"]
        neighbors_correlation = []

        for neighbor in commit_graph.neighbors(node_name):

            number_modifications_same_commit = commit_graph.edges[node_name, neighbor]["number_modifications_same_commit"]
            number_modifications_neighbor = commit_graph.nodes[neighbor]["number_modifications"]

            if method == 'basic':
                correlation = Correlation.Correlation.basic_correlation(number_modifications_same_commit, number_modifications)

            elif method == 'addition':

                correlation = Correlation.Correlation.addition_correlation(number_modifications_same_commit, number_modifications, number_modifications_neighbor, alpha)
            
            elif method == 'multiplication':

                correlation = Correlation.Correlation.multiplication_correlation(number_modifications_same_commit, number_modifications, number_modifications_neighbor, alpha)

            neighbors_correlation.append((neighbor, 100*number_modifications_same_commit/number_modifications, number_modifications_same_commit))
        
        neighbors_correlation.sort(key=lambda x: x[1], reverse=True)

        print(f'Correlation of {node_name} (modified in {number_modifications} commits) with :')
        for i, neighbor in enumerate(neighbors_correlation):
            if i < 50:
                print(f'{neighbor[0]} : {neighbor[1]}% (modified {neighbor[2]} times)')
            else:
                break


    def compute_same_level_correlation(self, node_path):
        """ Compute correlation between a file/folder and another one in commit TreeGraph based on value of edge.
        Correlation = Value of edge / max value of edge for this node
        """

        def compute_same_level_correlation_iteration(tree_graph, splitted_path):

            if len(splitted_path) == 1 and splitted_path[0] in tree_graph.kids:
                self.compute_correlation(splitted_path[0], tree_graph.graph)
            elif len(splitted_path) > 1 and splitted_path[0] in tree_graph.kids:
                compute_same_level_correlation_iteration(tree_graph.kids[splitted_path[0]], splitted_path[1:])


        tree_graph = self.commit_tree_graph

        splitted_path = node_path.split('\\')
        print(splitted_path)

        compute_same_level_correlation_iteration(tree_graph, splitted_path)

    def compute_files_that_should_be_in_commit(self, commit_hash):

        similar_commits = {}
        potential_nodes = set()

        # Get list of files modified in commit
        modified_files = []
        modified_files_dict = {}
        for commit in pydriller.RepositoryMining(self.repo_folder, single=commit_hash).traverse_commits():
            for modification in commit.modifications:
                modified_files.append(modification.new_path)
                modified_files_dict[modification.new_path] = 1

        # Compute each commit similarity score
        print('Computing similarity score')
        for commit in tqdm.tqdm(pydriller.RepositoryMining(self.repo_folder).traverse_commits()):
            if commit.hash != commit_hash:
                modified_files_other_commit = []
                new_nodes = []
                similar_nodes = 0
                for modification in commit.modifications:
                    if modification.new_path in modified_files_dict:
                        similar_nodes += 1
                    else:
                        new_nodes.append(modification.new_path)
                    modified_files_other_commit.append(modification.new_path)
                similarity = similar_nodes / max(len(modified_files), len(modified_files_other_commit))
                if similarity > 0.3:
                    similar_commits[commit.hash] = (similarity, new_nodes)
                    for node in new_nodes:
                        if node not in potential_nodes:
                            potential_nodes.add(node)

        # Compute score of new potential nodes
        print('Compute node scores')
        for node in tqdm.tqdm(potential_nodes):
            node_score = 0
            for _, (similarity, nodes) in similar_commits.items():
                if node in nodes:
                    node_score += similarity
            node_score /= len(similar_commits)
            modified_files_dict[node] = node_score

        for node in self.repo_files_path:
            if node not in modified_files_dict:
                modified_files_dict[node] = 0

        return modified_files_dict

    def create_commits_dataframe(self):

        files_commits = {}
        current_length = 0
        columns = []

        pbar = tqdm.tqdm(total=self.total_commits)
        for commit in self.repository_mining.traverse_commits():

            current_length += 1
            columns.append(commit.hash)

            for modification in commit.modifications:
                
                if modification.new_path in self.repo_files_path:

                    if modification.new_path in files_commits:

                        while len(files_commits[modification.new_path]) < current_length - 1:
                            files_commits[modification.new_path].append(0)
                        files_commits[modification.new_path].append(1)
                    
                    else:
                        files_commits[modification.new_path] = [0 for _ in range(current_length-1)]
                        files_commits[modification.new_path].append(1)

            pbar.update(1)
        pbar.close()

        dataframe_list = []
        index = []
        for key, value in files_commits.items():

            if len(value) < current_length:

                while len(files_commits[key]) < current_length:
                        files_commits[key].append(0)

            index.append(key)
            dataframe_list.append(value)

        return pd.DataFrame(dataframe_list, index=index, columns=columns)

    def cluster_dataframe(self, df):

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        clusterer.fit(df)

        filenames = df.index.tolist()
        clusters = {}

        for (filename, cluster) in zip(filenames, clusterer.labels_):

            if filename in self.repo_files_path:

                if cluster in clusters:
                    clusters[cluster].append(filename)
                else:
                    clusters[cluster] = [filename]

        for key, value in clusters.items():

            print(f'Cluster {key} : {value}')

                    





if __name__ == "__main__":
    
    # url = "https://github.com/apache/spark.git"
    # url = "https://github.com/ishepard/pydriller.git"
    url = "https://github.com/oilshell/oil.git"
    
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)
    
    print("Running analysis")


    print("Clustering analysis")
    # df = ca.create_commits_dataframe()
    # ca.cluster_dataframe(df)
    

    print("Correlation analysis")
    # ca.analyze_correlation(treecommit_analysis=False, commit_analysis=True, commit_lines_analysis=False)
    # ca.save_graph(ca.commit_graph, './commit_graph_oil.bz2')
    # ca.analyze_correlation(treecommit_analysis=False, commit_analysis=False, commit_lines_analysis=True)
    # ca.save_graph(ca.commit_graph_lines, './commit_graph_lines_oil.bz2')
    # ca.load_commit_graph_lines('./commit_graph_lines_specter.bz2')
    ca.load_commit_graph('./commit_graph_oil.bz2')

    print("Commit analysis")
    modified_files = ca.compute_files_that_should_be_in_commit('225a29a2b904427f955756f67db6c5d572edcddc')

    
    with open('modified_files_oil.pickle', 'wb') as handle:
        pickle.dump(modified_files, handle)

    
    # with open('modified_files_spark.pickle', 'rb') as handle:
    #    modified_files = pickle.load(handle)
    
    print(modified_files)

    related_nodes = []
    for (key, value) in modified_files.items():
        if value > 0:
            related_nodes.append((key, value))
    
    related_nodes.sort(key=lambda x: x[1], reverse=True)
    print(related_nodes[:50])
    

    print("Reverse correlation")
    # ca.compute_correlation_reverse('core\\pyos.py', ca.commit_graph, 0.5)
    
    print("Line correlation")
    # ca.find_lines_related_to_lines(12, 20, 'src/clj/com/rpl/specter/transients.cljc')

    print('Function correlation')
    # ca.find_lines_related_to_function('get_head', 'pydriller/git_repository.py')
    
    print("Same level correlation")
    # ca.compute_same_level_correlation('pydriller')
    
    print("Drawing results")
    # drawer = CommitGraphDrawer.CommitGraphDrawer(ca.commit_graph)
    # drawer.draw_commit_missing_files_bokeh(modified_files)
    # drawer.draw_bokeh()
    