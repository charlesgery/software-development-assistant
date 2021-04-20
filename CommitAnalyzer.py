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


import TreeGraph
import compute_layout
import CommitGraphDrawer
import CommitTreeGraphDrawer

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

        # Get a RepositoryMining object
        self.repository_mining = pydriller.RepositoryMining(self.repo_folder)

        # Get a GitRepository object
        self.git_repo = pydriller.GitRepository(self.repo_folder)
        self.git_repo2 = git.Repo(self.repo_folder)

        # Get list of files
        self.repo_files_paths = self.git_repo.files()
        self.repo_files = [os.path.basename(file) for file in self.repo_files_paths]
        self.total_commits = self.git_repo.total_commits()

        # Create graph of all commits
        self.commit_graph = nx.Graph()
        self.commit_graph.add_nodes_from([(filename, {'number_modifications': 0, 'index': filename}) for filename in self.repo_files])
        print([self.commit_graph.nodes[node]["index"] for node in self.commit_graph.nodes])
        print('\n\n')

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
        
            
    def find_related_lines(self, start_line, end_line, path):
        """ Find lines in other files that are related to line in a given file,
        based on commit history.
        """

        modified_in_commits = self.get_commits_that_modified_line(start_line, end_line, path)
        related_lines = {}
        line_history = {}

        for commit in pydriller.RepositoryMining(self.repo_folder, only_commits=modified_in_commits).traverse_commits():

            for modification in tqdm.tqdm(commit.modifications):

                path = path.replace("/", "\\")
                if modification.filename in self.filename_to_path and not modification.filename[-4:] == '.zip' and self.filename_to_path[modification.filename] != path:

                    # Get path to file to count number of lines
                    filepath = self.repo_folder + '\\' + self.filename_to_path[modification.filename]
                    if os.path.getsize(filepath):
                        with open(filepath) as f:
                            for i, _ in enumerate(f):
                                pass
                            linenumber = i + 1
                    else:
                        linenumber = 0
                    
                    # Split file in group of 10 lines and check of they are linked to the modified line
                    if linenumber > 0:
                        self.get_related_lines2(related_lines, linenumber, modification.filename, commit.hash, line_history)

        self.display_related_lines(related_lines)

    def get_related_lines(self, related_lines, linenumber, filename, commit_hash):

        for i in range(1, linenumber, 10):
            if i + 10 > linenumber:
                modified_in_commits2 = self.get_commits_that_modified_line(i, linenumber, self.filename_to_path[filename])
            else:
                modified_in_commits2 = self.get_commits_that_modified_line(i, i+9, self.filename_to_path[filename])
        
            if commit_hash in modified_in_commits2:
                if filename in related_lines:
                    if i not in related_lines[filename]:
                        for j in range(10):
                            related_lines[filename][i+j] += 1
                    else:
                        for j in range(10):
                            related_lines[filename][i+j] = 1
                    if not self.interval_contained_in_list(related_lines[filename], (i, i+9)):
                        self.insert_interval_in_list(related_lines[filename], (i, i+9))
                else:
                    related_lines[filename] = {i:1}
                    for j in range(1, 10):
                            related_lines[filename][i+j] = 1
                    


    def get_related_lines2(self, related_lines, linenumber, filename, commit_hash, line_history):

        if filename not in line_history:
            line_history[filename] = {}
            for i in range(1, linenumber):
                modified_in_commits2 = self.get_commits_that_modified_line(i, i, self.filename_to_path[filename])
                line_history[filename][i] = modified_in_commits2

        for i in range(1, linenumber):
            if commit_hash in line_history[filename][i]:
                if filename in related_lines:
                    if i in related_lines[filename]:
                        related_lines[filename][i] += 1
                    else:
                        related_lines[filename][i] = 1
                    
                else:
                    related_lines[filename] = {i:1}

    @staticmethod
    def display_related_lines(related_lines):

        most_correlated_lines = []

        for filename in related_lines:

            file_correlation_string = ''
            file_correlation_string += f'File {filename}'
            lines = []
            for key in related_lines[filename]:
                lines.append(key)
                most_correlated_lines.append((key, filename, related_lines[filename][key]))
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
        print(most_correlated_lines[:10])
                    


    def get_commits_that_modified_line(self, start_line, end_line, path):

        history = self.git_repo2.git.log('-L', f'{start_line},{end_line}:{path}').split('\n')
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


    def analyze_correlation(self):
        """ Find files/folders that are modified together (ie. in same commit).
        Update commit and TreeCommit graphs accordingly.
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

    @staticmethod
    def compute_correlation(node_name, commit_graph):
        """ Compute correlation between a file and another one in commit graph based on value of edge.
        Correlation = Value of edge / max value of edge for this node
        """

        number_modifications = commit_graph.nodes[node_name]["number_modifications"]
        neighbors_correlation = []

        for neighbor in commit_graph.neighbors(node_name):

            number_modifications_same_commit = commit_graph.edges[node_name, neighbor]["number_modifications_same_commit"]
            neighbors_correlation.append((neighbor, 100*number_modifications_same_commit/number_modifications, number_modifications_same_commit))
        
        neighbors_correlation.sort(key=lambda x: x[1], reverse=True)

        print(f'Correlation of {node_name} (modified in {number_modifications} commits) with :')
        for i, neighbor in enumerate(neighbors_correlation):
            if i < 5    0:
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



if __name__ == "__main__":
    
    url = "https://github.com/ishepard/pydriller.git"
    
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)


    
    
    
    print("Running analysis")
    ca.analyze_correlation()

    ca.find_related_lines(94, 101, 'pydriller/git_repository.py')
    ca.compute_correlation('git_repository.py', ca.commit_graph)
    # print("\n\n")
    # ca.compute_same_level_correlation('pydriller')
    
    #print("Drawing results")
    #drawer = CommitGraphDrawer.CommitGraphDrawer(ca.commit_graph)
    #drawer.draw_bokeh()
    