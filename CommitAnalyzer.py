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
import concurrent.futures
import threading
import time
import subprocess
import sklearn
import prince
import numpy as np
import copy

from sklearn import cluster

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

        print('Creating Object')

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
        self.forbidden_file_extensions = ['.zip', '.gif', '.png']
        repo_files_paths = self.git_repo.files()
        self.path_prefix = os.path.commonpath(repo_files_paths)
        self.repo_files_path = []
        for file_path in repo_files_paths:
            _, file_extension = os.path.splitext(file_path)
            if file_extension not in self.forbidden_file_extensions:
                file_path = file_path[len(self.path_prefix)+1:]
                self.repo_files_path.append(file_path)
                split_path = file_path.split('\\')
                self.commit_tree_graph.add_children(split_path)
        self.commit_graph.add_nodes_from([(file_path, {'number_modifications': 0, 'index': file_path}) for file_path in self.repo_files_path])
        
        # Find earlier names and paths of these files
        self.old_to_new_path = {}
        pbar = tqdm.tqdm(total=self.total_commits)
        for commit in self.repository_mining.traverse_commits():
            for modification in commit.modifications:
                if modification.old_path != modification.new_path and modification.old_path is not None:
                    self.old_to_new_path[modification.old_path] = modification.new_path
            pbar.update(1)
        pbar.close()

        # print(self.old_to_new_path)
        
        
        # Remove temp folder at end of execution
        atexit.register(self._cleanup)

    def retrieve_current_path(self, old_path):

        path = old_path

        while path is not None and path not in self.repo_files_path:
            if path in self.old_to_new_path:
                path = self.old_to_new_path[path]
            else:
                path = None

        return path
    
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
        # print(clone_folder)

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

    def find_lines_related_to_lines(self, start_line, end_line, path, concurrent=False):
        """ Find lines in other files that are related to line in a given file,
        based on commit history.
        """
        cwd = os.getcwd()
        os.chdir(self.repo_folder)

        modified_in_commits = self.get_commits_that_modified_line(start_line, end_line, path)
        modified_in_commits = [commit[1:-1] for commit in modified_in_commits]

        if concurrent:
            self.find_related_lines_concurrent(path, modified_in_commits)
        else:
            self.find_related_lines(path, modified_in_commits)

        os.chdir(cwd)
        
    def find_related_lines(self, path, modified_in_commits):

        related_lines = {}
        line_history = {}

        for commit in pydriller.RepositoryMining(self.repo_folder, only_commits=modified_in_commits).traverse_commits():

            for modification in tqdm.tqdm(commit.modifications):

                path = path.replace("/", "\\")
                if modification.new_path in self.repo_files_path:
                    current_path = modification.new_path
                else:
                    current_path = self.retrieve_current_path(modification.new_path)
                if current_path is not None and modification.new_path[-4:] not in self.forbidden_file_extensions and current_path != path:

                    print(modification.new_path)
                    # Get path to file to count number of lines
                    filepath = self.repo_folder + '\\' + current_path
                    if os.path.getsize(filepath):
                        with open(filepath, 'rb') as f:
                            for i, _ in enumerate(f):
                                pass
                            linenumber = i + 1
                    else:
                        linenumber = 0
                    # Split file in group of 10 lines and check of they are linked to the modified line
                    if linenumber > 0:
                        self.get_related_lines_precise(related_lines, linenumber, current_path, commit.hash, line_history)

        print(related_lines)
        self.display_related_lines(related_lines, len(modified_in_commits))

    def find_related_lines_concurrent(self, path, modified_in_commits):

        related_lines = {}
        line_history = {}
        related_files = {}

        for commit in pydriller.RepositoryMining(self.repo_folder, only_commits=modified_in_commits).traverse_commits():

            for modification in commit.modifications:

                path = path.replace("/", "\\")
                if modification.new_path in self.repo_files_path:
                    current_path = modification.new_path
                else:
                    current_path = self.retrieve_current_path(modification.new_path)
                if current_path not in related_files:
                    if current_path is not None and modification.new_path[-4:] not in self.forbidden_file_extensions and current_path != path:

                        # Get path to file to count number of lines
                        filepath = self.repo_folder + '\\' + current_path
                        if os.path.getsize(filepath):
                            with open(filepath, 'rb') as f:
                                for i, _ in enumerate(f):
                                    pass
                                linenumber = i + 1
                        else:
                            linenumber = 0
                        related_files[current_path] = linenumber

        file_lines = []
        for filepath, linenumber in related_files.items():
            for line in range(1, linenumber+1):
                file_lines.append((filepath, line))

        
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            future_to_line = {executor.submit(self.analyze_line, file_line): file_line for file_line in file_lines}

            pbar = tqdm.tqdm(total=len(file_lines))
            for future in concurrent.futures.as_completed(future_to_line):
                file_line = future_to_line[future]
                try:
                    modified_in_commits_2 = future.result()
                    modified_in_commits_2 = [commit[1:-1] for commit in modified_in_commits_2]
                    if file_line[0] not in related_lines:
                        related_lines[file_line[0]] = {file_line[1]:len(set(modified_in_commits_2).intersection(set(modified_in_commits)))}
                    else:
                        related_lines[file_line[0]][file_line[1]] = len(set(modified_in_commits_2).intersection(set(modified_in_commits)))
                except Exception as exc:
                    print(f'Error during execution : {exc}')
                pbar.update(1)
            pbar.close()


        print(related_lines)
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

        most_correlated_lines.sort(key=lambda x: (-x[2], x[1], x[0]), reverse=False)
        for (line, file_path, num_modifications_line, correlation) in most_correlated_lines:
            print(f'Line {line} of {file_path} is {correlation} correlated ({num_modifications_line} modifs)')
                    


    def get_commits_that_modified_line(self, start_line, end_line, path):

        # history = self.git_repo2.git.log('-L', f'{start_line},{end_line}:{path}').split('\n')
        history = subprocess.run(['git', 'log', '-L', f'{start_line},{end_line}:{path}', '--format=\"%H\"', '-s'], capture_output=True, encoding='utf_8').stdout.split('\n')
        modified_in_commits = [line for line in history if len(line) > 0]
    
        '''
        for line in history:
            if line[0:6] == 'commit':
                modified_in_commits.append(line[7:])
        '''
        
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


    def analyze_correlation(self, 
                        treecommit_analysis=False, 
                        commit_analysis=False, 
                        commit_lines_analysis=False, 
                        concurrent=False,
                        single_line=None):
        """ Find files/folders that are modified together (ie. in same commit).
        Update commit and TreeCommit graphs accordingly.
        """

        if treecommit_analysis or commit_analysis:
            pbar = tqdm.tqdm(total=self.total_commits)
            for commit in self.repository_mining.traverse_commits():

                modified_files = []
                for modification in commit.modifications:

                    if modification.new_path in self.repo_files_path:
                        current_path = modification.new_path
                    else:
                        current_path = self.retrieve_current_path(modification.new_path)

                    if current_path is not None:
                        modified_files.append(current_path)

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
            if concurrent:
                self.analyze_correlation_commit_lines_graph_concurent(single_line=single_line)
            else:
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
    def get_file_number_of_lines(file_path):
        
        if os.path.getsize(file_path):
            with open(file_path, 'rb') as f:
                for i, _ in enumerate(f):
                    pass
                linenumber = i + 1
        else:
            linenumber = 0

        return linenumber

    def analyze_correlation_commit_lines_graph_concurent(self, single_line=None):

        cwd = os.getcwd()
        os.chdir(self.repo_folder)

        commit_to_lines = {}

        # Print analyzing all the lines of the repo
        print('Print analyzing all the lines of the repo')
        file_lines = []
        

        if single_line:

            already_seen_files = set()
            modified_in_commits = self.get_commits_that_modified_line(single_line[1], single_line[1], single_line[0])
            modified_in_commits = [commit[1:-1] for commit in modified_in_commits]
            for commit in pydriller.RepositoryMining(self.repo_folder, only_commits=modified_in_commits).traverse_commits():

                for modification in commit.modifications:

                    path = single_line[0].replace("/", "\\")
                    if modification.new_path in self.repo_files_path:
                        current_path = modification.new_path
                    else:
                        current_path = self.retrieve_current_path(modification.new_path)

                    if current_path not in already_seen_files:
                        if current_path is not None and modification.new_path[-4:] not in self.forbidden_file_extensions:

                            # Get path to file to count number of lines
                            filepath = self.repo_folder + '\\' + current_path
                            linenumber = self.get_file_number_of_lines(filepath)
                            already_seen_files.add(current_path)

                            for i in range(1, linenumber):
                                file_lines.append((current_path, i))

        else:

            for file_path in tqdm.tqdm(self.repo_files_path):

                # Get path to file and count number of lines
                complete_file_path = self.repo_folder + '\\' + file_path
                linenumber = self.get_file_number_of_lines(complete_file_path)

                for i in range(1, linenumber):
                    file_lines.append((file_path, i))

        line_to_commits = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            future_to_line = {executor.submit(self.analyze_line, file_line): file_line for file_line in file_lines}

            pbar = tqdm.tqdm(total=len(file_lines))
            for future in concurrent.futures.as_completed(future_to_line):
                file_line = future_to_line[future]
                try:
                    modified_in_commits = future.result()
                    line_to_commits[file_line] = modified_in_commits
                except Exception as exc:
                    print(f'Error during execution : {exc}')
                pbar.update(1)
            pbar.close()

        for file_line, modified_in_commits in line_to_commits.items():

            file_path, line = file_line
            self.commit_graph_lines.add_node(f'{file_path}:{line}', number_modifications=len(modified_in_commits))

            for commit in modified_in_commits:

                if commit in commit_to_lines:
                    commit_to_lines[commit].append(f'{file_path}:{line}')
                else:
                    commit_to_lines[commit] = [f'{file_path}:{line}']


        # Building the graph
        print('\n\nBuilding the graph')
        for (_, list_lines) in tqdm.tqdm(commit_to_lines.items()):

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

        os.chdir(cwd)

    def analyze_line(self, file_line):

        file_path, line = file_line

        return self.get_commits_that_modified_line(line, line, file_path)



    def compute_correlation(self, node_name, commit_graph, method='basic', alpha=0.5):
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

            neighbors_correlation.append((neighbor, correlation, number_modifications_same_commit))
        

        neighbors_correlation = self.parse_neighbors_correlation(neighbors_correlation)

        print(f'Correlation of {node_name} (modified in {number_modifications} commits) with :')
        for i, neighbor in enumerate(neighbors_correlation):
            if i < 200:
                print(f'{neighbor[0]}:{neighbor[1]} : {neighbor[2]}% (modified {neighbor[3]} times)')
            else:
                break


    def parse_neighbors_correlation(self, neighbors_correlation):

        correlation_intervals = {}

        for neighbor, correlation, num_mod in neighbors_correlation:

            filepath, line = neighbor.split(':')
            line = int(line)

            if filepath not in correlation_intervals:
                correlation_intervals[filepath] = {(line, line):(correlation, num_mod)}
            else:
                merge_left, merge_right = False, False
                for (a, b) in correlation_intervals[filepath].keys():
                    if b == line - 1 and correlation_intervals[filepath][(a,b)][0] == correlation:
                        merge_left = True
                        merge_left_pair = (a, b)
                    if a == line + 1 and correlation_intervals[filepath][(a,b)][0] == correlation:
                        merge_right = True
                        merge_right_pair = (a, b)
                if merge_left and merge_right:
                    correlation_intervals[filepath].pop(merge_left_pair)
                    correlation_intervals[filepath].pop(merge_right_pair)
                    correlation_intervals[filepath][(merge_left_pair[0], merge_right_pair[1])] = (correlation, num_mod)
                elif merge_left:
                    correlation_intervals[filepath].pop(merge_left_pair)
                    correlation_intervals[filepath][(merge_left_pair[0], line)] = (correlation, num_mod)
                elif merge_right:
                    correlation_intervals[filepath].pop(merge_right_pair)
                    correlation_intervals[filepath][(line, merge_right_pair[1])] = (correlation, num_mod)
                else:
                    correlation_intervals[filepath][(line, line)] = (correlation, num_mod)


        neighbors_correlation_packed = []
        for filepath, linedict in correlation_intervals.items():
            for line_interval, data in linedict.items():
                neighbors_correlation_packed.append((filepath, line_interval, data[0], data[1]))
        
        neighbors_correlation_packed.sort(key=lambda x: (-x[2], x[0], x[1][0]), reverse=False)

        return neighbors_correlation_packed



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

                    if modification.new_path in self.repo_files_path:
                        current_path = modification.new_path
                    else:
                        current_path = self.retrieve_current_path(modification.new_path)

                    if current_path is not None and current_path in modified_files_dict:
                        similar_nodes += 1
                    else:
                        new_nodes.append(current_path)
                    modified_files_other_commit.append(current_path)
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
                    current_path = modification.new_path
                else:
                    current_path = self.retrieve_current_path(modification.new_path)
                
                if current_path is not None:

                    if current_path in files_commits:

                        while len(files_commits[current_path]) < current_length - 1:
                            files_commits[current_path].append(0)
                        files_commits[current_path].append(1)
                    
                    else:
                        files_commits[current_path] = [0 for _ in range(current_length-1)]
                        files_commits[current_path].append(1)

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


    def create_commits_dataframe_lines(self):

        files_commits = {}
        current_length = 0
        columns = []

        pbar = tqdm.tqdm(total=self.total_commits)
        for commit in self.repository_mining.traverse_commits():

            columns.append(commit.hash)

            pbar.update(1)
        pbar.close()


        dataframe_list = []
        index = []


        cwd = os.getcwd()
        os.chdir(self.repo_folder)

        commit_to_lines = {}

        # Print analyzing all the lines of the repo
        print('Print analyzing all the lines of the repo')
        file_lines = []
        

        for file_path in tqdm.tqdm(self.repo_files_path):

            # Get path to file and count number of lines
            complete_file_path = self.repo_folder + '\\' + file_path
            linenumber = self.get_file_number_of_lines(complete_file_path)

            for i in range(1, linenumber):
                file_lines.append((file_path, i))

        line_to_commits = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            future_to_line = {executor.submit(self.analyze_line, file_line): file_line for file_line in file_lines}

            pbar = tqdm.tqdm(total=len(file_lines))
            for future in concurrent.futures.as_completed(future_to_line):
                file_line = future_to_line[future]
                try:
                    
                    modified_in_commits = future.result()
                    modified_in_commits = [commit[1:-1] for commit in modified_in_commits]
                    index.append(f'{file_line[0]}:{file_line[1]}')
                    file_line_commits = []
                    for commit in columns:
                        if commit in modified_in_commits:
                            file_line_commits.append(1)
                        else:
                            file_line_commits.append(0)
                    dataframe_list.append(file_line_commits)
                except Exception as exc:
                    print(f'Error during execution : {exc}')
                pbar.update(1)
            pbar.close()


        os.chdir(cwd)

        return pd.DataFrame(dataframe_list, index=index, columns=columns)

    def create_commits_dataframe2(self):

        columns = ['num_commits', 
                    #'average_num_files_in_commits',
                    'number_of_neighbors',
                    'average_num_modif_with_neighbors']
        df = pd.DataFrame(columns=columns)

        for filename in self.repo_files_path:

            num_commits = self.commit_graph.nodes[filename]['number_modifications']
            total_connections = 0
            num_neighbors = 0
            for neighbor in self.commit_graph[filename]:
                num_neighbors += 1
                total_connections += self.commit_graph.edges[filename, neighbor]['number_modifications_same_commit']
            average_num_modif_with_neighbor = total_connections/num_neighbors if num_neighbors > 0 else 0
            data = [num_commits, num_neighbors, average_num_modif_with_neighbor]

            df.loc[filename] = data

        return df

       

    def dimensionality_reduction(self, df, method='tSNE'):

        if method == 'tSNE':
            tsne = sklearn.manifold.TSNE(n_components=2, perplexity=5)
            embedded_data = tsne.fit_transform(df)

        elif method == 'MCA':
        
            df.replace({0: "False", 1: "True"}, inplace = True)
            mca = prince.MCA(n_components=2)
            embedded_data = mca.fit_transform(df)

        elif method == 'NMDS':

            nmds = sklearn.manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed",
                    n_init=1)
            embedded_data = nmds.fit_transform(df)

        df_embedded = pd.DataFrame(embedded_data, index=df.index)
        return df_embedded

    def get_distance_matrix(self, df):

        dist = sklearn.neighbors.DistanceMetric.get_metric('jaccard')
        distance_matrix = dist.pairwise(df.iloc[:,:].to_numpy())
        print(f'Distance matrix : {distance_matrix}')
        print(f'{len(distance_matrix)}, {len(distance_matrix[0])}')

        distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)

        return distance_df

    def cluster_dataframe(self, df, method='HDBSCAN', distance_matrix=True, min_size=2, max_eps=None):

        if method == 'HDBSCAN':

            clusterer = hdbscan.HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.5)
            clusterer.fit(df)
        
        elif method == 'OPTICS':

            if distance_matrix:
                if max_eps is not None:
                    clusterer = sklearn.cluster.OPTICS(min_samples=min_size, metric='precomputed', n_jobs=4, max_eps=max_eps)
                else:
                    clusterer = sklearn.cluster.OPTICS(min_samples=min_size, metric='precomputed', n_jobs=4)
            else:
                clusterer = sklearn.cluster.OPTICS(min_samples=min_size, n_jobs=4)
            clusterer.fit(df)

        filenames = df.index.tolist()
        clusters = {}

        for (filename, cluster) in zip(filenames, clusterer.labels_):

            filename = filename.replace("/", "\\")

            if cluster in clusters:
                clusters[cluster].append(filename)
            else:
                clusters[cluster] = [filename]

        return clusters, clusterer.labels_

    def count_clusters_common_commits(self, df, clusters, lines=False):

        clusters_extended = {}

        for key, value in clusters.items():

            number_common_commits = 0

            for column in df:

                number_common_files_commit = 0
                for filename in value:

                    if df.loc[filename, column] == 1:

                        number_common_files_commit += 1

                if number_common_files_commit == len(value):
                    number_common_commits += 1

            if lines:
                value = self.parse_fileline(value)
            
            clusters_extended[key] = (number_common_commits, value)
            # print(f'Cluster {key}, {number_common_commits} common commits : {value}\n')

        return clusters_extended

    def display_df(self, df, clusters_labels):

        X = df.iloc[:, 0]
        Y = df.iloc[:, 1]

        _, ax = plt.subplots()
        ax.scatter(X, Y, c=clusters_labels)

        for i, txt in enumerate(clusters_labels):
            ax.annotate(txt, (X[i], Y[i]))

        # plt.scatter(X, Y, c=clusters_labels)
        plt.show()
                    
    def print_commits(self):

        for commit in self.repository_mining.traverse_commits():
            print(f'Commit : {commit.hash}')
            print(f'Parents : {commit.parents}')

    def analyze_clusters(self, clusters):

        print('Starting cluster analysis')
        cluster_to_files = {}
        file_to_cluster = {}

        for cluster_number, values in clusters.items():

            parsed_values = self.parse_fileline(values)
            cluster_to_files[cluster_number] = parsed_values

            for file_line in values:

                file_path, _ = file_line.split(":")
                
                if file_path not in file_to_cluster:
                    file_to_cluster[file_path] = [cluster_number]
                elif cluster_number not in file_to_cluster[file_path]:
                    file_to_cluster[file_path].append(cluster_number)

        '''
        # print(f'Cluster to files : {cluster_to_files}\n\n')
        print(f'File to clusters : {file_to_cluster}')

        for key, value in cluster_to_files.items():
            print(f'Cluster number {key} : {value}')
        '''

    def parse_fileline(self, files_lines):

        beautiful_files_lines = {}

        for file_line in files_lines:

            file_path, line = file_line.split(":")

            if file_path not in beautiful_files_lines:
                beautiful_files_lines[file_path] = [int(line)]
            else:
                beautiful_files_lines[file_path].append(int(line))

        for file_path, lines in beautiful_files_lines.items():

            lines.sort()
            joined_lines = []

            start = lines[0]
            end = lines[0]
            for i in range(1, len(lines)):
                if lines[i] == end + 1:
                    end += 1
                else:
                    joined_lines.append((start, end))
                    start = lines[i]
                    end = lines[i]
            joined_lines.append((start,end))
            beautiful_files_lines[file_path] = joined_lines

        return beautiful_files_lines


    def rearchitecture_clusters(self, clusters_extended):

        interesting_clusters = {}
        pool_of_lines = {}

        for cluster, value in clusters_extended.items():
            if value[0] >= 2 and len(value[1]) >= 2:
                print(f'Cluster {cluster}, num common mod {value[0]} : {value[1]}')
                interesting_clusters[cluster] = value
            else:
                for file_path in value[1].keys():
                    if file_path not in pool_of_lines:
                        pool_of_lines[file_path] = []

                    for line in value[1][file_path]:
                        pool_of_lines[file_path].append(line)



        print('\n\n')
        print(clusters_extended[0][1])
        
        for cluster_number, (num_mod, files_lines) in interesting_clusters.items():

            for file_path in files_lines.keys():
                if file_path in pool_of_lines:
                    for line in pool_of_lines[file_path]:
                        interesting_clusters[cluster_number][1][file_path].append(line)
                    
                    lines_to_be_sorted = interesting_clusters[cluster_number][1][file_path]
                    lines_to_be_sorted.sort(key=lambda x: x[0])

                    joined_lines = []

                    start = lines_to_be_sorted[0][0]
                    end = lines_to_be_sorted[0][1]
                    for i in range(1, len(lines_to_be_sorted)):
                        if lines_to_be_sorted[i][0] == end + 1:
                            end = lines_to_be_sorted[i][1]
                        else:
                            joined_lines.append((start, end))
                            start = lines_to_be_sorted[i][0]
                            end = lines_to_be_sorted[i][1]
                    joined_lines.append((start,end))
                    interesting_clusters[cluster_number][1][file_path] = joined_lines
        
        print('\n\nExtended clusters')
        for cluster, value in interesting_clusters.items():
            print(f'Cluster {cluster}, num common mod {value[0]} : {value[1]}')


        print('\n\nMerging clusters\n\n')

        initial_entropy = self.compute_entropy(self.commit_graph)
        print(f'Initial entropy : {initial_entropy}\n\n')

        for cluster, value in interesting_clusters.items():
            print(f'Entropy gain of cluster {cluster} merge')
            
            nodes = list(value[1].keys())

            new_node_name = nodes[0]
            new_commit_graph = copy.deepcopy(self.commit_graph)
            for i in range(1, len(nodes)):
                new_commit_graph = self.merge_nodes(new_node_name, nodes[i], new_commit_graph)
                new_node_name += f':{nodes[i]}'
                
            new_entropy = self.compute_entropy(new_commit_graph)
            print(f'New entropy : {new_entropy}, gain : {new_entropy - initial_entropy}\n\n')


    def compute_file_lines(self, filename):

        filepath = self.repo_folder + '\\' + filename
        if os.path.getsize(filepath):
            with open(filepath, 'rb') as f:
                for i, _ in enumerate(f):
                    pass
                lines = i + 1
        else:
            lines = 0

        return lines

    def compute_entropy(self, commit_graph):

        # Entropy computation is not perfect
        # * New size won't be the sum of old sizes exactly
        # * We have to take into account the times when node1 and node2 were modified
        # together with one of their neighbor

        entropy = 0

        for node in commit_graph.nodes:


            # Compute number of lines
            if node in self.repo_files_path:
                lines = self.compute_file_lines(node)
            else:
                files = node.split(':')
                lines = 0
                for file in files:
                    lines += self.compute_file_lines(file)

            # Compute coupling with other nodes
            coupling = 0
            for neighbor in commit_graph.neighbors(node):
                coupling += commit_graph.edges[node, neighbor]['number_modifications_same_commit']


            entropy += lines * coupling

        return entropy

    
    def merge_nodes(self, node1, node2, initial_commit_graph):

        new_commit_graph = copy.deepcopy(initial_commit_graph)

        # Etapes pour merger les nodes
        # 1. Get list of out connections with a dict
        # eg. {node3 : 5, node4 : 6}
        # 2. Get list of in connections with a dict
        # 3. Merge nodes

        # 1 and 2

        connections = {}

        for neighbor in initial_commit_graph.adj[node1]:
            if neighbor != node2:
                if neighbor not in connections:
                    connections[neighbor] = initial_commit_graph.edges[node1, neighbor]['number_modifications_same_commit']
                else:
                    connections[neighbor] += initial_commit_graph.edges[node1, neighbor]['number_modifications_same_commit']
        
        for neighbor in initial_commit_graph.adj[node2]:
            if neighbor != node1:
                if neighbor not in connections:
                    connections[neighbor] = initial_commit_graph.edges[node2, neighbor]['number_modifications_same_commit']
                else:
                    connections[neighbor] += initial_commit_graph.edges[node2, neighbor]['number_modifications_same_commit']

        new_commit_graph.remove_node(node1)
        new_commit_graph.remove_node(node2)

        new_node = f'{node1}:{node2}'
        new_commit_graph.add_node(new_node)

        for neighbor, num_mod in connections.items():
            new_commit_graph.add_edge(new_node, neighbor)
            new_commit_graph.edges[new_node, neighbor]['number_modifications_same_commit'] = num_mod

        
        return new_commit_graph



            




if __name__ == "__main__":
    
    # url = "https://github.com/apache/spark.git"
    url = "https://github.com/ishepard/pydriller.git"
    # url = "https://github.com/oilshell/oil.git"
    # url = "https://github.com/smontanari/code-forensics.git"
    # url = "https://github.com/nvbn/thefuck.git"
    
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)
    # ca.print_commits()

    
    print("Running analysis")

    
    print("Correlation analysis")
    # print(ca.get_commits_that_modified_line(10, 10, 'pydriller\\git_repository.py'))
    ca.analyze_correlation(treecommit_analysis=False, commit_analysis=True, commit_lines_analysis=False)
    # ca.save_graph(ca.commit_graph, './commit_graph.bz2')
    # start_time = time.time()
    # ca.analyze_correlation(treecommit_analysis=False, commit_analysis=False, commit_lines_analysis=True, concurrent=True)
    # print(f'{time.time() - start_time} seconds elapsed')
    # ca.save_graph(ca.commit_graph_lines, './commit_graph_lines_code_forensics.bz2')
    
    # ca.load_commit_graph_lines('./commit_graph_lines.bz2')
    # ca.load_commit_graph('./commit_graph.bz2')

    '''
    for node in ca.commit_graph_lines:
        if ca.commit_graph_lines.nodes[node]["number_modifications"] > 5:
            print(f'{node}, modifications : {ca.commit_graph_lines.nodes[node]["number_modifications"]}')
    print('\n\n')
    '''

    
    print("Clustering analysis")
    
    # df = ca.create_commits_dataframe()
    # df = ca.create_commits_dataframe_lines()
    # df.to_csv('./df_lines.csv')
    df = pd.read_csv('./df_lines.csv', index_col=0)
    # distance = ca.get_distance_matrix(df)
    # distance.to_csv('./df_distance_lines.csv')
    distance = pd.read_csv('./df_distance_lines.csv', index_col=0)
    # print(distance)

    # data = pd.read_csv('./test.tsv_files_data.tsv', sep='\t', header=None)
    # metadata = pd.read_csv('./test.tsv_files_meta.tsv', sep='\t', header=0, index_col=0)
    # data.index = metadata.index
    
    '''
    clusters, clusters_labels = ca.cluster_dataframe(
                distance,
                method='OPTICS',
                distance_matrix=True,
                min_size=20,
                max_eps=1)

    with open("./clusters.txt", "wb") as fp:
        pickle.dump(clusters, fp)
    '''
    

    with open("./clusters.txt", "rb") as fp:
        clusters = pickle.load(fp)
    # print(clusters)
    clusters_extended = ca.count_clusters_common_commits(df, clusters, lines=True)

    ca.rearchitecture_clusters(clusters_extended)

    ca.analyze_clusters(clusters)
    

    '''
    df_reduced = ca.dimensionality_reduction(distance, method='tSNE')
    ca.display_df(df_reduced, clusters_labels)
    '''

    print("Commit analysis")
    #modified_files = ca.compute_files_that_should_be_in_commit('225a29a2b904427f955756f67db6c5d572edcddc')
    '''
    
    #with open('modified_files_oil.pickle', 'wb') as handle:
    #    pickle.dump(modified_files, handle)

    
    # with open('modified_files_spark.pickle', 'rb') as handle:
    #    modified_files = pickle.load(handle)
    
    

    '''
    '''
    print(modified_files)

    related_nodes = []
    for (key, value) in modified_files.items():
        if value > 0:
            related_nodes.append((key, value))
    
    related_nodes.sort(key=lambda x: x[1], reverse=True)
    print(related_nodes[:50])
    '''
    
    

    print("Reverse correlation")
    # ca.compute_correlation('pydriller\\repository_mining.py:208', ca.commit_graph_lines, alpha=0.5)
    
    print("Line correlation")
    '''
    ca.analyze_correlation(treecommit_analysis=False,
                    commit_analysis=False,
                    commit_lines_analysis=True,
                    concurrent=True,
                    single_line=('tests/shells/test_generic.py', 18))
    ca.compute_correlation('tests\\shells\\test_generic.py:18', ca.commit_graph_lines, alpha=0.5)
    '''
    
    # ca.find_lines_related_to_lines(208, 208, 'pydriller/repository_mining.py', concurrent=True)

    print('Function correlation')
    # ca.find_lines_related_to_function('get_head', 'pydriller/git_repository.py')
    
    print("Same level correlation")
    # ca.compute_same_level_correlation('pydriller')
    
    print("Drawing results")
    # drawer = CommitGraphDrawer.CommitGraphDrawer(ca.commit_graph)
    # drawer.draw_commit_missing_files_bokeh(modified_files)
    # drawer.draw_bokeh()
    