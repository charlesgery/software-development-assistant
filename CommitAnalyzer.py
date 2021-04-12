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
        self.commit_graph.add_nodes_from([(filename, {'number_modifications':0}) for filename in self.repo_files])

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
            pairs_of_modified_files = []
            for i in range(len(modified_files)):
                for j in range(i+1, len(modified_files)):
                    pairs_of_modified_files.append((modified_files[i], modified_files[j]))

            for edge in pairs_of_modified_files:

                if self.commit_graph.has_edge(edge[0], edge[1]):
                    self.commit_graph.edges[edge[0], edge[1]]['number_modifications_same_commit'] += 1
                else:
                    self.commit_graph.add_edge(edge[0], edge[1], number_modifications_same_commit=0)

            pbar.update(1)
        pbar.close()




if __name__ == "__main__":

    url = "https://github.com/ishepard/pydriller.git"

    print("Init CommitAnalyzer")
    ca = CommitAnalyzer(url)

    print("Running analysis")
    ca.analyze_correlation()

    
    print("Drawing results")

    # Layout
    pos = nx.spring_layout(ca.commit_graph, weight='number_modifications_same_commit')

    # Edge Width
    edges = ca.commit_graph.edges()
    number_time_modified_together = [ca.commit_graph[u][v]['number_modifications_same_commit'] for u,v in edges]
    max_number_time_modified_together = max(number_time_modified_together)
    width = [num / max_number_time_modified_together for num in number_time_modified_together]

    # Draw
    nx.draw(ca.commit_graph, pos=pos, with_labels=True, width=width)
    plt.show()
    

    
