import pydriller
import git
import tempfile
import atexit
import shutil
import os

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
            _tmp_dir : location of temp directory
        """

        self.url = url

        if self._is_remote_repository(url):
            self.repo_folder = self._clone_remote_repository(self._clone_folder(), url)
        else:
            self.repo_folder = url

        self.repository_mining = pydriller.RepositoryMining(self.repo_folder)
        self.git_repo = pydriller.GitRepository(self.repo_folder)

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


if __name__ == "__main__":

    url = "https://github.com/ishepard/pydriller.git"
    ca = CommitAnalyzer(url)

    """
    for commit in ca.repository_mining.traverse_commits():
        print("Project {}, commit {}, date {}".format(commit.project_path, commit.hash, commit.author_date))
    """
