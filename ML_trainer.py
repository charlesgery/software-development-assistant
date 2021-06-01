import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec

import CommitAnalyzer

class ML_trainer:

    def __init__(self, graph=None) -> None:
        
        self.graph = graph
        self.node2vec = Node2Vec(graph, weight_key='number_modifications_same_commit',
                dimensions=64,
                walk_length=30,
                num_walks=200,
                workers=1)
        self.model = None

    def fit(self):

        self.model = self.node2vec.fit(window=10, min_count=1, batch_words=4)

    def save(self):

        self.model.save('./node2vec_model')

    def load(self):

        self.model = Word2Vec.load('./node2vec_model')


    def display_results(self):

        print(self.model.wv.most_similar('pydriller\\repository.py'))



if __name__ == "__main__":
    
    url = "https://github.com/ishepard/pydriller.git"
    
    print("Init CommitAnalyzer")
    ca = CommitAnalyzer.CommitAnalyzer(url)
    # ca.print_commits()
    
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

    print("ML analysis")

    ml_analyzer = ML_trainer(ca.commit_graph)
    ml_analyzer.fit()
    ml_analyzer.save()

    ml_analyzer.load()
    ml_analyzer.display_results()
    