import networkx as nx

class TreeGraph:

    def __init__(self, name, is_file):

        self.name = name
        self.graph = nx.Graph()
        self.kids = {}
        self.is_file = is_file


    def add_children(self, path):

        if path[0] in self.graph:
            if len(path) > 1:
                self.kids[path[0]].add_children(path[1:])
        
        else:
            self.graph.add_node(path[0], number_modifications=0)
            if len(path) > 1:
                self.kids[path[0]] = TreeGraph(path[0], False)
                self.kids[path[0]].add_children(path[1:])
            else:
                self.kids[path[0]] = TreeGraph(path[0], True)

    def add_edge(self, path, node1, node2):

        if len(path) > 0:
            self.graph.nodes[path[0]]['number_modifications'] += 1
            self.kids[path[0]].add_edge(path[1:], node1, node2)
        else:
            if node1 in self.graph.nodes and node2 in self.graph.nodes:

                self.graph.nodes[node1]['number_modifications'] += 1
                self.graph.nodes[node2]['number_modifications'] += 1
                
                if self.graph.has_edge(node1, node2):
                    self.graph.edges[node1, node2]['number_modifications_same_commit'] += 1
                else:
                    self.graph.add_edge(node1, node2, number_modifications_same_commit=1)

    def print(self):
        print(self.name)
        for kid in self.kids:
            self.kids[kid].print()




    