import random
import numpy as np
import networkx as nx
import smt_contract_syn as scs
from sklearn.cluster import SpectralClustering
from ipdb import set_trace as st
import matplotlib.pyplot as plt
class Map:
    def __init__(self, extent, obstacles):
        self.extent = extent
        self.obstacles = obstacles
        self.nodes = self.get_nodes()
        self.node_to_index_map = self.get_node_to_index_map()
        self.adjacency_graph = self.get_adjacency_graph()

    def get_nodes(self):
        return [(i,j) for i in range(self.extent[0], self.extent[1])
                for j in range(self.extent[2], self.extent[3]) if
                (i,j) not in self.obstacles]

    def get_node_to_index_map(self):
        node_to_index_map = dict()
        for node_idx, node in enumerate(self.nodes):
            node_to_index_map[node] = node_idx
        return node_to_index_map

    def get_adjacency_graph(self):
        def get_neighbors(node):
            return [(node[0]+i*j, node[1]+(i+1)%2*j) for i in [0, 1]
                    for j in [-1, 1] if (node[0]+i*j,
                    node[1]+(i+1)%2*j) in self.nodes]
        # construct adjacency graph
        G = nx.DiGraph()
        for node in self.nodes:
            neighbors = get_neighbors(node)
            for neighbor in neighbors:
                G.add_edge(node, neighbor, weight=1)
        return G

    def get_clusters(self, N):
        numpy_func = True
        delta = 0.1 # similarity transform param
        D = nx.floyd_warshall_numpy(self.adjacency_graph, nodelist=self.nodes) # distance map
        affinity_matrix = np.array(D) # convert numpy matrix to darray for numerical consistency

        # compute similarity matrix
        similarity_matrix = np.exp(- affinity_matrix ** 2 / (2. * delta ** 2))
        clustering = SpectralClustering(n_clusters=N,
                    assign_labels="discretize", random_state=0,
                    affinity='precomputed').fit(similarity_matrix)
        node_to_cluster_dict = dict()
        for node_idx, cluster in enumerate(clustering.labels_):
            node = self.nodes[node_idx]
            node_to_cluster_dict[node] = cluster
        return node_to_cluster_dict

if __name__ == '__main__':
    random.seed(0)
    N_obstacles = 150
    N_clusters = 10
    map_length = 50
    x_extent = [0, map_length]
    y_extent = [0, map_length]
    extent = x_extent + y_extent
    obstacles = scs.create_random_obstacles(N=N_obstacles, extent=extent,
            agents=[])
    the_map = Map(extent, obstacles)
    cluster_map = the_map.get_clusters(N=N_clusters)
    for node in cluster_map:
        x, y = node
        color = 'C' + str(cluster_map[node]) + '.'
        plt.plot(x, y, color, markersize='10')
    plt.axis('scaled')
    plt.show()


