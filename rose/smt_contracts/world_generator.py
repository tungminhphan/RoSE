import random
import numpy as np
import networkx as nx
import smt_contract_syn as scs
from sklearn.cluster import SpectralClustering
from ipdb import set_trace as st
import matplotlib.pyplot as plt
import smt_contract_syn as syn

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

class Region:
    def __init__(self, nodes, region_id):
        self.nodes = nodes
        self.id = region_id
        self.boundary_nodes = self.find_boundary_nodes()
        self.box_extent = self.find_box_extent()
        self.box_obstacles = self.find_box_obstacles()

    def find_box_obstacles(self):
        box_obstacles = []
        x_min, x_max, y_min, y_max = self.box_extent
        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                if (x, y) not in self.nodes:
                    box_obstacles.append((x,y))
        return box_obstacles

    def find_boundary_nodes(self):
        boundary_nodes = []
        for node in self.nodes:
            if self.is_boundary_node(node):
                boundary_nodes.append(node)
        return boundary_nodes

    def find_box_extent(self):
        x_coords = [node[0] for node in self.boundary_nodes]
        y_coords = [node[1] for node in self.boundary_nodes]
        x_extent = [min(x_coords), max(x_coords)]
        y_extent = [min(y_coords), max(y_coords)]
        return x_extent + y_extent

    def is_boundary_node(self, node):
        """
        Check if node is not an internal node
        """
        assert node in self.nodes
        displacements = [[1,0],[0,1],[-1,0],[0,-1]]
        for displacement in displacements:
            adjacent_node = (node[0]+displacement[0], node[1]+displacement[1])
            if adjacent_node not in self.nodes:
                return True
        return False

    def get_game_constraints(self, inputs, outputs, T, step_count_max):
        assert len(inputs) == len(outputs)
        assert len(inputs+outputs) == len(set(inputs)) + len(set(outputs))
        gridders = []
        for idx, io in enumerate(zip(inputs,outputs)):
            start_pos, end_pos = io
            gridder = SMTGridder(init_state=start_pos, goal_state=end_pos,
                                 name='gridder' + str(i), color='C' + str(i))
            gridders.append(gridder)
        game = SMTGame(agents=gridders, T=T, extent=self.box_extent, obstacles=self.box_obstacles)
        game_constraints = game.get_constraints(counter_constraint=step_count_max)
        return game_constraints

    def find_contract(self, inputs, outputs, optimal=True):
        T_upperbound = 100
        step_count_max_upperbound = 200
        constraints = get_game_constraints(inputs, outputs, T, step_count_max)
        with Solver(name='z3') as solver:
            solver.add_assertion(constraints)
            if solver.solve():
                pass

class InterRegion:
    def __init__(self, list_of_regions):
        self.regions = list_of_regions
        self.region_id_to_region_map = self.get_region_id_to_region_map()
        self.connectivity_map = self.get_connectivity_map()

    def get_region_id_to_region_map(self):
        region_id_to_region_map = {region.id: region for region in
                self.regions}
        return region_id_to_region_map

    def get_connectivity_map(self):
        connectivity_map = dict()
        for from_region_idx in range(len(self.regions)):
            from_region = self.regions[from_region_idx]
            for to_region_idx in range(from_region_idx+1, len(self.regions)):
                to_region = self.regions[to_region_idx]
                for from_node in from_region.nodes:
                    for to_node in to_region.nodes:
                        if abs(from_node[0]-to_node[0]) + abs(from_node[1]-to_node[1]) <= 1:
                            if (from_region, to_region) in connectivity_map:
                                connectivity_map[(from_region, to_region)].append((from_node, to_node))
                                connectivity_map[(to_region, from_region)].append((to_node, from_node))
                            else:
                                connectivity_map[(from_region, to_region)] = [(from_node, to_node)]
                                connectivity_map[(to_region, from_region)] = [(to_node, from_node)]
        return connectivity_map

def clusters_to_regions(cluster_map):
    regions = []
    all_cluster_ids = list(set(cluster_map.values()))
    all_cluster_ids.sort()
    all_clusters = {k: [] for k in all_cluster_ids}
    for node in cluster_map:
        all_clusters[cluster_map[node]].append(node)
    for cluster_id in all_clusters:
        region = Region(all_clusters[cluster_id], cluster_id)
        regions.append(region)
    return regions

if __name__ == '__main__':
    random.seed(0)
    N_obstacles = 15
    N_clusters = 10
    map_length = 15
    x_extent = [0, map_length]
    y_extent = [0, map_length]
    extent = x_extent + y_extent
    obstacles = scs.create_random_obstacles(N=N_obstacles, extent=extent,
            agents=[])
    the_map = Map(extent, obstacles)
    cluster_map = the_map.get_clusters(N=N_clusters)
    regions = clusters_to_regions(cluster_map)
    interregion = InterRegion(regions)
    for region in regions:
        region_id = region.id
        for node in region.nodes:
            x, y = node
            color = 'C' + str(region_id)
            if node not in region.boundary_nodes:
                markersize = '10'
                symbol = 'x'
            else:
                markersize = '10'
                symbol = 'o'
            plt.plot(x, y, color+symbol, markersize=markersize)
    plt.axis('scaled')
    plt.show()
