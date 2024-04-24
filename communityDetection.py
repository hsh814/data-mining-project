import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse
import multiprocessing as mp

from sklearn.metrics.cluster import normalized_mutual_info_score

from typing import List, Set, Dict, Tuple, Any

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Reading the Network Data
def load_network(file_path: str) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

# Applying the Louvain Algorithm
def detect_communities_louvain(G: nx.Graph) -> List[List[int]]:
    partition = community_louvain.best_partition(G)
    # Convert partition dictionary to list of lists for NMI calculation
    community_to_nodes: Dict[int, List[int]] = {}
    for node, community in partition.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())

class GraphKMeans:
    G: nx.Graph
    dist: Dict[int, Tuple[Dict[int, int]]]
    cutoff: float
    count: int
    
    def compute_distance(self, n: int):
        return n, nx.single_source_dijkstra(self.G, n, cutoff=self.cutoff, weight="weight")[0]
    
    def update_distances(self, res):
        self.dist[res[0]] = res[1]
    
    def precompute_distances(self):
        pool = mp.Pool(mp.cpu_count())
        for n in self.G.nodes():
            pool.apply_async(self.compute_distance, args=(n,), callback=self.update_distances)
        pool.close()
        pool.join()
        
    def __init__(self, G: nx.Graph):
        self.count = 0
        # Update weights
        self.cutoff = max(10.0, np.log2(len(G.nodes())))
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        print("[stat] [weights updated]")
        self.G = G
        # Compute all pairs shortest path
        self.dist = dict()
        # print("[stat] [precomputing distances]")
        # self.precompute_distances()
        
    def get_distance(self, na: int, nb: int) -> int:
        if na == nb:
            return 0
        if na in self.dist and nb in self.dist[na]:
            return self.dist[na][nb]
        if nb in self.dist and na in self.dist[nb]:
            return self.dist[nb][na]
        if na not in self.dist:
            dist, paths = nx.single_source_dijkstra(self.G, na, cutoff=self.cutoff, weight="weight")
            self.dist[na] = dist
            self.count += 1
            print(f"[stat] [na {na}] [nb {nb}] [done {self.count}] [total {len(self.G.nodes())}]")
        if nb in self.dist[na]:
            return self.dist[na][nb]
        return len(self.G.nodes()) + 1

    def k_means(self, k: int) -> List[List[int]]:
        centroids = random.sample(self.G.nodes(), k)
        prev_centroids = list()
        iter = 1
        while centroids != prev_centroids:
            print(f"Iteration {iter}: {centroids} vs {prev_centroids}")
            iter += 1
            clusters = [[] for _ in range(k)]
            for node in self.G.nodes():
                distances = [self.get_distance(centroid, node) for centroid in centroids]
                closest_centroid_index = np.argmin(distances)
                clusters[closest_centroid_index].append(node)
            prev_centroids = centroids
            centroids = []
            for cluster in clusters:
                # Compute the centroid of the cluster
                centroids.append(cluster[np.argmin([np.mean([self.get_distance(node, other) for other in cluster]) for node in cluster])])
        return clusters

    def detect_communities_kmeans(self) -> List[List[int]]:
        k = int(np.sqrt(len(self.G.nodes())))
        clusters = self.k_means(k)
        return clusters


# Step 5: Save the result
def save_communities_to_file(communities: List[List[int]], file_path: str):
    # Convert the list of lists into a dictionary with community as key and nodes as values
    community_dict = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_dict[node] = community_id

    # Sort the dictionary by community key (which are the node numbers here)
    sorted_community_items = sorted(community_dict.items())

    # Write to file, now ensuring nodes are listed in the sorted order of their community keys
    with open(file_path, 'w') as f:
        for node, community_id in sorted_community_items:
            f.write(f"{node} {community_id}\n")


# Load the data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect communities in a network.')
    parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
    parser.add_argument("--method", "-m", type=str, help="The method to use for community detection.", default="kmeans")
    args = parser.parse_args()

    community_file_path = args.networkFile.replace('.dat', '.cmty')

    G = load_network(args.networkFile)

    if args.method == "kmeans":
        gkm = GraphKMeans(G)
        detected_communities = gkm.detect_communities_kmeans()
    else:
        # Detect communities using Louvain method
        detected_communities = detect_communities_louvain(G)

    save_communities_to_file(detected_communities, community_file_path)
