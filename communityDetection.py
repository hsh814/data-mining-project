import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse
import enum
import queue

from sklearn.metrics.cluster import normalized_mutual_info_score

from typing import List, Set, Dict, Tuple, Any

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

class NodeType(enum.Enum):
    Unknown = 0
    Core = 1
    NonMember = 2
    Hub = 3
    Outlier = 4
    

class GraphScan:
    G: nx.Graph
    epsilon: float
    core_threshold: int

    def __init__(self, G: nx.Graph, epsilon: float = 0.5, core_threshold: int = 3):
        self.count = 0
        # Update weights
        self.G = G
        self.epsilon = epsilon
        self.core_threshold = core_threshold
        
    def similarity(self, a: int, b: int) -> float:
        a_neighbors = len(self.G.neighbors(a))
        b_neighbors = len(self.G.neighbors(b))
        if a_neighbors == 0 or b_neighbors == 0:
            return 0
        return len(nx.common_neighbors(self.G, a, b)) / (np.sqrt(a_neighbors * b_neighbors))
    
    def get_neighbors(self, node: int) -> Set[int]:
        return set(self.G.neighbors(node))
    
    def epsilon_neighborhood(self, node: int) -> Set[int]:
        en = set()
        for n in self.get_neighbors(node):
            if self.similarity(node, n) >= self.epsilon:
                en.add(n)
        return en

    def is_core(self, node: int) -> int:
        return len(self.epsilon_neighborhood(node)) >= self.core_threshold
    
    def direct_reachability(self, core: int, node: int) -> Set[int]:
        # check node is neighbor of core
        return G.has_edge(core, node)
    
    

    def scan(self, k: int) -> List[List[int]]:
        clusters = list()
        for node in self.G.nodes():
            if self.is_core(node):
                # Make new cluster
                cluster = list()
                q = queue.Queue()
                for neighbor in self.epsilon_neighborhood(node):
                    q.put(neighbor)
                while not q.empty():
                    current = q.get()
                    neighbors_of_current = 
                    if current in cluster:
                        continue
                    cluster.append(current)
                    if self.is_core(current):
                        for neighbor in self.epsilon_neighborhood(current):
                            q.put(neighbor)
                
                
        return clusters

    def detect_communities(self) -> List[List[int]]:
        clusters = self.scan()
        return clusters


# Reading the Network Data
def load_network(file_path: str) -> nx.Graph:
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    for node in G.nodes():
        G.nodes[node]['type'] = NodeType.Unknown
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
    parser.add_argument("--method", "-m", type=str, help="The method to use for community detection.", default="scan")
    args = parser.parse_args()

    community_file_path = args.networkFile.replace('.dat', '.cmty')

    G = load_network(args.networkFile)

    # Detect communities using Louvain method
    if args.method == "scan":
        gs = GraphScan(G)
        detected_communities = gs.detect_communities()
    else:
        # Detect communities using Louvain method
        detected_communities = detect_communities_louvain(G)

    save_communities_to_file(detected_communities, community_file_path)
