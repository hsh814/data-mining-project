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
    InCluster = 2
    NonMember = 3
    Hub = 4
    Outlier = 5
    

class GraphScan:
    G: nx.Graph
    epsilon: float
    core_threshold: int
    use_modularity: bool
    total_nodes: int
    total_edges: int
    total_degree: int

    def __init__(self, G: nx.Graph, epsilon: float = 0.25, core_threshold: int = 3, use_modularity: bool = False):
        self.count = 0
        self.G = G
        self.epsilon = epsilon
        self.core_threshold = core_threshold
        self.use_modularity = use_modularity
        self.total_nodes = G.number_of_nodes()
        self.total_edges = G.number_of_edges()
        self.total_degree = self.total_edges * 2
        print(self.total_degree, self.total_nodes, self.total_edges)
        print(f"[init] [epsilon {epsilon}] [core_threshold {core_threshold}] [use_modularity {use_modularity}]")
        
    def get_neighbors(self, node: int) -> Set[int]:
        return set(self.G.neighbors(node))
        
    def similarity(self, a: int, b: int) -> float:
        a_neighbors = self.get_neighbors(a)
        b_neighbors = self.get_neighbors(b)
        inter = a_neighbors.intersection(b_neighbors)
        if len(inter) == 0:
            return 0
        return (len(inter) + 2) / (np.sqrt((len(a_neighbors) + 1) * len(b_neighbors) + 1))
    
    def epsilon_neighborhood(self, node: int) -> Set[int]:
        en = set()
        for n in self.get_neighbors(node):
            if self.similarity(node, n) >= self.epsilon:
                en.add(n)
        return en

    def is_core(self, node: int) -> int:
        return len(self.epsilon_neighborhood(node)) > self.core_threshold
    
    def direct_reachability(self, core: int, node: int) -> Set[int]:
        # check node is neighbor of core
        return G.has_edge(core, node)
    
    def has_label(self, node: int) -> bool:
        return self.G.nodes[node]['type'] != NodeType.Unknown
    
    def in_cluster(self, node: int) -> bool:
        nt = self.G.nodes[node]['type']
        return nt == NodeType.InCluster or nt == NodeType.Core
    
    def is_non_member(self, node: int) -> bool:
        return self.G.nodes[node]['type'] == NodeType.NonMember
    
    def add_core(self, node: int, clusters: Dict[int, List[int]], cid: int):
        clusters[cid] = list()
        clusters[cid].append(node)
        self.G.nodes[node]['type'] = NodeType.Core
        self.G.nodes[node]['cid'] = cid
    
    def add_to_cluster(self, node: int, clusters: Dict[int, List[int]], cid: int):
        clusters[cid].append(node)
        self.G.nodes[node]['type'] = NodeType.InCluster
        self.G.nodes[node]['cid'] = cid
        
    def mark_as_non_member(self, node: int):
        self.G.nodes[node]['type'] = NodeType.NonMember
        
    def check_hub(self, node: int) -> bool:
        # Check if node is a hub
        # A hub is a node that neighbors belong to more than one cluster
        neighbors = self.get_neighbors(node)
        belong = set()
        for n in neighbors:
            if self.in_cluster(n):
                belong.add(self.G.nodes[n]['cid'])
        return len(belong) > 1
    
    def calculate_modularity(self, partition: Dict[int, int], clusters: Dict[int, List[int]], non_members: Set[int]) -> float:
        cid = len(clusters) + 1
        non_mem = 0
        for node in non_members:
            if node not in partition:
                partition[node] = cid
                cid += 1
                non_mem += 1
        res = community_louvain.modularity(partition, self.G)
        print(f"[modularity] [{res}] [cid {cid}] [clusters {len(clusters)}] [non_mem {non_mem}]")
        return res
    
    def calculate_modularity_cluster(self, cluster: List[int]) -> Tuple[int, int]:
        in_degree = 0
        tot_degree = 0
        for u in cluster:
            tot_degree += self.G.degree(u)
            for v in cluster:
                if u == v:
                    continue
                if self.G.has_edge(u, v):
                    in_degree += 1
        return in_degree, tot_degree
    
    def calculate_modularity_change(self, clusters: Dict[int, List[int]], cache: Dict[int, float], node: int, cluster: int) -> float:
        # Calculate the modularity change if node is added to cluster
        c2 = set(clusters[cluster])
        # c2.add(node)
        node_degree = self.G.degree(node)
        node_degree_in_cluster = 0
        for u in self.get_neighbors(node):
            if u in c2:
                node_degree_in_cluster += 1
        in_degree = 0
        tot_degree = 0
        for u in c2:
            tot_degree += self.G.degree(u)
            for v in c2:
                if u == v:
                    continue
                if self.G.has_edge(u, v):
                    in_degree += 1
        in_degree_new = in_degree + 2 * node_degree_in_cluster
        tot_degree_new = tot_degree + node_degree
        delta = (in_degree_new / (self.total_degree) - (tot_degree_new / (self.total_degree)) ** 2) - (in_degree / (self.total_degree) - (tot_degree / (self.total_degree)) ** 2)
        return delta

    def assign_to_clusters_by_modularity(self, non_members: Set[int], clusters: Dict[int, List[int]]):
        for hub in non_members:
            neighbors = self.get_neighbors(hub)
            belong = dict()
            for n in neighbors:
                if self.in_cluster(n):
                    cid = self.G.nodes[n]['cid']
                    if cid in belong:
                        belong[cid] += 1
                    else:
                        belong[self.G.nodes[n]['cid']] = 1
            # If no neighbors belong to any cluster, create a new cluster
            if len(belong) == 0:
                new_cid = len(clusters) + 1
                self.add_core(hub, clusters, new_cid)
            else:
                best_delta = 0.0
                best_cid = -1
                for cid in belong:
                    # Temporarily add the hub to this cluster and calculate the modularity
                    delta = self.calculate_modularity_change(clusters, hub, cid)
                    if delta > best_delta:
                        best_delta = delta
                        best_cid = cid
                if best_cid >= 0:
                    self.add_to_cluster(hub, clusters, best_cid)
                else:
                    # If no better cluster is found, create a new cluster
                    print(f"[new] [{hub}] to [{len(clusters) + 1}]")
                    new_cid = len(clusters) + 1
                    self.add_core(hub, clusters, new_cid)
    
    def assign_to_clusters(self, non_members: Set[int], clusters: Dict[int, List[int]]):
        for hub in non_members:
            neighbors = self.get_neighbors(hub)
            belong = dict()
            for n in neighbors:
                if self.in_cluster(n):
                    cid = self.G.nodes[n]['cid']
                    if cid in belong:
                        belong[cid] += 1
                    else:
                        belong[self.G.nodes[n]['cid']] = 1
            # If no neighbors belong to any cluster, create a new cluster
            if len(belong) == 0:
                new_cid = len(clusters) + 1
                self.add_core(hub, clusters, new_cid)
            # If all neighbors belong to the same cluster, assign to that cluster
            elif len(belong) == 1:
                cid = list(belong)[0]
                self.add_to_cluster(hub, clusters, cid)
            else:
                # Assign to the cluster with the most neighbors
                max_cid = -1
                max_count = -1
                for cid in belong:
                    count = belong[cid]
                    if count > max_count:
                        max_count = count
                        max_cid = cid
                self.add_to_cluster(hub, clusters, max_cid)

    def scan(self) -> Dict[int, List[int]]:
        cid = 0
        clusters: Dict[int, List[int]] = dict()
        non_members = set()
        for node in self.G.nodes():
            if self.has_label(node):
                continue
            nbrs = self.epsilon_neighborhood(node)
            # Check if node is core
            if len(nbrs) > self.core_threshold:
                cid += 1
                self.add_core(node, clusters, cid)
                q = queue.Queue()
                for nbr in nbrs:
                    q.put(nbr)
                while not q.empty():
                    current = q.get()
                    neighbors_of_current = self.epsilon_neighborhood(current)
                    neighbors_of_current.add(current)
                    for nc in neighbors_of_current:
                        if not self.has_label(nc):
                            q.put(nc)
                            self.add_to_cluster(nc, clusters, cid)
                        if self.is_non_member(nc):
                            self.add_to_cluster(nc, clusters, cid)
                            non_members.remove(nc)
            else: # node is not core
                non_members.add(node)
                self.mark_as_non_member(node)
        outliers = list()
        hubs = list()
        for n in non_members:
            if self.check_hub(n):
                hubs.append(n)
                self.G.nodes[n]['type'] = NodeType.Hub
            else:
                outliers.append(n)
                self.G.nodes[n]['type'] = NodeType.Outlier
        # Assign hubs and outliers to clusters
        print(f"[stat] [clusters {len(clusters)}] [hubs {len(hubs)}] [outliers {len(outliers)}]")
        if self.use_modularity:
            self.assign_to_clusters_by_modularity(non_members, clusters)
        else:
            self.assign_to_clusters(non_members, clusters)
        print(f"[stat] [final] [clusters {len(clusters)}]")
        return clusters

    def detect_communities(self) -> List[List[int]]:
        clusters = self.scan()
        return list(clusters.values())


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


class ParameterAdjustment:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.epsilon = 0.0
        self.core_threshold = 0
    
    def heuristic(self) -> Tuple[float, int]:
        # partition = community_louvain.best_partition(self.G)
        # community_to_nodes: Dict[int, List[int]] = {}
        # for node, community in partition.items():
        #     if community not in community_to_nodes:
        #         community_to_nodes[community] = []
        #     community_to_nodes[community].append(node)
        # find proper epsilon
        epsilon = 0.0
        gs = GraphScan(self.G.copy(), epsilon=epsilon, core_threshold=3, use_modularity=False)
        eps_dict: Dict[int, List[float]] = dict()
        for node in gs.G.nodes():
            for n in gs.get_neighbors(node):
                sim = gs.similarity(node, n)
                if node not in eps_dict:
                    eps_dict[node] = list()
                eps_dict[node].append(sim)
        # sort the similarities
        eps_list = list()
        for node, val in eps_dict.items():
            val.sort(reverse=True)
            core = len(val)
            if core >= 3:
                eps = val[2]
                eps_list.append(eps)
        eps_list.sort()
        epsilon = eps_list[int(len(eps_list) * 0.25)]
        print(f"[epsilon] [{epsilon}]")
        return epsilon, 3

    def find_parameters(self):
        max_modularity = 0.0
        best_epsilon = 0.0
        best_core_threshold = 0
        best_use_modularity = False
        for epsilon in np.arange(0.1, 0.5, 0.025):
            for core_threshold in range(2, 6):
                for use_modularity in [True, False]:
                    gs = GraphScan(self.G.copy(), epsilon=epsilon, core_threshold=core_threshold, use_modularity=use_modularity)
                    detected_communities = gs.detect_communities()
                    partition = {}
                    for cid, nodes in enumerate(detected_communities):
                        for node in nodes:
                            partition[node] = cid
                    modularity = community_louvain.modularity(partition, G)
                    if modularity > max_modularity:
                        max_modularity = modularity
                        best_epsilon = epsilon
                        best_core_threshold = core_threshold
                        best_use_modularity = use_modularity
        print(f"[best] [epsilon {best_epsilon}] [core_threshold {best_core_threshold}] [use_modularity {best_use_modularity}] [modularity {max_modularity}]")
        return best_epsilon, best_core_threshold, best_use_modularity


# Load the data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect communities in a network.')
    parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
    parser.add_argument("--method", "-m", type=str, help="The method to use for community detection.", default="scan")
    parser.add_argument("--out", "-o", type=str, help="The path to save the community file.", default="")
    parser.add_argument("--epsilon", "-e", type=float, help="The epsilon value for GraphScan.", default=0.25)
    parser.add_argument("--coreThreshold", "-c", type=int, help="The core threshold value for GraphScan.", default=3)
    parser.add_argument("--useModularity", "-u", action="store_true", help="Use modularity for GraphScan.")
    args = parser.parse_args()

    community_file_path = args.networkFile.replace('.dat', '.cmty')
    if args.out != "":
        community_file_path = args.out

    G = load_network(args.networkFile)

    # Detect communities using Louvain method
    if args.method == "scan":
        pa = ParameterAdjustment(G)
        epsilon, core_threshlod = pa.heuristic()
        gs = GraphScan(G, epsilon=epsilon, core_threshold=core_threshlod, use_modularity=True)
        detected_communities = gs.detect_communities()
    else:
        # Detect communities using Louvain method
        detected_communities = detect_communities_louvain(G)

    save_communities_to_file(detected_communities, community_file_path)
