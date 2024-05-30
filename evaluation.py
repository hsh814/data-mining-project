# Evaluate NMI between detected communities and ground truth
from sklearn.metrics.cluster import normalized_mutual_info_score
import argparse

from typing import List, Set, Dict, Tuple, Any

# Reading the Ground-Truth Community Data
def load_ground_truth(file_path: str) -> List[List[int]]:
    node_to_community = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node, community = int(parts[0]), int(parts[1])
                node_to_community[node] = community
    # Convert to list of lists for compatibility with NMI calculation
    community_to_nodes = {}
    for node, community in node_to_community.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    print(f"Found {len(community_to_nodes)} communities / {len(node_to_community)} nodes in {file_path}")
    return list(community_to_nodes.values())


# Calculating NMI Score
def calculate_nmi(true_communities: List[List[int]], detected_communities: List[List[int]]) -> float:
    # Flatten the lists and create label vectors
    true_labels = {}
    for i, community in enumerate(true_communities):
        for node in community:
            true_labels[node] = i
    detected_labels = {}
    for i, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node] = i

    # Ensure the labels are in the same order for both true and detected
    nodes = sorted(set(true_labels) | set(detected_labels))
    true_labels_vector = [true_labels[node] for node in nodes]
    detected_labels_vector = [detected_labels.get(node, -1) for node in nodes]

    return normalized_mutual_info_score(true_labels_vector, detected_labels_vector)


def eval(network_file_path: str, community_file_path: str) -> float:
    # Replace or append file extensions as necessary to construct paths
    # community_file_path = network_file_path.replace('.dat', '.cmty')
    ground_truth_file_path = network_file_path.replace('.dat', '-c.dat')

    detected_communities = load_ground_truth(community_file_path)
    true_communities = load_ground_truth(ground_truth_file_path)
    nmi_score = calculate_nmi(true_communities, detected_communities)
    return nmi_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect communities in a network.')
    parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
    parser.add_argument("--result", "-r", type=str, help="The path to save the result file.", default="")
    args = parser.parse_args()

    network_file = args.networkFile
    if network_file in [str(i) for i in range(1, 11)]:
        network_file = f"./data/TC1-{network_file}/1-{network_file}.dat"

    network_file_path = network_file
    community_file_path = network_file_path.replace('.dat', '.cmty')
    if args.result != "":
        community_file_path = args.result
    
    print(network_file_path, eval(network_file_path, community_file_path))
