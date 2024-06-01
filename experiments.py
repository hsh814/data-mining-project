import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse
import enum
import queue
import os
from typing import List, Set, Dict, Tuple, Any


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect communities in a network.')
    parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
    parser.add_argument("--method", "-m", type=str, help="The method to use for community detection.", default="scan")
    parser.add_argument("--out", "-o", type=str, help="The path to save the community file.", default="")
    parser.add_argument("--epsilon", "-e", type=float, help="The epsilon value for GraphScan.", default=0.25)
    parser.add_argument("--coreThreshold", "-c", type=int, help="The core threshold value for GraphScan.", default=3)
    parser.add_argument("--useModularity", "-u", action="store_true", help="Use modularity for GraphScan.")
    args = parser.parse_args()
    for i in range(1, 11):
        os.system(f"python3 communityDetection.py -n {i} -m scan -o tmp/s{i}.res")
        os.system(f"python3 communityDetection.py -n {i} -m louvain -o tmp/l{i}.res")
        os.system(f"python3 evaluation.py -n {i} -r tmp/s{i}.res")
        os.system(f"python3 evaluation.py -n {i} -r tmp/l{i}.res")
    
