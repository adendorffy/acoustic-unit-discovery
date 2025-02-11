import numpy as np
import argparse
import json
from pathlib import Path



def cluster(input_dir, cluster_dir, model_name, layer_num, distance_threshold=0.7):
    input_dir = Path(input_dir / model_name / str(layer_num))
    files = list(input_dir.rglob("*"))

    filenames = {}
    for file in files:
      
        if "norm" in file.stem:
            norm_dist_mat = np.load(file)
        
        elif "filenames" in file.stem:
            with open(file, "r", encoding="utf-8") as f:
                filenames = json.load(f)

        elif "distance" in file.stem:
            dist_mat = np.load(file)
                

    norm_dist_mat += norm_dist_mat.T

    DISTANCE_THRESHOLD = distance_threshold
    num_nodes = norm_dist_mat.shape[0]
    graph = {i: set() for i in range(num_nodes)}

    for i in range(num_nodes - 1): 
        for j in range(i + 1, num_nodes):  
            if norm_dist_mat[i, j] < DISTANCE_THRESHOLD:
                graph[i].add(j)
                graph[j].add(i)  


    clusters = []
    visited = set()

    def bfs(start_node):
        """ Traverse a cluster using BFS """
        queue = [start_node]
        cluster = []
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            queue.extend(graph[node])  

        return cluster

    for node in range(num_nodes):
        if node not in visited:
            new_cluster = bfs(node)
            clusters.append(new_cluster)

    output_file = cluster_dir / f"{model_name}_{layer_num}_d{distance_threshold}.txt"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, cluster in enumerate(clusters):
            f.write(f"\nCluster {i}:\n")  
            for j in range(len(cluster)):
                f.write(f"{cluster[j]} = {filenames[str(cluster[j])]}\n")
    print(f"Cluster info written in {output_file}")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Apply DTW to HuBERT features to get a distance matrix.")
    parser.add_argument(
        "input_dir",
        help="Directory with distance matrices.",
        default=None,
        type=Path
    )
    parser.add_argument(
        "cluster_dir",
        help="Directory with clustering output.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "model_name",
        help="Name of the model to use.",
        default="all",
        choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"],  
    )

    parser.add_argument(
        "layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )
    parser.add_argument(
        "dist",
        help="Layer number to extract from.",
        type=float,
        default=0.7,
    )
    args = parser.parse_args()

    cluster(args.input_dir, args.cluster_dir, args.model_name, args.layer_num, args.dist)

    # python cluster.py output/dtw/ output/dtw/clusters/ hubert_base 8 0.5