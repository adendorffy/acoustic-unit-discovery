from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from pathlib import Path

def edit_distance(seq1, seq2):
    """
    Compute the edit distance between two sequences using dynamic programming.
    """
    N, M = len(seq1), len(seq2)
    dp = np.zeros((N + 1, M + 1))
    for i in range(N + 1):
        dp[i, 0] = i
    for j in range(M + 1):
        dp[0, j] = j
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[N, M] 

def calculate_distance(just_words, num_words):
    num_words = int(num_words/2)
    dist_mat = np.zeros((num_words, num_words))

    for i in tqdm(range(num_words), desc="Calculating Distances"):
        dist_mat[i, i] = 0
        js = [j for j in range(i + 1, num_words)]
        dists_i = Parallel(n_jobs=8)(
            delayed(edit_distance)(just_words[i], just_words[j]) for j in js
        )

        for j, dist in zip(js, dists_i):
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist  
        
    return dist_mat

def cluster(dist_mat):
    DISTANCE_THRESHOLD = 11
    print(f"Distance Threshold: {DISTANCE_THRESHOLD}")

    num_nodes = dist_mat.shape[0]
    graph = {i: set() for i in range(num_nodes)}

    for i in range(num_nodes - 1): 
        for j in range(i + 1, num_nodes):  
            if dist_mat[i, j] < DISTANCE_THRESHOLD:
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
    return clusters

if __name__ == "__main__":
    out_dir = Path("output/codes/librispeech_subset")
    out_paths = list(out_dir.rglob("*.npy"))
    model = "wavlm_base"
    layer = 6
    
    path = [p for p in out_paths if f"{model}_{layer}_collapsed_codes" in p.stem]
    if len(path) > 0:
        path = path[0]
    collapsed_words = np.load(path, allow_pickle=True)

    path = [p for p in out_paths if f"{model}_{layer}_codes" in p.stem]
    if len(path) > 0:
        path = path[0]
    words = np.load(path, allow_pickle=True)

    num_words = len(collapsed_words)
    collapsed_dist_mat = calculate_distance(collapsed_words, num_words)
    dist_mat = calculate_distance(words, num_words)
    print(dist_mat)
    
    clusters = cluster(dist_mat)
    for i,c in enumerate(clusters):
        print(f"Cluster {i}: {c}")
