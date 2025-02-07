import numpy as np
import scipy.spatial.distance as dist
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler

def dp(dist_mat):

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

def dtw(input_dir, output_dir, vad_dir, model_name="hubert_base", layer_num=6):
    file_dir = input_dir / model_name / str(layer_num)
    files = list(file_dir.rglob("*.npy"))
    
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for file in tqdm(files, desc="Loading Features"):

        alignment_files = [a for a in list(vad_dir.rglob("*.list")) if a.stem == file.stem]
        alignment_file = alignment_files[0]

        with open(str(alignment_file), "r") as f:
            boundaries = []
            for line in f:
                boundaries.append(int(line))

        encodings = torch.from_numpy(np.load(file))
        if len(encodings.shape) == 1: 
            encodings = encodings.unsqueeze(0)
        
        for i in range(1, len(boundaries)):
            new_feature = encodings[boundaries[i-1]:boundaries[i], :]
            features.append(new_feature)

        if len(boundaries) == 1:
            features.append(encodings[boundaries[0]:, :]) 
    
    stacked_features = torch.cat(features, dim=0)
    scaler = StandardScaler()
    scaler.fit(stacked_features) # (n_samples, n_features)
    normalized_features = []
    flat_norm_features = []

    for feature in tqdm(features, desc="Normalising Features"):
        norm_feature = torch.from_numpy(scaler.transform(feature))
        normalized_features.append(norm_feature) 
        flat_norm_feature = norm_feature.squeeze().numpy()
        flat_norm_features.append(flat_norm_feature)
    
    print()

    norm_distance_mat = np.zeros((len(flat_norm_features), len(flat_norm_features)))
    distance_mat = np.zeros((len(flat_norm_features), len(flat_norm_features)))

    for i in range(len(flat_norm_features)):
        for j in range(i+1, len(flat_norm_features)):
            dist_mat = dist.cdist(flat_norm_features[i], flat_norm_features[j], "cosine")
            path, cost_mat = dp(dist_mat)
            distance_mat[i, j] = round(cost_mat[-1, -1], 4)

            length = flat_norm_features[i].shape[1] + flat_norm_features[j].shape[1]
            # print(f"Normalized Distance between {files[i].stem} and {files[j].stem}: {round(cost_mat[-1, -1]/length, 4)}")
            print(f"Distance between {files[i].stem} and {files[j].stem}: {round(cost_mat[-1, -1], 4)}")
            norm_distance_mat[i, j] = round(cost_mat[-1, -1]/length, 4)
        print()
    
    print(norm_distance_mat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply DTW to HuBERT features to get a distance matrix.")
    parser.add_argument(
            "input_dir",
            help="Input directory.",
            default=None,
            type=Path
    )
    parser.add_argument(
        "output_dir",
        help="Output directory.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "vad_dir",
        help="Directory with VAD alignments.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--model_name",
        help="Name of the HuBERT model to use.",
        default="hubert_base",
        choices=["hubert_base", "hubert_large", "hubert_xlarge"],  
    )

    parser.add_argument(
        "--layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    dtw(args.input_dir, args.output_dir, args.vad_dir ,args.model_name, args.layer_num)
    
    # python dtw/dtw.py data/dtw_test_rec/encodings data/dtw_test_rec/alignments data/ --model_name=hubert_large --layer_num=6