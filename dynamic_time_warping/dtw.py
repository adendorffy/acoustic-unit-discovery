# Author: Danel Adendorff
# Date: 11 February 2025

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int)->int:
    """
    Convert timestamp (in seconds) to frame index based on sampling rate and frame size.
    """
    hop_size = frame_size_ms/1000 * sample_rate
    hop_size = np.max([hop_size, 1])
    return int((timestamp * sample_rate) / hop_size)


def compute_distance(i: int, j: int, norm_features: torch.tensor)->tuple:
    """
    Compute DTW distance for a given pair of features.
    """
    N = norm_features[i].shape[0]
    M = norm_features[j].shape[0]

    dist_mat = torch.cdist(norm_features[i], norm_features[j], p=2)
    cost_mat = dp(dist_mat)
    distance = round(float(cost_mat[-1, -1])*1000/1000, 4)
    norm_distance = distance/(N+M)

    return distance, norm_distance

def dp(dist_mat):
    """
    Calculate the cost matrix for a given distance matrix.
    """

    N, M = dist_mat.shape

    cost_mat = torch.zeros((N + 1, M + 1), device=device)
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf
    
    traceback_mat = torch.zeros((N, M), device=device)
    for i in range(N):
        for j in range(M):
            penalty = torch.tensor([
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            ).to(device) 
            i_penalty = torch.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty
    
    cost_mat = cost_mat[1:, 1:]
    return cost_mat


def dtw(encoding_dir, alignment_dir:Path, output_dir:Path, model_name:str, layer_num:int)->None:
    """
    Use Dynmic Time Warping to calculate the distances between different words.
    """
    file_dir = encoding_dir / model_name / str(layer_num)
    files = list(file_dir.rglob("*.npy"))

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    filenames = {}
    index = 0
    
    for file in tqdm(files, desc="Loading Features"):
        alignment_file = [a for a in list(alignment_dir.rglob("*.list")) if a.stem == file.stem]

        if not alignment_file:
            continue
        else:
            alignment_file = alignment_file[0]
                    
        with open(str(alignment_file), "r") as f:
            boundaries = [get_frame_num(float(line.strip()), 16000, 20) for line in f]
        
        encodings = torch.from_numpy(np.load(file)).to(device)

        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)
   
        for i in range(0,len(boundaries)-1):
            new_feature = encodings[boundaries[i]:boundaries[i+1], :]
            features.append(new_feature)
            filenames[index] = f"{file.stem}_{i}"
            index += 1
    
    normalized_features = []
    for feature in tqdm(features, desc="Normalising Features"):
        norm_feature = torch.nn.functional.normalize(feature, p=2, dim=1)
        normalized_features.append(norm_feature) 
    
    num_features = len(normalized_features)
    norm_distance_mat = np.zeros((num_features, num_features))
    distance_mat = np.zeros((num_features, num_features))

    for i in tqdm(range(num_features), "Calculating Distances"):
        for j in range(i+1, num_features):
            distance, norm_distance = compute_distance(i, j, normalized_features)
            distance_mat[i, j] = distance
            norm_distance_mat[i, j] = norm_distance
    
    print(norm_distance_mat)
    output_path = Path(output_dir / model_name / str(layer_num))
    output_path.mkdir(parents=True, exist_ok=True)

    norm_dist_file = output_path / "norm_distance_matrix.npy"
    dist_file = output_path / "distance_matrix.npy"
    filenames_file = output_path / "filenames.txt"
    print(f"saving to {output_path}")

    np.save(norm_dist_file, norm_distance_mat)
    np.save(dist_file, distance_mat)

    with open(filenames_file, "w") as file:
        json.dump(filenames, file, indent=4)
    

if __name__ == "__main__":

    print(f"Using device {device}")
    parser = argparse.ArgumentParser(description="Apply DTW to HuBERT features to get a distance matrix.")
    parser.add_argument(
        "encoding_dir",
        help="Directory with audio encodings.",
        default=None,
        type=Path
    )
    parser.add_argument(
        "align_dir",
        help="Directory with alignments.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "output_dir",
        help="Output Directory for distance matrices.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--model_name",
        help="Name of the HuBERT model to use.",
        default="all",
        choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"], 
    )

    parser.add_argument(
        "--layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    dtw(args.encoding_dir, args.align_dir, args.output_dir ,args.model_name, args.layer_num)

    