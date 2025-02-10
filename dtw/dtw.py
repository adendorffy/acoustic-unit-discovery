import numpy as np
import scipy.spatial.distance as dist
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def dp(dist_mat):

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = torch.zeros((N + 1, M + 1), device=device)
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = torch.zeros((N, M), device=device)
    for i in range(N):
        for j in range(M):
            penalty = torch.tensor([
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]] ) # deletion (2)
            i_penalty = torch.argmin(penalty)
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

def compute_distance(i, j, norm_features):
    """Compute DTW distance for a given pair of features."""
    N = norm_features[i].shape[0]
    M = norm_features[j].shape[0]

    norm_i = norm_features[i].clone()
    norm_j = norm_features[j].clone()


    dist_mat = torch.cdist(norm_i, norm_j)
    path, cost_mat = dp(dist_mat)  
    distance = float(torch.round(cost_mat[-1, -1] * 10000) / 10000 )
    length = N + M 
    norm_distance = round(distance / length, 4)

    return i, j, distance, norm_distance

def get_frame_num(timestamp, sample_rate, frame_size_ms):
    """Convert timestamp (in seconds) to frame index based on sampling rate and frame size."""
    hop_size = frame_size_ms/1000 * sample_rate
    hop_size = np.max([hop_size, 1])
    return int((timestamp * sample_rate) / hop_size)

def dtw(input_dir, output_dir, vad_dir, model_name="hubert_base", layer_num=6):
    
    file_dir = input_dir / model_name / str(layer_num)
    files = list(file_dir.rglob("*.npy"))
    
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    associated_filenames = []
    # for file in tqdm(files, desc="Loading Features"):
    for i, file in enumerate(files):
        if i > 10:
            break

        alignment_files = [a for a in list(vad_dir.rglob("*.list")) if a.stem == file.stem]
        alignment_file = alignment_files[0]

        if not alignment_files:
            continue

        with open(str(alignment_file), "r") as f:
            boundaries = [get_frame_num(float(line.strip()), 16000, 20) for line in f]
        
        encodings = torch.from_numpy(np.load(file)).to(device) 
        if len(encodings.shape) == 1: 
            encodings = encodings.unsqueeze(0)

        for i in range(1, len(boundaries), 2):
            new_feature = encodings[boundaries[i-1]:boundaries[i], :]
            features.append(new_feature)
            associated_filenames.append(file.stem)

    
    stacked_features = torch.cat(features, dim=0).cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(stacked_features) 
    normalized_features = []

    for feature in tqdm(features, desc="Normalising Features"):
        norm_feature = torch.from_numpy(scaler.transform(feature.cpu().numpy())).to(device)
        normalized_features.append(norm_feature) 
    
    num_features = len(normalized_features)
    norm_distance_mat = np.zeros((num_features, num_features))
    distance_mat = np.zeros((num_features, num_features))
    
    for i in tqdm(range(num_features), "Calculating Distances"):
        for j in range(i+1, num_features):
            i, j, distance, norm_distance = compute_distance(i, j, normalized_features)
            distance_mat[i, j] = distance
            norm_distance_mat[i, j] = norm_distance

    
    norm_out_file = output_dir / "norm_distance_matrix.npy"
    out_file = output_dir / "distance_matrix.npy"
    print(f"saving to {norm_out_file} & {out_file}")

    np.save(norm_out_file, norm_distance_mat)
    np.save(out_file, distance_mat)


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
    
    # python dtw/dtw.py data/dtw_test_rec/encodings data/dtw_test_rec/alignments data/ --model_name=hubert_large --layer_num=6 --use_gpu