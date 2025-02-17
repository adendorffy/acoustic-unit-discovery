from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import json

def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int)->int:
    """
    Convert timestamp (in seconds) to frame index based on sampling rate and frame size.
    """
    hop_size = frame_size_ms/1000 * sample_rate
    hop_size = np.max([hop_size, 1])
    return int((timestamp * sample_rate) / hop_size)

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

def distance(args):
    input_dir = args.input_dir / str(args.gamma) / args.model_name 
    in_paths = list(input_dir.rglob("*.npz"))
    align_paths = list(args.align_dir.rglob("*.list"))
 
    
    args.output_dir.parent.mkdir(exist_ok=True, parents=True)

    features = []
    filenames = {}
    index = 0

    for path in tqdm(in_paths, total=len(in_paths), desc="Loading Features"):
        alignment_file = [a for a in align_paths if a.stem == path.stem]

        if not alignment_file:
            continue
        else:
            alignment_file = alignment_file[0]

        with open(str(alignment_file), "r") as f:
            bounds = [get_frame_num(float(line.strip()), 16000, 20) for line in f]
        
        true_bounds = [0]  
        true_bounds.extend(bounds)  

        data = np.load(path)
        encodings = data['codes']
        boundaries = data['boundaries']

        prev_j = 1
        index = 0
        for i in range(1, len(true_bounds)):
            new_feature = []
            for j in range(prev_j, len(boundaries)):
                if true_bounds[i] < boundaries[j]:
                    continue

                if true_bounds[i] == boundaries[j]:
                    new_feature.append(int(encodings[j-1]))
                    
                elif true_bounds[i] > boundaries[j]:
                    new_feature.append(int(encodings[j-1]))
                
                prev_j = j + 1
            if new_feature:
                features.append(new_feature)
                filenames[index] = f"{path.stem}_{i}"
                index += 1

    num_features = len(features)
    print(num_features)
    dist_mat = np.zeros((num_features, num_features))

    for i in tqdm(range(num_features), desc="Calculating Distances"):
        for j in range(i+1, num_features):
            distance = edit_distance(features[i], features[j])
            dist_mat[i,j] = distance
            dist_mat[j,i] = distance
           
    print(dist_mat)

    output_path = Path(args.output_dir / args.model_name / str(args.layer_num)) / str(args.gamma)
    output_path.mkdir(parents=True, exist_ok=True)

    norm_dist_file = output_path / "norm_distance_matrix.npy"
    filenames_file = output_path / "filenames.txt"
    print(f"saving to {output_path}")

    np.save(norm_dist_file, dist_mat)

    with open(filenames_file, "w") as file:
        json.dump(filenames, file, indent=4)

class Args:
    def __init__(self, in_dir, out_dir, align_dir, model_name, layer_num, gamma):
        self.input_dir =in_dir
        self.output_dir = out_dir
        self.align_dir = align_dir
        self.model_name = model_name
        self.layer_num = layer_num
        self.gamma = gamma

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(description="Apply DTW to HuBERT features to get a distance matrix.")
    # parser.add_argument(
    #     "input_dir",
    #     help="Directory with audio encodings.",
    #     default=None,
    #     type=Path
    # )
    # parser.add_argument(
    #     "align_dir",
    #     help="Directory with alignments.",
    #     default=None,
    #     type=Path,
    # )
    # parser.add_argument(
    #     "output_dir",
    #     help="Output Directory for distance matrices.",
    #     default=None,
    #     type=Path,
    # )
    # parser.add_argument(
    #     "model_name",
    #     help="Name of the HuBERT model to use.",
    #     default="all",
    #     choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"], 
    # )

    # parser.add_argument(
    #     "layer_num",
    #     help="Layer number to extract from.",
    #     type=int,
    #     default=6,
    # )

    # args = parser.parse_args()
    gammas = [0.05, 0.1, 0.15, 0.2]
    
    for g in gammas:
        args = Args(
            in_dir=Path("codes/librispeech_subset"),
            out_dir=Path("output/librispeech_subset"),
            align_dir=Path('data/all_alignments'),
            model_name="wavlm_base",
            layer_num=8,
            gamma=g
        )

        distance(args)