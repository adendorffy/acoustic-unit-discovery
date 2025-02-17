# Author: Danel Adendorff
# Date: 11 February 2025

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import json
import itertools
import os
from cython_dtw import _dtw
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor

dtw_cost_func = _dtw.multivariate_dtw_cost_cosine

def generate_pairs_in_batches(num_features, batch_size=100):
    for i in range(num_features):
        for j in range(i + 1, num_features):
            yield (i, j)  
            if (j - i) % batch_size == 0:
                yield None  


def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int)->int:
    """
    Convert timestamp (in seconds) to frame index based on sampling rate and frame size.
    """
    hop_size = frame_size_ms/1000 * sample_rate
    hop_size = np.max([hop_size, 1])
    return int((timestamp * sample_rate) / hop_size)

def process_pair(pair, normalized_features):
    i, j = pair
    norm_distance = dtw_sweep_min(normalized_features[i], normalized_features[j])
    return i, j, norm_distance


def dtw_sweep_min(query_seq, search_seq, n_step=3):
    """
    Return the minimum DTW cost as `query_seq` is swept across `search_seq`.

    Step size can be specified with `n_step`.
    """
    i_start = 0
    n_query = query_seq.shape[0]
    n_search = search_seq.shape[0]
    min_cost = np.inf

    while i_start <= n_search - n_query or i_start == 0:
        cost = dtw_cost_func(
            query_seq, search_seq[i_start:i_start + n_query], True
        )
        i_start += n_step
        if cost < min_cost:
            min_cost = cost

    return min_cost

def dtw(encoding_dir, alignment_dir:Path, output_dir:Path, model_name:str, layer_num:int, batch_size:int)->None:
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
        
        encodings = np.load(file)

        if len(encodings.shape) == 1:
            encodings = encodings.unsqueeze(0)
   
        for i in range(0,len(boundaries)-1):
            new_feature = encodings[boundaries[i]:boundaries[i+1], :]
            if len(new_feature) == 0:
                continue
            features.append(new_feature)
            filenames[index] = f"{file.stem}_{i}"
            index += 1
    
    tensor_features = [torch.from_numpy(f) for f in features]
    stacked_features = torch.cat(tensor_features, dim=0)
    normalized_features = []

    scaler = StandardScaler()
    scaler.fit(stacked_features) 
    normalized_features = []
    for feature in tqdm(features, desc="Normalizing Features"):
        normalized_features.append(torch.from_numpy(scaler.transform(feature))) 
    
    num_features = len(normalized_features)
    norm_distance_mat = np.zeros((num_features, num_features))
    normalized_features = [f.cpu().numpy().astype(np.float64) for f in normalized_features]

    print(len(normalized_features))
    num_pairs = (num_features * (num_features - 1)) // 2
    current_batch = []
    for pair in tqdm(generate_pairs_in_batches(num_features, 1000), total=num_pairs, desc="Calculating Distances"):
        if pair is None:    
            for i, j in current_batch:  
                i, j, norm_distance = process_pair((i, j), normalized_features)
                norm_distance_mat[i, j] = norm_distance
                norm_distance_mat[j, i] = norm_distance  
            
            current_batch = []  
        else:
            current_batch.append(pair)

    if current_batch:
        for i, j in current_batch:
            norm_distance = process_pair((i, j), normalized_features)
            norm_distance_mat[i, j] = norm_distance
            norm_distance_mat[j, i] = norm_distance  

    print(norm_distance_mat)
    print(norm_distance_mat.shape)
    # output_path = Path(output_dir / model_name / str(layer_num))
    # output_path.mkdir(parents=True, exist_ok=True)

    # norm_dist_file = output_path / "norm_distance_matrix.npy"
    # filenames_file = output_path / "filenames.txt"
    # print(f"saving to {output_path}")

    # np.save(norm_dist_file, norm_distance_mat)

    # with open(filenames_file, "w") as file:
    #     json.dump(filenames, file, indent=4)
    

if __name__ == "__main__":

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
        "model_name",
        help="Name of the HuBERT model to use.",
        default="all",
        choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"], 
    )

    parser.add_argument(
        "layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    dtw(args.encoding_dir, args.align_dir, args.output_dir ,args.model_name, args.layer_num, batch_size=100)

#  python dtw.py encodings/librispeech_subset/ data/all_alignments/ output/dtw/ wavlm_base 8
#  python dtw.py encodings/librispeech-wav/ data/all_alignments/ full_output/cython_dtw/ wavlm_base 8