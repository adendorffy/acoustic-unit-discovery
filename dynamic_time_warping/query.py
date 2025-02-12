import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from encode import encode
import numpy as np
from dtw import get_frame_num, compute_distance
from cluster import cluster
from eval import parse_text_to_dict
from utils import Cluster

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_dir",
        help="Query files directory.",
        default=None,
        type=Path
    )
    parser.add_argument(
        "clusters_dir",
        help="Cluster files directory.",
        default=None,
        type=Path
    )
    parser.add_argument(
        "align_dir",
        help="Alignment files directory.",
        default=None,
        type=Path
    )
    parser.add_argument(
        "encoding_dir",
        help="Encoding files directory.",
        default=None,
        type=Path
    )
    parser.add_argument(
            "output_dir",
            help="Output files directory.",
            default=None,
            type=Path
    )
    parser.add_argument(
        "text_indices",
        help="Directory with text indices.",
        default=None,
        type=str
    )
    
    args = parser.parse_args()

    cluster_files = list(args.clusters_dir.rglob("*.txt"))
    indices_dict = parse_text_to_dict(args.text_indices)

    for k, cluster_file in enumerate(cluster_files):
        file_name = cluster_file.stem
        parts = file_name.split("_")

        model_name = parts[0] + "_" + parts[1]
        layer_num = int(parts[2])
        dist_threshold = float(f"0.{parts[3].split(".")[1]}")
        print(f"{model_name}, {layer_num}, {dist_threshold}")

        encodings_dict = encode(args.query_dir, None, model_name, layer_num, "wav")

        features = []
        file_names = {}
        index = 0
        for file in encodings_dict.keys():
            alignment_file = [a for a in list(args.align_dir.rglob("**/*.list")) if a.stem == file]
            if not alignment_file:
                continue
            else:
                alignment_file = alignment_file[0]

            with open(str(alignment_file), "r") as f:
                boundaries = [get_frame_num(float(line.strip()), 16000, 20) for line in f]
            
            new_encodings = torch.from_numpy(encodings_dict[file]).to(device)

            if len(new_encodings.shape) == 1:
                new_encodings = new_encodings.unsqueeze(0)
    
            for i in range(0,len(boundaries)-1):
                new_feature = new_encodings[boundaries[i]:boundaries[i+1], :]
                features.append(new_feature)
                file_names[index] = f"{file}_{i}"
                index += 1
            
        normalized_features = []
        for feature in features:
            norm_feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            normalized_features.append(norm_feature) 
            
        old_features = []
        encoding_path = args.encoding_dir / model_name / str(layer_num)
        encoding_files = list(encoding_path.rglob("*.npy"))
        
        for file in encoding_files:
            alignment_file = [a for a in list(args.align_dir.rglob("*.list")) if a.stem == file.stem]

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
                old_features.append(new_feature)
                file_names[index] = f"{file.stem}_{i}"
                index += 1

        normalized_old_features = []
        for feature in old_features:
            norm_feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            normalized_old_features.append(norm_feature) 
        
        output_file = args.output_dir / model_name / str(layer_num) / "norm_distance_matrix.npy"
        norm_dist_mat = np.load(str(output_file))

        extra_num_features = len(normalized_features)
        num_features = len(normalized_old_features)
        
        new_norm_dist_mat = np.pad(norm_dist_mat, pad_width=(0,extra_num_features), constant_values=0)
        all_norm_features = normalized_old_features + normalized_features

        for i in tqdm(range(num_features +1, num_features + extra_num_features), "Calculating Distances"):
            for j in range(0, i):
                distance, norm_distance = compute_distance(i, j, all_norm_features)
                new_norm_dist_mat[j, i] = norm_distance

        dist_mat_dir = Path(f"output/dtw/{model_name}/{layer_num}/d{dist_threshold}")
        dist_mat_dir.mkdir(parents=True, exist_ok=True)

        np.save(dist_mat_dir / "norm_dist_mat.npy", norm_dist_mat)
        print(new_norm_dist_mat)

        clusters = cluster(new_norm_dist_mat, file_names, model_name, layer_num, dist_threshold)
        
        appended_clusters = []
        for i, clust in enumerate(clusters):
            new_cluster = Cluster(i)
            for j in range(len(clust)):
                filename = file_names[clust[j]]
                wordunit_id = j
                file_parts = filename.split("_")
                file_name = file_parts[0]
                index = int(file_parts[1])  

                new_cluster.add_word_unit(wordunit_id, index, file_name)
            
            for word_unit in new_cluster.word_dict:
                word = indices_dict[word_unit.file][word_unit.index]
                new_cluster.add_true_word(word)
                    

            appended_clusters.append(new_cluster.true_word_dict)
            new_cluster.cluster_purity()

            if new_cluster.purity < 0.8:
                print(f"purity : {new_cluster.purity*100}%")
                print(new_cluster.true_word_dict)

                    
        print()

# python query.py data/query_set/ output/dtw/clusters/ data/all_alignments/ encodings/librispeech_subset/ output/dtw/ data/words_and_indices.txt