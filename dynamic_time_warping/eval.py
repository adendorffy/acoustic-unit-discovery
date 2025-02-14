
import argparse
from pathlib import Path
from utils import Cluster
import csv
import os

def parse_text_to_dict(file):
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_dict = {}
    current_id = None
    word_dict = {}

    for line in lines: 
        line = line.strip()

        if not line: 
            continue
        
        if line.endswith(":") and not line.split(":")[0].isdigit():
            if current_id is not None:
                data_dict[current_id] = word_dict
            
            current_id = line[:-1]
            word_dict = {}
        else:
            parts = line.split(": ")
            if len(parts) == 2:
                index, word = parts
                word_dict[int(index)] = word.strip()
            else:
                parts = parts[0].split(":")
                index = parts[0]
                word_dict[int(index)] = " "
            
            if current_id is not None:
                data_dict[current_id] = word_dict
        
    return data_dict



def parse_cluster_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    clusters = []
    cluster_id = None
    current_cluster = None

    for line in lines:
        line = line.strip()

        if line.startswith("Cluster "):
            if current_cluster:  
                clusters.append(current_cluster)
            cluster_id = int(line.split(" ")[1].replace(":", ""))
            current_cluster = Cluster(cluster_id)
            continue

        if "=" in line:
            parts = line.split("=")
            wordunit_id = int(parts[0].strip())
            file_parts = parts[1].strip().split("_")
            file_name = file_parts[0]
            index = int(file_parts[1])  
            
            if current_cluster:
                current_cluster.add_word_unit(wordunit_id, index, file_name)

    if current_cluster:
        clusters.append(current_cluster)

    return clusters




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply DTW to HuBERT features to get a distance matrix.")
    parser.add_argument(
        "text_indices",
        help="Directory with text indices.",
        default=None,
        type=str
    )
    parser.add_argument(
        "cluster_dir",
        help="Directory with clustering output.",
        default=None,
        type=str,
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
    )

    parser.add_argument(
        "results_file",
        help="File with results (.csv)",
        type=Path,        
    )
    parser.add_argument(
        "--write_header",
        help="Should header be written to results file.",
        default=False, 
        action='store_true',         
    )
    args = parser.parse_args()

    indices_dict = parse_text_to_dict(args.text_indices)
    
    cluster_dir = Path(f"{args.cluster_dir}/")
    cluster_files = list(cluster_dir.rglob(f"{args.model_name}_{args.layer_num}_d*.txt"))
    if len(cluster_files) == 0:
        raise ReferenceError("Cluster Directory empty")

    args.results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.results_file, "a", encoding="utf-8", newline="") as f:
        
        writer = csv.writer(f)
        if args.write_header:
                writer.writerow(["Model Name", "Layer Number", "Distance Threshold", "Average Total Purity", "Duplicate Clusters", "Total Clusters"])

        for cluster_file in cluster_files:
            cluster_file_stem = cluster_file.stem
            print(cluster_file_stem) 

            parts = cluster_file_stem.split("_")
            parts = parts[-1].split(".")
            dist = float(f"0.{parts[1]}")

            clusters = parse_cluster_file(cluster_file)
            
            total_length = 0
            total_purity = 0
            clusters_array = []

            for clust in clusters:
                
                for word_unit in clust.word_dict:
                    word = indices_dict[word_unit.file][word_unit.index]
                    clust.add_true_word(word)

                if len(clust.word_dict) > 0:
                    clust.cluster_purity()

                    total_length += clust.length
                    total_purity += clust.purity * clust.length

                print(clust.true_word_dict)
                clusters_array.append(clust.true_word_dict)
                
            total_purity = total_purity / total_length if total_length > 0 else 1.0 
            num_clusters = len(clusters)
            print(f"total purity: {total_purity}")
            
            condensed_clusters = []
            for c in clusters_array:
                condensed_clusters.append(list(set(c)))

            num_duplicate_clusters = Cluster.duplicate_clusters(condensed_clusters)
            print(f"Duplicate clusters: {num_duplicate_clusters}\n")

            # writer.writerow([
            #     args.model_name,
            #     args.layer_num,
            #     dist,
            #     f"{total_purity * 100:.2f}%",  
            #     num_duplicate_clusters,
            #     num_clusters
            # ])
            # print(condensed_clusters)

# python eval.py data/librispeech_subset_alignments/words_and_indices.txt output/dtw/clusters wavlm_base 8 