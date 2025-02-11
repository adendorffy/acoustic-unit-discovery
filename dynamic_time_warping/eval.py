
import argparse
from pathlib import Path

class Cluster:
    def __init__(self,id, word_dict=None, true_words=None):
        self.id = id
        self.word_dict = word_dict if word_dict is not None else []
        self.true_word_dict = true_words if true_words is not None else []
    
    def add_word_unit(self, id, index, file):
        word_unit = WordUnit(file, index, id)
        self.word_dict.append(word_unit)

    def add_true_word(self, word):
        self.true_word_dict.append(word)

    @classmethod
    def print_cluster(self, cluster):
        print(f"Cluster {cluster.id}")
        for word in cluster.word_dict:
            print(f"Word {word.id}: Index {word.index} in File {word.file}")

class WordUnit:
    def __init__(self, file, index, id):
        self.index = int(index)
        self.file = file
        self.id = int(id)

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

def cluster_purity(clusters):
    total_length = 0
    total_purity = 0

    for cluster in clusters: 
        length = len(cluster.true_word_dict)
        word_counts = {}
        for word in cluster.true_word_dict:
            word_counts[word] = word_counts.get(word, 0) + 1

        max_count = max(word_counts.values()) if word_counts else 0
        cluster_purity = max_count / length if length > 0 else 0

        total_purity += cluster_purity * length
        total_length += length
        
    return total_purity / total_length if total_length > 0 else 0


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
        default=6,
    )
    parser.add_argument(
        "dist",
        help="Distance threshold",
        type=float,
        default=-1,
    )
    args = parser.parse_args()

    indices_dict = parse_text_to_dict(args.text_indices)
    if args.dist == -1: 
        cluster_files = list(Path(f"{args.cluster_dir}/{args.model_name}_{args.layer_num}_d").rglob("*.txt"))
    else:
        cluster_files = [f"{args.cluster_dir}/{args.model_name}_{args.layer_num}_d{args.dist}.txt"]

    for cluster_file in cluster_files:
        clusters = parse_cluster_file(cluster_file)

        for clust in clusters:
            if len(clust.word_dict) > 0:
                print(f"cluster {clust.id}")
                for word_unit in clust.word_dict:
                    word = indices_dict[word_unit.file][word_unit.index]
                    clust.add_true_word(word)
                print(clust.true_word_dict)

        print(cluster_purity(clusters))

# python eval.py data/librispeech_subset_alignments/words_and_indices.txt output/dtw/clusters wavlm_base 8