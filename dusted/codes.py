import torch
from sklearn.cluster import KMeans
import segment
from tqdm import tqdm
import numpy as np
from pathlib import Path

def Kmeans():
    model = KMeans(100)
    checkpoint = torch.hub.load_state_dict_from_url(
    "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt"
    )
    model.__dict__["n_features_in_"] = checkpoint["n_features_in_"]
    model.__dict__["_n_threads"] = checkpoint["_n_threads"]
    model.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return model, segment

def process_file(paths, codebook, segment, gamma):
    in_path, out_path = paths
    sequence = np.load(in_path)
    codes, boundaries = segment(sequence, codebook, gamma)
    np.savez(out_path.with_suffix(f".npz"), codes=codes, boundaries=boundaries)

    return sequence.shape, np.mean(np.diff(boundaries))

def get_frame_num(timestamp: float, sample_rate: int, frame_size_ms: int)->int:
    """
    Convert timestamp (in seconds) to frame index based on sampling rate and frame size.
    """
    hop_size = frame_size_ms/1000 * sample_rate
    hop_size = np.max([hop_size, 1])
    return int((timestamp * sample_rate) / hop_size)

def segment_dataset(args):
    kmeans, segment = Kmeans()
    input_dir = args.input_dir / args.model_name 
    in_paths = list(input_dir.rglob("*.npy"))
    out_dir = args.output_dir / str(args.gamma)
    out_paths = [out_dir / path.relative_to(args.input_dir) for path in in_paths]

    for path in out_paths:
        path.parent.mkdir(exist_ok=True, parents=True)

    results = []
    for path in tqdm(zip(in_paths, out_paths), total=len(in_paths), desc="Processing segments"):

        result = process_file(paths=path, codebook=kmeans.cluster_centers_, segment=segment.segment, gamma=args.gamma)
        
        results.append(result)


class Args:
    def __init__(self, in_dir, out_dir, align_dir, model_name, layer_num, gamma):
        self.input_dir =in_dir
        self.output_dir = out_dir
        self.align_dir = align_dir
        self.model_name = model_name
        self.layer_num = layer_num
        self.gamma = gamma


if __name__ == "__main__":
    
    gammas = [0.05, 0.1, 0.15, 0.2]
    
    for g in gammas:
        args = Args(
            in_dir=Path("features/librispeech_subset"),
            out_dir=Path("codes/librispeech_subset"),
            align_dir=Path('data/all_alignments'),
            model_name="wavlm_base",
            layer_num=8,
            gamma=g
        )

        segment_dataset(args)