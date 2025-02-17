# Author: Danel Adendorff
# Date: 17 February 2025

import argparse
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch
import numpy as np
import torchaudio.pipelines
from sklearn.cluster import KMeans
import segment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pipelines = {
    "hubert_base": torchaudio.pipelines.HUBERT_BASE,
    "hubert_large": torchaudio.pipelines.HUBERT_LARGE,
    "hubert_xlarge": torchaudio.pipelines.HUBERT_XLARGE,
    "wavlm_base": torchaudio.pipelines.WAVLM_BASE,
    "wavlm_large": torchaudio.pipelines.WAVLM_LARGE,
    "wavlm_base_plus": torchaudio.pipelines.WAVLM_BASE_PLUS,
}

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
    np.savez(out_path.with_suffix(".npz"), codes=codes, boundaries=boundaries)
    return sequence.shape[0], np.mean(np.diff(boundaries))

def segment_dataset(args):
    kmeans, segment = Kmeans()
    input_dir = args.input_dir / args.model_name 
    in_paths = list(input_dir.rglob("*.npy"))
    out_paths = [args.output_dir / path.relative_to(args.input_dir) for path in in_paths]

    for path in out_paths:
        path.parent.mkdir(exist_ok=True, parents=True)

    results = []
    for path in tqdm(zip(in_paths, out_paths), total=len(in_paths), desc="Processing segments"):
        result = process_file(paths=path, codebook=kmeans.cluster_centers_, segment=segment.segment, gamma=args.gamma)
        
        results.append(result)

    frames, boundary_length = zip(*results)
    print(f"Segmented {sum(frames) * 0.02 / 60 / 60:.2f} hours of audio")
    print(f"Average segment length: {np.mean(boundary_length) * 0.02:.2f} seconds")
    

def encode(input_dir:Path, output_dir:Path, model_name:str, layer_num:int, audio_ext:str)->None:
    """
    Encode audio files using specified model. Extract the encodings at the specified layer.
    """
    audios = list(input_dir.rglob(f"*{audio_ext}"))

    bundle = model_pipelines.get(model_name, torchaudio.pipelines.HUBERT_BASE)
    model = bundle.get_model().to(device)
    model.eval()

    if output_dir:
        output_dir = Path(output_dir) / model_name / str(layer_num)

    encodings = {}
    for audio in tqdm(audios, desc="Encoding Audio Features"):
        wav, sr = torchaudio.load(audio)
        wav = torchaudio.functional.resample(wav, sr, 16000).cuda()
        
        with torch.inference_mode():
            encoding, _ = model.extract_features(wav, num_layers=layer_num)

        encoding = encoding[layer_num-1].squeeze().cpu().numpy()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = Path(output_dir) / f"{audio.stem}.npy"
            np.save(output_path, encoding)
            
        else:
            encodings[audio.stem] = encoding

    if output_dir:
        return f"Stored encodings in {output_dir}."
    
    return encodings
    


if __name__ == "__main__":
    print(f"Using device {device}.")
    parser = argparse.ArgumentParser(description="Convert audio files from one format to another.")
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
        "model_name",
        help="Name of the model to use.",
        default="all",
        choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"],  
    )

    parser.add_argument(
        "layer_num",
        help="Layer number to extract from.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--gamma",
        help="Gamma hyperparameter",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--audio_ext",
        help="Extension of the audio files.",
        default=".wav",
    )

    args = parser.parse_args()

    if args.layer_num == 0:
        segment_dataset(args)
    else:
        if args.model_name == "all":
            for model in ["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"]:
                for layer_num in range(6, 10):
                    encode(args.input_dir, args.output_dir, model, layer_num, args.audio_ext)
        else:
            encode(args.input_dir, args.output_dir, args.model_name, args.layer_num, args.audio_ext)

    