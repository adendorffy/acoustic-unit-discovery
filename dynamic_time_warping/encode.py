# Author: Danel Adendorff
# Date: 11 February 2025

import argparse
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch
import numpy as np
import torchaudio.pipelines

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_pipelines = {
    "hubert_base": torchaudio.pipelines.HUBERT_BASE,
    "hubert_large": torchaudio.pipelines.HUBERT_LARGE,
    "hubert_xlarge": torchaudio.pipelines.HUBERT_XLARGE,
    "wavlm_base": torchaudio.pipelines.WAVLM_BASE,
    "wavlm_large": torchaudio.pipelines.WAVLM_LARGE,
    "wavlm_base_plus": torchaudio.pipelines.WAVLM_BASE_PLUS,
}

def encode(input_dir:Path, output_dir:Path, model_name:str, layer_num:int, audio_ext:str)->None:
    """
    Encode audio files using specified model. Extract the encodings at the specified layer.
    """
    audios = list(input_dir.rglob(f"*{audio_ext}"))

    bundle = model_pipelines.get(model_name, torchaudio.pipelines.HUBERT_BASE)
    model = bundle.get_model().to(device)
    model.eval()

    output_dir = Path(output_dir) / model_name / str(layer_num)

    for audio in tqdm(audios, desc="Encoding Audio Features"):
        waveform, sample_rate = torchaudio.load(str(audio))
        waveform = waveform.to(device)

        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        
        with torch.inference_mode():
            features, _ = model.extract_features(waveform, num_layers=layer_num)

        encoding = features[layer_num-1].squeeze(0).cpu().numpy()

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / f"{audio.stem}.npy"
        np.save(output_path, encoding)
    
    print(f"Stored encodings in {output_dir}.")


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
        "--model_name",
        help="Name of the model to use.",
        default="all",
        choices=["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"],  
    )

    parser.add_argument(
        "--layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--audio_ext",
        help="Extension of the audio files.",
        default=".wav",
    )

    args = parser.parse_args()

    if args.model_name == "all":
        for model in ["hubert_base", "hubert_large", "hubert_xlarge", "wavlm_base", "wavlm_large", "wavlm_xlarge"]:
            for layer_num in range(6, 10):
                encode(args.input_dir, args.output_dir, model, layer_num, args.audio_ext)
    else:
        encode(args.input_dir, args.output_dir, args.model_name, args.layer_num, args.audio_ext)