# Author: Danel Adendorff
# Date: 11 February 2024

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
    "wavlm_xlarge": torchaudio.pipelines.WAVLM_XLARGE,
}

def encode(input_dir:Path, output_dir:Path, model_name:str, layer_num:int, audio_ext:str)->None:
    """
    Encode audio files using specified model. Extract the encodings at the specified layer.
    Args:
        
    
    """
    audios = list(Path(input_dir.rglob(f"*{audio_ext}")))

    bundle = model_pipelines.get(model_name, torchaudio.pipelines.HUBERT_BASE)

if __name__ == "__main__":
    print(f"Using device {device}.")
