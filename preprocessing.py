import logging
import argparse
from tqdm import tqdm
from torchaudio.pipelines import HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE
import numpy as np
from pathlib import Path
import torch
import torchaudio

logging.basicConfig(level=logging.INFO) 

def encode_audio(args):
        logging.info(f"Starting encoding from {args.input_dir}")
        layer_num = int(args.layer_number)
        bundle=HUBERT_BASE
        if args.model_name == "hubert_large":
            bundle=HUBERT_LARGE
        elif args.model_name == "hubert_xlarge":
            bundle=HUBERT_XLARGE

        model = bundle.get_model()
        model.eval()
        output_dir = Path(args.output_dir) / args.model_name

        audios = [a for a in Path(args.input_dir).rglob(f"*{args.extension}")]
        for audio_path in tqdm(audios):
            waveform, sample_rate = torchaudio.load(str(audio_path))

            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=bundle.sample_rate
            )
            waveform = transform(waveform)

            with torch.no_grad():
                features, _ = model.extract_features(waveform, num_layers=layer_num)
            
            encoding = features[layer_num- 1].squeeze(0).cpu().numpy()

            rel_path = Path(audio_path).relative_to(args.input_dir)
            npy_path = Path(output_dir) / f"{args.layer_number}" / rel_path.with_suffix(".npy")

            npy_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(npy_path, encoding)
        logging.info(f"Finished encoding to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode audio data using HuBERT models.")
    parser.add_argument(
        "input_dir", 
        help="Directory with the audio data.", 
        default=None
    )
    parser.add_argument(
        "output_dir", 
        help="Directory where HuBERT encodings should be stored.", 
        default=None
    )
    parser.add_argument(
        "model_name", 
        help="Name of the HuBERT model to be used in encoding.", 
        default="hubert_base"
    )
    parser.add_argument(
        "layer_number", 
        help="Layer at which the HuBERT encoding should be extracted.", 
        default=12
    )
    parser.add_argument(
        "--extension", 
        help="Audio files extension.", 
        default=".flac"
    )

    args = parser.parse_args()

    logging.debug(f"input_dir: {args.input_dir}\noutput_dir: {args.output_dir}\nmodel_name: {args.model_name}\nlayer_number: {args.layer_number}\n--extension: {args.extension}")
    encode_audio(args)