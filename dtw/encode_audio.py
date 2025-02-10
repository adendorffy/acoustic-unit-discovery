import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps, read_audio
from torchaudio.pipelines import HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE
import argparse
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch


def get_frame_num(seconds, frames_per_ms):
    return int(np.round(seconds / frames_per_ms * 1000))

def encode_audio(input_dir, output_dir, model_name="hubert_base", layer_num=6, audio_extension=".wav", use_vad=False):
    
    audios = list(input_dir.rglob(f"*{audio_extension}"))

    bundle = HUBERT_BASE
    if model_name == "hubert_large":
        bundle = HUBERT_LARGE
    elif model_name == "hubert_xlarge":
        bundle = HUBERT_XLARGE
    model = bundle.get_model()
    model.eval()

    vad_model = load_silero_vad()
    output_dir = Path(output_dir) / model_name / str(layer_num)

    for audio in tqdm(audios, desc="Encoding Audio Features"):
        waveform, sample_rate = torchaudio.load(str(audio))
        transform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=bundle.sample_rate
        )
        waveform = transform(waveform)

        with torch.no_grad():
            features, _ = model.extract_features(waveform, num_layers=layer_num)
        
        encoding = features[layer_num- 1].squeeze(0).cpu().numpy()

        rel_path = Path(audio).relative_to(Path(input_dir))
        npy_path = Path(output_dir) / rel_path.with_suffix(".npy")

        npy_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(npy_path, encoding)

        if use_vad:

            vad_wav = read_audio(str(audio))
            speech_timestamps = get_speech_timestamps(vad_wav, vad_model, return_seconds=True)
            alignments_dir = Path(*input_dir.parts[:2]) / "alignments" 
            alignments_dir.mkdir(parents=True, exist_ok=True)
            list_filename = alignments_dir / f"{audio.stem}.list"  

            with open(list_filename, 'w') as f: 
                for timestamp in speech_timestamps:
                  
                    f.write(f"{timestamp['start']}\n")  
                    f.write(f"{timestamp['end']}\n")  

    if use_vad:
        print(f"Stored alignments in {str(alignments_dir)}")
            


if __name__ == "__main__":
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
        help="Name of the HuBERT model to use.",
        default="hubert_base",
        choices=["hubert_base", "hubert_large", "hubert_xlarge"],  # Restrict valid values
    )

    parser.add_argument(
        "--layer_num",
        help="Layer number to extract from.",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--audio_extension",
        help="Extension of the audio files.",
        default=".wav",
    )

    parser.add_argument(
        "--use_vad",
        help="Use VAD to get alignments.",
        action="store_true", 
    )

    args = parser.parse_args()

    encode_audio(args.input_dir, args.output_dir, args.model_name, args.layer_num, args.audio_extension, args.use_vad)

# python dtw/encode_audio.py data/dtw_test_rec/wav data/dtw_test_rec/encodings --model_name=hubert_base --layer_num=10 --use_vad