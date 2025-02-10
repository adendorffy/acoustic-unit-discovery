from pydub import AudioSegment
from pathlib import Path
import argparse

def convert_audio(input_dir, output_dir, convert_from, convert_to):
    files = list(input_dir.rglob(f"*/**/*.{convert_from}"))

    if len(files) == 0:
        return "No files in input directory."
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        save_file = output_dir / Path(file.stem).with_suffix(f".{convert_to}")
        audio = AudioSegment.from_file(file, format=convert_from)
        audio.export(save_file, format=convert_to)

    return "Conversion complete!"

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
        "convert_from",
        help="Audio extension to convert from.",
        default="m4a",
    )
    parser.add_argument(
        "convert_to",
        help="Audio extension to convert to.",
        default="wav",
    )
    args = parser.parse_args()

    message = convert_audio(args.input_dir, args.output_dir, args.convert_from, args.convert_to)
    print(message)

    # python dtw/convert_audio.py data/dtw_test_rec/originals data/dtw_test_rec/wav m4a wav
