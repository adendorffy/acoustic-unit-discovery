import logging
import argparse
from tqdm import tqdm
from torchaudio.pipelines import HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE
import numpy as np
from pathlib import Path
import torch
import torchaudio
import textgrids

logging.basicConfig(level=logging.INFO) 

def encode_audio(args):
        logging.info(f"Starting encoding from {args["input_dir"]}")
        layer_num = int(args["layer_number"])
        bundle=HUBERT_BASE
        if args["model_name"] == "hubert_large":
            bundle=HUBERT_LARGE
        elif args["model_name"] == "hubert_xlarge":
            bundle=HUBERT_XLARGE

        model = bundle.get_model()
        model.eval()
        output_dir = Path(args["output_dir"]) / args["model_name"]

        audios = [a for a in Path(args["input_dir"]).rglob(f"*{args['extension']}")]
        for audio_path in tqdm(audios, desc="Encoding Waveforms"):
            waveform, sample_rate = torchaudio.load(str(audio_path))

            transform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=bundle.sample_rate
            )
            waveform = transform(waveform)

            with torch.no_grad():
                features, _ = model.extract_features(waveform, num_layers=layer_num)
            
            encoding = features[layer_num- 1].squeeze(0).cpu().numpy()

            rel_path = Path(audio_path).relative_to(args["input_dir"])
            npy_path = Path(output_dir) / f"{args["layer_number"]}" / rel_path.with_suffix(".npy")

            npy_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(npy_path, encoding)
        logging.info(f"Finished encoding to {output_dir}")
    
def extract_gt_boundaries(args):
    gt_list = []
    files = list(args.alignment_dir.rglob(f'**/*' + args.alignment_format))
    if args.alignment_format == '.TextGrid':
        for file in tqdm(files):
            for word in textgrids.TextGrid(file)['words']:
               gt_list.append(float(word.xmax))
            save_file = Path(*file.parts[:2], "ground_truth_boundaries", *file.parts[3:]).with_suffix(".list")

            save_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_file, "w") as f: # save the landmarks to a file
                for l in gt_list:
                    f.write(f"{l}\n")
            gt_list = []
    


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Encode audio data using HuBERT models.")
#     parser.add_argument(
#         "input_dir", 
#         help="Directory with the audio data.", 
#         default=None
#     )
#     parser.add_argument(
#         "output_dir", 
#         help="Directory where HuBERT encodings should be stored.", 
#         default=None
#     )
#     parser.add_argument(
#         "model_name", 
#         help="Name of the HuBERT model to be used in encoding.", 
#         default="hubert_base"
#     )
#     parser.add_argument(
#         "layer_number", 
#         help="Layer at which the HuBERT encoding should be extracted.", 
#         default=12
#     )
#     parser.add_argument(
#         "--extension", 
#         help="Audio files extension.", 
#         default=".flac"
#     )

#     args = parser.parse_args()
#     args_dict = {
#             "input_dir": args.input_dir,
#             "output_dir": args.output_dir,
#             "model_name": args.model_name,
#             "layer_number": args.layer_number,
#             "extension": args.extension
#         }

#     logging.debug(f"input_dir: {args.input_dir}\noutput_dir: {args.output_dir}\nmodel_name: {args.model_name}\nlayer_number: {args.layer_number}\n--extension: {args.extension}")
#     encode_audio(args_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alignment_dir", 
        help="Directory with the audio data.", 
        default=None,
        type=Path
    )
    parser.add_argument(
        "alignment_format", 
        help="Directory where HuBERT encodings should be stored.", 
        default=None
    )

    args = parser.parse_args()

    extract_gt_boundaries(args)