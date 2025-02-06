from preprocessing import encode_audio
from prom_word_seg.wordseg.utils import Features
from prom_word_seg.extract_segments import get_features, get_word_segments
import json
import numpy as np
import os
import tqdm
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO) 

if __name__== "__main__":
    print("What would you like to do?\n1: Preprocess My audio\n2: Find the Boundaries of my word units from HuBERT features\n3: Cluster my word units from Word Unit Boundaries")
    choice = int(input("Please input the number of your choice: "))

    if choice == 1:
        args = {
            "input_dir": str(input("Input directory: ")),
            "output_dir": str(input("Output directory: ")),
            "model_name": str(input("Model name (hubert_base, hubert_large, hubert_xlarge): ")),
            "layer_number": int(input("Layer number: ")),
            "extension": str(input("Audio file extension: "))
        }
        
        encode_audio(args)
    if choice == 2:
        args = {
            "model": str(input("The model used for feature extraction: ")),
            "layer_number": int(input("Layer number: ")),
            "features_dir": Path(input("Features directory: ")),
            "sample_size": int(input("Number of features to sample (-1 for all features): ")),
            "save_out": Path(input("Directory for saving out or 0 if not save: ")),
            "load_hyperparams": str(input("Filename for hyperparameters to read in or 0 if not: ")),
            "batch_size": int(input("Batch size: "))
        }
        
        if args["load_hyperparams"] != "0":
            with open(args["load_hyperparams"]) as json_file:
                params = json.load(json_file)
                dataset_name = args["features_dir"].stem
                if args["layer_number"] != -1:
                    params = params[args["model"]][str(args["layer_number"])]
                else:
                    params = params[args["model"]]
                dist = params['distance']
                window = params['window_size']
                prom = params['prominence']
        else:
            print("Enter the hyperparameters for the segmentation algorithm: ")
            dist = str(input("Distance metric (euclidean, cosine): "))
            window = int(input("Moving average window size (int): "))
            prom = float(input("Peak detection prominence value (float): "))
        
        if args["model"] in ["mfcc", "melspec"]:
            frames_per_ms = 10
        else:
            frames_per_ms = 20

        data = Features(root_dir=args["features_dir"], model_name=args["model"], layer=args["layer_number"], num_files=args["sample_size"], frames_per_ms=frames_per_ms)

        batch_num = 0 
        batch = True

        while batch: 
            logging.info(f"Batch number: {batch_num}")

            sample, features, norm_features, index_one_frame, batch, batch_num = get_features(data, args["batch_size"], batch_num)

            # Remove features with only one frame
            sample_one_frame = [sample[i] for i in index_one_frame]
            for i in sorted(index_one_frame, reverse=True):
                del sample[i]
                del features[i]
                del norm_features[i]
            
            # Segmenting
            peaks, prominences, segmentor = get_word_segments(norm_features, distance_type=dist, prominence=prom, window_size=window)

            for i, peak in enumerate(peaks):
                if len(peak) == 0:
                    peak = np.array([features[i].shape[0] - 1]) # add a peak at the end of the file
                elif peak[-1] != features[i].shape[0] and peak[-1] != features[i].shape[0] - 1: # add at last frame (if not there or in tolerance)
                    peak = np.append(peak, features[i].shape[0] - 1)
                peaks[i] = peak

            # Add samples and peaks for features with only one frame
            np.append(sample, sample_one_frame)
            peaks.extend([np.array([1]) for _ in range(len(sample_one_frame))])

            # Optionally save the output segment boundaries
            if args["save_out"] != "0":
                root_save = args["save_out"] / args["model"] / str(args["layer_number"])
                for peak, file in tqdm.tqdm(zip(peaks, sample), desc="Saving boundaries"):
                    peak = data.get_sample_second(peak) # get peak to seconds
                    save_dir = (root_save / os.path.split(file)[-1]).with_suffix(".list")
                    save_dir.parent.mkdir(parents=True, exist_ok=True)
                    with open(save_dir, "w") as f:
                        for l in peak:
                            f.write(f"{l}\n")
    
            del sample, features, norm_features, peaks, prominences, segmentor