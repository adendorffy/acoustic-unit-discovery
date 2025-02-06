"""
Main script to extract features, segment the audio, and evaluate the resulting segmentation.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

from prom_word_seg.wordseg.segment import Segmentor
from prom_word_seg.wordseg.utils import Features
from tqdm import tqdm
import numpy as np
import argparse
import json
import os
from pathlib import Path

def get_features(data, batch_size, batch_num):
    sample = data.sample_features() 

    if batch_size*(batch_num+1) <= len(sample):
        sample = sample[batch_size*batch_num:batch_size*(batch_num+1)]
        batch = True

    else:
        sample = sample[batch_size*batch_num:]
        batch = False

    batch_num = batch_num + 1

    features = data.load_features(sample)
    norm_features = data.normalise_features(features) 

    index_del = []
    for i, norm_feature in enumerate(norm_features): 
        if norm_feature.shape[0] == 1:
            index_del.append(i)
    
    if len(sample) == 0:
        print('No features to segment, sampled a file with only one frame.')
        exit()
    
    return sample, features, norm_features, index_del, batch, batch_num

def get_word_segments(norm_features, distance_type="euclidean", prominence=0.6, window_size=5):
    segmentor = Segmentor(distance_type=distance_type, prominence=prominence, window_size=window_size)
    segmentor.get_distance(norm_features)

    segmentor.moving_average()

    peaks, prominences = segmentor.peak_detection()
    return peaks, prominences, segmentor 

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description="Segment speech audio.")
    parser.add_argument(
        "model",
        help="the model used for feature extraction",
        default="mfcc",
    )
    parser.add_argument(
        "layer", 
        type=int,
    )
    parser.add_argument(
        "features_dir",
        metavar="features-dir",
        help="path to the features directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of features to sample (-1 to sample all available data).",
        type=int,
    )
    parser.add_argument(
        "--save_out",
        help="option to save the output segment boundaries in ms to the specified directory.",
        default=None,
        type=Path,
    )
    parser.add_argument( 
        '--load_hyperparams',
        default=None,
        type=Path,
    )
    parser.add_argument(
        '--batch_size',
        default=25000,
        type=int,
    )

    args = parser.parse_args()

    if args.load_hyperparams is None: 
        print("Enter the hyperparameters for the segmentation algorithm: ")
        dist = str(input("Distance metric (euclidean, cosine): "))
        window = int(input("Moving average window size (int): "))
        prom = float(input("Peak detection prominence value (float): "))

    else:
        with open(args.load_hyperparams) as json_file:
            params = json.load(json_file)
            dataset_name = args.features_dir.stem
            if args.layer != -1:
                params = params[args.model][str(args.layer)]
            else:
                params = params[args.model]
            dist = params['distance']
            window = params['window_size']
            prom = params['prominence']

    if args.model in ["mfcc", "melspec"]:
        frames_per_ms = 10
    else:
        frames_per_ms = 20

    data = Features(root_dir=args.features_dir, model_name=args.model, layer=args.layer, num_files=args.sample_size, frames_per_ms=frames_per_ms)

    batch_num = 0
    batch = True

    while batch:
        print(f"Batch Number: {batch_num}")
        sample, features, norm_features, index_one_frame, batch, batch_num = get_features(data, args.batch_size, batch_num)

        # Delete all the features for which there is only one frame from the normal feature lists
        sample_one_frame = [sample[i] for i in index_one_frame]
        for i in sorted(index_one_frame, reverse=True):
            del sample[i]
            del features[i]
            del norm_features[i]
        
        peaks, prominences, segmentor = get_word_segments(norm_features, distance_type=dist, prominence=prom, window_size=window)

        # TODO understand the peaks at the last frames
        for i, peak in enumerate(peaks):
            if len(peak) == 0:
                peak = np.array([features[i].shape[0] - 1]) 

            elif peak[-1] != features[i].shape[0] and peak[-1] != features[i].shape[0] - 1: 
                peak = np.append(peak, features[i].shape[0] - 1)

            peaks[i] = peak
        
        # Add peaks for all the features where there is only one frame
        np.append(sample, sample_one_frame)
        peaks.extend([np.array([1]) for _ in range(len(sample_one_frame))])

        if args.save_out is not None:
            root_save = args.save_out / args.model / str(args.layer)

            for peak, file in tqdm.tqdm(zip(peaks, sample), desc="Saving boundaries"):
                peak = data.get_sample_second(peak) 
                save_dir = (root_save / os.path.split(file)[-1]).with_suffix(".list")
                save_dir.parent.mkdir(parents=True, exist_ok=True)

                with open(save_dir, "w") as f:
                    for l in peak:
                        f.write(f"{l}\n")

                        
        del sample, features, norm_features, peaks, prominences, segmentor