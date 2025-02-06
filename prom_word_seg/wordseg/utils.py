"""
Utility functions to sample audio features, normalize them, and get the corresponding alignments with their attributes.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: February 2024
"""

import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import os
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

class Features:
    def __init__(
        self, root_dir, model_name, layer, num_files=2000, frames_per_ms=20
    ):
        self.root_dir = root_dir
        self.model_name = model_name
        self.layer = layer
        self.num_files = num_files
        self.frames_per_ms = frames_per_ms

    def sample_features(self):
        if self.layer != -1:
            layer = str(self.layer)
            all_features = glob(os.path.join(self.root_dir, self.model_name, layer, "**/*.npy"), recursive=True)
        else:
            all_features = glob(os.path.join(self.root_dir, self.model_name, "**/*.npy"), recursive=True)

        if self.num_files == -1: 
            return all_features
        
        features_sample = np.random.choice(all_features, self.num_files, replace=False)
        return features_sample

    def load_features(self, files):
        features = []

        for file in tqdm(files, desc="Loading features"):
            encodings = torch.from_numpy(np.load(file))

            if len(encodings.shape) == 1:
                features.append(encodings.unsqueeze(0))
            else:
                features.append(encodings)
        return features

    def normalise_features(self, features):
        stacked_features = torch.cat(features, dim=0)

        scaler = StandardScaler()
        scaler.partial_fit(stacked_features) 

        normalized_features = []
        for feature in tqdm(features, desc="Normalizing Features"):
            normalized_features.append(torch.from_numpy(scaler.transform(feature))) 

        return normalized_features

    def get_frame_num(self, seconds):
        return np.round(seconds / self.frames_per_ms * 1000) 
    
    def get_sample_second(self, frame_num):
        return frame_num * self.frames_per_ms / 1000 