"""
Funtions used to apply word segementation on sampled embeddings.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: March 2024
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.signal import peak_prominences 
from sklearn.preprocessing import StandardScaler 
import prom_word_seg.wordseg.utils as utils
import matplotlib as plt

class Segmentor:
    def __init__(
        self, distance_type, prominence=0.1, window_size=2
    ):
        self.distance = distance_type
        self.prominence = prominence
        self.window_size = window_size
        self.distances = []
        self.smoothed_distances = []

    def get_distance(self, embeddings):
        scaler = StandardScaler()
        
        for embedding in tqdm(embeddings, desc="Calculating Distances"):
            if self.distance == "euclidean":
                embedding_dist = np.diff(embedding, axis=0)
                euclidean_dist = np.linalg.norm(embedding_dist, axis=1)
                scaler.fit(euclidean_dist.reshape(-1, 1))
                euclidean_dist = scaler.transform(euclidean_dist.reshape(-1, 1))
                self.distances.append(euclidean_dist.reshape(-1))

            elif self.distance == "cosine":
                cosine_distances = np.array([distance.cosine(embedding[i], embedding[i + 1]) for i in range(embedding.shape[0] - 1)])
                scaler.fit(cosine_distances.reshape(-1, 1))
                cosine_distances = scaler.transform(cosine_distances.reshape(-1, 1))
                self.distances.append(cosine_distances.reshape(-1))

            else:
                raise ValueError("Distance type not supported")
        
    def moving_average(self):
        for dist in tqdm(self.distances, desc="Moving Average"):
            dist = np.pad(dist, (self.window_size // 2, self.window_size // 2), mode='edge') # TODO: understand this padding step
            box = np.ones(self.window_size) / self.window_size
            self.smoothed_distances.append(np.convolve(dist, box, 'valid'))

    def peak_detection(self):
        peaks = []
        prominences = []

        for smooth_distance in tqdm(self.smoothed_distances, desc="Peak Detection"):
            peaks_found, _ = find_peaks(smooth_distance, prominence=self.prominence)
            prominences_found = peak_prominences(smooth_distance, peaks_found)[0]
            peaks.append(peaks_found)
            prominences.append(prominences_found)
        
        return peaks, prominences
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
        choices=["w2v2_hf", "w2v2_fs", "hubert_hf", "hubert_fs", "hubert_shall"],
        default="w2v2_hf",
    )
    parser.add_argument(
        "layer",
        help="available models (wav2vec2.0: HuggingFace, fairseq, hubert: HuggingFace, fairseq, hubert:main)",
        type=int,
    )
    parser.add_argument(
        "embeddings_dir",
        metavar="embeddings-dir",
        help="path to the embeddings directory.",
        type=Path,
    )
    parser.add_argument(
        "alignments_dir",
        metavar="alignments-dir",
        help="path to the alignments directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of embeddings to sample.",
        type=int,
    )
    parser.add_argument(
        "--align_format",
        help="extension of the alignment files (defaults to .TextGrid).",
        default=".TextGrid",
        type=str,
    )

    args = parser.parse_args()

    data = utils.Features(root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, alignment_format=args.align_format, num_files=args.sample_size)

    # Embeddings
    sample = data.sample_embeddings() # sample from the feature embeddings
    # print(sample)
    embeddings = data.load_embeddings(sample) # load the sampled embeddings
    norm_embeddings = data.normalize_features(embeddings) # normalize the sampled embeddings
    print('original embedding shape', norm_embeddings[0].shape)

    # Segmenting
    segment = Segmentor(distance_type="cosine", prominence=0.65, window_size=2) # window_size is in frames (1 frame = 20ms for wav2vec2 and HuBERT)
    segment.get_distance(norm_embeddings)
    print('distance shape', segment.distances[0].shape)

    segment.moving_average() # calculate the moving average of the distances
    print('smoothed distance shape', segment.smoothed_distances[0].shape)

    print('mean distance (normalized)', np.mean(segment.smoothed_distances[0]))
    print('std distance (normalized)', np.std(segment.smoothed_distances[0]))

    peaks, prominences = segment.peak_detection() # find the peaks in the distances
    print('peaks, prominences', peaks[0], prominences[0])

    # Alignments
    alignments = data.get_alignment_paths(files=sample) # get the paths to the alignments corresponding to the sampled embeddings
    # print(alignments)
    data.set_alignments(files=alignments) # set the text, start and end attributes of the alignment files
    # print(data.alignment_data[0])

    # Plot the distances, moving average and peaks AND compare to the alignments
    fig, ax = plt.subplots()
    ax.plot(segment.distances[0], label='Distances', color='blue', alpha=0.5)
    ax.plot(segment.smoothed_distances[0], label='Smooth Distances', color='red', alpha=0.5)
    ax.scatter(peaks, segment.smoothed_distances[0][peaks], marker='x', label='Peaks', color='green')

    alignment_end_times = data.alignment_data[0].end
    alignment_end_frames = [data.get_frame_num(end_time) for end_time in alignment_end_times]
    print('Alignment end times and frames:')
    print(alignment_end_times)
    print(alignment_end_frames)
    print(data.alignment_data[0].text)

    for frame in alignment_end_frames:
        ax.axvline(x=frame, label='Ground Truth', color='black', linewidth=0.5)

    custom_ticks = alignment_end_frames
    custom_tick_labels = data.alignment_data[0].text
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_tick_labels, rotation=90, fontsize=6)

    plt.savefig('distances.png', dpi=300)    