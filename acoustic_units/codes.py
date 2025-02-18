from features import Features
from pathlib import Path
import torch
from sklearn.cluster import KMeans
import numpy as np
from segment import segment
from tqdm import tqdm
from itertools import groupby

def kmeans_model(url):
    model = KMeans(100)
    checkpoint = torch.hub.load_state_dict_from_url(url)

    model.__dict__["n_features_in_"] = 20
    model.__dict__["_n_threads"] = checkpoint["_n_threads"]
    model.__dict__["cluster_centers_"] = checkpoint["cluster_centers_"].numpy()
    return model

def  apply_kmeans(kmeans_model, encoding):
    # C = cluster centers matrix
    C_np = kmeans_model.cluster_centers_.transpose()
    Cnorm_np = (C_np ** 2).sum(0, keepdims=True)

    C = torch.from_numpy(C_np)
    Cnorm = torch.from_numpy(Cnorm_np)

    if torch.cuda.is_available():
        C = C.cuda()
        Cnorm = Cnorm.cuda()
    
    if isinstance(encoding, torch.Tensor):
        dist = (
            encoding.pow(2).sum(1, keepdims=True)-2*torch.matmul(encoding, C)+Cnorm
        )
    else:
        dist = (
            (encoding**2).sum(1, keepdims=True)-2*np.matmul(encoding, C_np)+Cnorm_np
        )
    return np.argmin(dist, axis=1)


def kmeans_codes(cut_encodings, kmeans_url):
    word_count = 0
    codes_dict = {}
    kmeans = kmeans_model(kmeans_url)

    for path in tqdm(cut_encodings, desc="Extracting KMeans Codes"):
        word_codes = []
        for word in cut_encodings[path]:
            codes = apply_kmeans(kmeans, word).tolist()
            word_codes.append(codes)
            word_count += 1
        codes_dict[path] = word_codes
    return codes_dict

def dusted_codes(cut_encodings, kmeans_url, gamma):
    word_count = 0
    codes_dict = {}
    
    kmeans = kmeans_model(kmeans_url)

    for path in tqdm(cut_encodings, desc="Extracting DUSTED Codes"):
        word_codes = []
        for word in cut_encodings[path]:
            print(word.shape)
            print(kmeans.cluster_centers_.shape)
            codes, _ = segment(word, kmeans.cluster_centers_, gamma)   
            word_codes.append(codes)
            word_count += 1
        codes_dict[path] = word_codes
    return codes_dict

def just_words(codes_dict):
    just_words = []
    just_words_collapsed = []

    for path in codes_dict:
        for word in codes_dict[path]:
            collapsed_word = [key for key, _ in groupby(word)]
            just_words.append(word)
            just_words_collapsed.append(collapsed_word)

    return just_words, just_words_collapsed


if __name__ == "__main__":
    in_dir = Path("data/librispeech_subset")
    align_dir = Path("data/all_alignments")
    model = "hubert_large"
    layer = 8

    features = Features(in_dir, align_dir, model, layer)
    
    encodings = features.create()
    cut_encodings = features.cut()
    codes_dict = kmeans_codes(cut_encodings, "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt")
    # codes_dict = dusted_codes(cut_encodings, "https://github.com/bshall/dusted/releases/download/v0.1/kmeans-english-50f36a.pt", 0.8)
    for path in codes_dict:
        print(codes_dict[path][0])
        break
    
    codes, collapsed_codes = just_words(codes_dict)
    codes_np = np.array(codes, dtype=object)
    collapsed_codes_np = np.array(collapsed_codes, dtype=object)

    out_dir = Path("output/codes/librispeech_subset")
    out_dir.mkdir(parents=True, exist_ok=True)

    collapsed_out_path  = out_dir / f"{model}_{layer}_collapsed_codes.npy"
    np.save(collapsed_out_path, collapsed_codes_np)

    out_path  = out_dir / f"{model}_{layer}_codes.npy"
    np.save(out_path, codes_np)


