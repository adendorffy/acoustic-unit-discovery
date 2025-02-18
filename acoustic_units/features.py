import numpy as np
from pathlib import Path
import torchaudio
from tqdm import tqdm
import torch
import librosa

model_pipelines = {
    "hubert_base": torchaudio.pipelines.HUBERT_BASE,
    "hubert_large": torchaudio.pipelines.HUBERT_LARGE,
    "hubert_xlarge": torchaudio.pipelines.HUBERT_XLARGE,
    "wavlm_base": torchaudio.pipelines.WAVLM_BASE,
    "wavlm_large": torchaudio.pipelines.WAVLM_LARGE,
    "wavlm_base_plus": torchaudio.pipelines.WAVLM_BASE_PLUS,
}

class Features:
    def __init__(self, in_dir, align_dir, model, layer):
        self.in_dir = in_dir
        self.align_dir = align_dir
        self.model = model
        if model == "mfcc":
            self.layer = 0
            self.feat_dir = Path(f"features/{model}/")
            self.feat_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.layer = layer
            self.feat_dir = Path(f"features/{model}/{layer}")
            self.feat_dir.mkdir(parents=True, exist_ok=True)
        self.encodings = {}
        self.cut_encodings = {}

    @classmethod
    def get_frame_num(self, timestamp: float, sample_rate: int, frame_size_ms: int)->int:
        """
        Convert timestamp (in seconds) to frame index based on sampling rate and frame size.
        """
        hop_size = frame_size_ms/1000 * sample_rate
        hop_size = np.max([hop_size, 1])
        return int((timestamp * sample_rate) / hop_size)
    
    @classmethod
    def preemphasis(self, signal, coeff=0.97):
        """Perform preemphasis on the input `signal`."""    
        return np.append(signal[0], signal[1:] - coeff*signal[:-1])
    
    def create(self, audio_ext=".wav"):

        paths = list(self.in_dir.rglob(f"*{audio_ext}"))
        if self.model != "mfcc":
            bundle = model_pipelines.get(self.model, torchaudio.pipelines.HUBERT_BASE)
            model = bundle.get_model()
            model.eval()

        encodings = {}
        for path in tqdm(paths, desc="Encoding Features"):
            if self.model != "mfcc":
                wav, sr = torchaudio.load(path)
                wav = torchaudio.functional.resample(wav, sr, 16000)

                with torch.inference_mode():
                    encoding, _ = model.extract_features(wav, num_layers=self.layer)
                
                encoding = encoding[self.layer-1].squeeze().cpu().numpy()

            else:
                wav, sr = librosa.core.load(path, sr=None)
                wav = self.preemphasis(wav, coeff=0.97)

                mfcc = librosa.feature.mfcc(
                    y=wav, sr=sr, n_mfcc=13, n_mels=24, 
                    n_fft=int(np.floor(0.025*sr)),
                    hop_length=int(np.floor(0.01*sr)), 
                    fmin=64, fmax=8000
                )
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
                encoding = np.hstack([mfcc.T, mfcc_delta.T, mfcc_delta_delta.T])
            
            output_path = Path(self.feat_dir) / f"{path.stem}.npy"
            np.save(output_path, encoding) 
            encodings[path.stem] = encoding

        self.encodings = encodings
        return encodings
    
    def load(self):
        if not self.encodings:
            paths = list(self.feat_dir.rglob(f"*.npy"))
            for path in tqdm(paths, desc="Loading Features"):
                encoding = np.load(path)
                self.encodings[path.stem] = encoding
            
        return self.encodings
    
    def cut(self, encodings=None):
        alignment_files = list(self.align_dir.rglob("*.list"))
        cut_encodings = {}

        if not self.encodings:
            encodings = self.load()
        else:
            encodings = self.encodings

        for path in tqdm(encodings, desc="Cut Encodings"):
            align_file = [a for a in alignment_files if a.stem == path]
            if not align_file:
                continue
            else:
                align_file = align_file[0]
            
            with open(str(align_file), "r") as f:
                bounds = [self.get_frame_num(float(line.strip()), 16000, 20) for line in f]

            cuts = []
            for i in range(len(bounds)-1):
                cut_encoding = encodings[path][bounds[i]: bounds[i+1]]
                cuts.append(cut_encoding)
            cut_encodings[path] = cuts

        self.cut_encodings = cut_encodings
        return cut_encodings    