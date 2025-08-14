import math
import numpy as np

try:  # pragma: no cover - optional dependency
    import torch # type: ignore
    import torch.nn as nn # type: ignore
    import torch.nn.functional as F  # type: ignore

    class _TDNNBlock(nn.Module):
        def __init__(self, inp, out, context):
            super().__init__()
            self.context = context
            self.lin = nn.Conv1d(inp, out, kernel_size=1)
            self.bn = nn.BatchNorm1d(out)
        def forward(self, x):  # x: B,C,T
            y = self.lin(x)
            return F.relu(self.bn(y))

    class TinyStressNet(nn.Module):
        def __init__(self, feat_dim=64):
            super().__init__()
            self.block1 = _TDNNBlock(feat_dim, 96, context=1)
            self.block2 = _TDNNBlock(96, 128, context=1)
            self.block3 = _TDNNBlock(128, 128, context=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            )
        def forward(self, x):  # x: B, F, T
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.pool(x).squeeze(-1)
            return self.head(x)
except Exception:  # torch not available
    torch = None  # type: ignore
    TinyStressNet = None  # type: ignore

class StressModel:
    """Encapsulates optional neural stress inference with graceful fallback."""
    def __init__(self, feat_dim=64):
        self.enabled = bool(torch)
        self.device = "cpu"
        if self.enabled:
            self.net = TinyStressNet(feat_dim=feat_dim).to(self.device)
            self.net.eval()
        else:
            self.net = None
    def infer(self, frame_feats: np.ndarray):
        """frame_feats: (T, F) numpy float32 log-mel+extras.
        Returns dict with model_stress_prob and embedding.
        """
        if not self.enabled or frame_feats.size == 0:
            return {"model_stress_prob": None, "embedding": None}
        with torch.no_grad():  # type: ignore
            t = torch
            assert t is not None
            x = t.from_numpy(frame_feats.T).unsqueeze(0).float()  # B,F,T
            logits = self.net(x.to(self.device))  # type: ignore
            prob = t.sigmoid(logits)[0,0].item()
            emb = x.mean(dim=2).cpu().numpy().tolist()
        return {"model_stress_prob": prob, "embedding": emb}

# Feature extraction helper
def compute_frame_features(y: np.ndarray, sr: int =16000, n_mels: int=64, win=0.025, hop=0.010):
    try:
        import librosa
        # Mel-spectrogram
        n_fft = int(sr*win)
        hop_len = int(sr*hop)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_len)
        logS = librosa.power_to_db(S + 1e-9).astype(np.float32)
        # MFCC (first 13)
        mfcc = librosa.feature.mfcc(S=logS, n_mfcc=13).astype(np.float32)
        # Spectral centroid & bandwidth
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        # Stack: mel + small summary (broadcast) via tiling to mel frames length
        # Align shapes
        T = logS.shape[1]
        def tile_feat(f):
            f2 = f if f.shape[1]==T else np.resize(f, (f.shape[0], T))
            return f2
        feats = np.vstack([
            logS,
            tile_feat(mfcc),
            tile_feat(centroid),
            tile_feat(bandwidth)
        ])
        return feats.T  # (T,F)
    except Exception:
        return np.zeros((0, n_mels), dtype=np.float32)
