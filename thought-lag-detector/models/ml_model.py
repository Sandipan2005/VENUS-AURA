import math
from collections import defaultdict

FEATURE_ORDER = [
    "reaction_ms_server",
    "pause_ratio",
    "pitch_cv",
    "intensity_var",
    "speaking_rate_wps",
    "filler_count",
]

class OnlineNormalizer:
    """Keeps running mean/std for each feature (Welford)."""
    def __init__(self):
        self.count = defaultdict(int)
        self.mean = defaultdict(float)
        self.M2 = defaultdict(float)
    def update(self, key, value):
        c = self.count[key] + 1
        delta = value - self.mean[key]
        self.mean[key] += delta / c
        delta2 = value - self.mean[key]
        self.M2[key] += delta * delta2
        self.count[key] = c
    def std(self, key):
        c = self.count[key]
        if c < 2:
            return 1.0
        return math.sqrt(self.M2[key] / (c - 1)) or 1.0
    def norm(self, key, value):
        return (value - self.mean[key]) / self.std(key)

class OnlineLogisticRegression:
    """Tiny online logistic regression with L2 regularization.
    Pseudo-labels derived from heuristic stress (> threshold => 1).
    """
    def __init__(self, lr=0.05, l2=1e-3, threshold=60.0, min_pos=5, min_neg=5):
        self.lr = lr
        self.l2 = l2
        self.threshold = threshold
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.w = [0.0] * (len(FEATURE_ORDER) + 1)  # bias + features
        self.norm = OnlineNormalizer()
        self.pos = 0
        self.neg = 0
        self._trained_once = False
    def _sigmoid(self, z):
        if z < -50:
            return 1e-22
        if z > 50:
            return 1 - 1e-22
        return 1.0 / (1.0 + math.exp(-z))
    def update_and_predict(self, features: dict, heuristic_stress: float):
        y = 1 if heuristic_stress >= self.threshold else 0
        if y:
            self.pos += 1
        else:
            self.neg += 1
        x_raw = []
        for k in FEATURE_ORDER:
            v = float(features.get(k, 0.0) or 0.0)
            self.norm.update(k, v)
            x_raw.append(self.norm.norm(k, v))
        if self.pos >= self.min_pos and self.neg >= self.min_neg:
            self._trained_once = True
            z = self.w[0]
            for i, xv in enumerate(x_raw):
                z += self.w[i + 1] * xv
            p = self._sigmoid(z)
            err = p - y
            self.w[0] -= self.lr * (err + self.l2 * self.w[0])
            for i, xv in enumerate(x_raw):
                grad = err * xv + self.l2 * self.w[i + 1]
                self.w[i + 1] -= self.lr * grad
        z = self.w[0]
        for i, xv in enumerate(x_raw):
            z += self.w[i + 1] * xv
        p = self._sigmoid(z)
        return {
            "ml_stress_prob": float(p * 100.0),
            "ml_trained": self._trained_once,
            "ml_pos": self.pos,
            "ml_neg": self.neg,
        }

MLModel = OnlineLogisticRegression
