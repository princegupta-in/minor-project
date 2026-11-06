"""Classical baseline models for forest cover change prediction.

Included for comparison against deep-learning approaches:
  - CAMarkov: Cellular Automata + Markov Chain transition probabilities
  - LinearExtrapolation: Simple per-pixel linear trend extrapolation
  - RandomForestBaseline: sklearn RF on flattened per-pixel spectral features

These do not require PyTorch and can run on CPU-only machines.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CAMarkov:
    """Cellular Automata – Markov Chain forest cover transition model.

    Learns a per-class transition probability matrix from historical label maps
    and applies it pixel-by-pixel with a CA neighbourhood suitability filter.

    Args:
        n_classes: number of LULC classes (6 for this project)
        ca_radius: neighbourhood radius for suitability filter (pixels)
    """

    def __init__(self, n_classes: int = 6, ca_radius: int = 3):
        self.n_classes = n_classes
        self.ca_radius = ca_radius
        self.transition_matrix: Optional[np.ndarray] = None  # (n_classes, n_classes)

    def fit(self, labels: np.ndarray) -> "CAMarkov":
        """Estimate transition matrix from a sequence of label maps.

        Args:
            labels: (T, H, W) integer label array
        """
        T = labels.shape[0]
        tm = np.zeros((self.n_classes, self.n_classes), dtype=np.float64)
        for t in range(T - 1):
            src = labels[t].ravel()
            dst = labels[t + 1].ravel()
            valid = (src >= 0) & (src < self.n_classes) & (dst >= 0) & (dst < self.n_classes)
            for s, d in zip(src[valid], dst[valid]):
                tm[s, d] += 1
        # Row-normalize
        row_sum = tm.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        self.transition_matrix = tm / row_sum
        return self

    def _ca_suitability(self, label_map: np.ndarray, target_class: int) -> np.ndarray:
        """Fraction of pixels in a square neighbourhood that equal target_class."""
        from scipy.ndimage import uniform_filter
        mask = (label_map == target_class).astype(np.float32)
        size = 2 * self.ca_radius + 1
        return uniform_filter(mask, size=size)

    def predict(self, label_t: np.ndarray) -> np.ndarray:
        """Predict the next-year label map from the current one.

        Args:
            label_t: (H, W) integer label map for time t
        Returns:
            (H, W) predicted label map for time t+1
        """
        if self.transition_matrix is None:
            raise RuntimeError("Call .fit() before .predict()")
        H, W = label_t.shape
        suitability = np.stack(
            [self._ca_suitability(label_t, c) for c in range(self.n_classes)], axis=0
        )  # (n_classes, H, W)
        # Transition probabilities for each pixel's current class
        trans_prob = self.transition_matrix[label_t.ravel()].reshape(H, W, self.n_classes)
        # Weight by CA suitability
        weighted = trans_prob * suitability.transpose(1, 2, 0)
        return weighted.argmax(axis=-1).astype(np.int16)

    def predict_sequence(self, label_t: np.ndarray, steps: int) -> np.ndarray:
        """Autoregressively predict `steps` future label maps."""
        preds = []
        current = label_t.copy()
        for _ in range(steps):
            current = self.predict(current)
            preds.append(current)
        return np.stack(preds, axis=0)


class LinearExtrapolation:
    """Per-pixel linear trend extrapolation on one-hot class maps.

    Simple baseline: fit a linear trend to the fraction of each class per pixel
    over time, then extrapolate and take argmax.
    """

    def __init__(self, n_classes: int = 6):
        self.n_classes = n_classes
        self._coeffs: Optional[np.ndarray] = None   # (n_classes, H, W, 2)
        self._t_max: int = 0

    def fit(self, labels: np.ndarray) -> "LinearExtrapolation":
        """labels: (T, H, W)"""
        T, H, W = labels.shape
        self._t_max = T - 1
        t = np.arange(T, dtype=np.float32)
        # One-hot: (T, n_classes, H, W)
        one_hot = (labels[:, None] == np.arange(self.n_classes)[None, :, None, None]).astype(np.float32)
        # Fit per-class per-pixel line via least squares
        A = np.stack([t, np.ones(T)], axis=1)  # (T, 2)
        # (n_classes, H, W, T) @ (T, 2) -> (n_classes, H, W, 2)
        oh_flat = one_hot.reshape(T, self.n_classes * H * W)
        coeffs  = np.linalg.lstsq(A, oh_flat, rcond=None)[0]   # (2, n_classes*H*W)
        self._coeffs = coeffs.T.reshape(self.n_classes, H, W, 2)
        return self

    def predict_at(self, t_future: int) -> np.ndarray:
        """Predict label map at absolute time index t_future."""
        if self._coeffs is None:
            raise RuntimeError("Call .fit() first")
        # prob(c, h, w) = slope * t + intercept
        probs = self._coeffs[..., 0] * t_future + self._coeffs[..., 1]  # (n_classes, H, W)
        probs = np.clip(probs, 0, 1)
        return probs.argmax(axis=0).astype(np.int16)

    def predict_sequence(self, steps: int) -> np.ndarray:
        """Predict `steps` future frames starting from t_max+1."""
        return np.stack(
            [self.predict_at(self._t_max + 1 + s) for s in range(steps)], axis=0
        )


class RandomForestBaseline:
    """Random Forest pixel classifier operating on per-pixel spectral time series.

    Each pixel's feature vector is the concatenation of its band values across
    all timesteps (flattened). Trains on the last labelled year and predicts
    the next year's label.

    Requires scikit-learn.
    """

    def __init__(self, n_estimators: int = 100, n_jobs: int = -1):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)
        self._H = self._W = 0

    def fit(self, frames: np.ndarray, label: np.ndarray) -> "RandomForestBaseline":
        """
        Args:
            frames: (T, C, H, W) float32
            label:  (H, W)       int labels for the prediction target year
        """
        T, C, H, W = frames.shape
        self._H, self._W = H, W
        X = frames.transpose(2, 3, 0, 1).reshape(H * W, T * C)   # (N, T*C)
        y = label.ravel()
        self.model.fit(X, y)
        return self

    def predict(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, C, H, W) → predicted label (H, W)"""
        T, C, H, W = frames.shape
        X = frames.transpose(2, 3, 0, 1).reshape(H * W, T * C)
        return self.model.predict(X).reshape(H, W).astype(np.int16)
