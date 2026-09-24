"""A small Gaussian process with fixed hyperparameters and expected improvement."""

from __future__ import annotations

import math

import numpy as np

from medialoop.space import FACTORS


def _phi(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z: np.ndarray) -> np.ndarray:
    # standard normal CDF via erf
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def normalize(x: np.ndarray) -> np.ndarray:
    lows = np.array([item["low"] for item in FACTORS])
    highs = np.array([item["high"] for item in FACTORS])
    return (x - lows) / (highs - lows)


def _kernel(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * dist / (length ** 2))


def predict(x_train: np.ndarray, y: np.ndarray, x_cand: np.ndarray, length: float = 0.35, noise: float = 0.02):
    y_mean = float(y.mean())
    y_std = float(y.std())
    if y_std < 1e-8:
        return np.full(len(x_cand), y_mean), np.ones(len(x_cand)) * 1e-3
    yn = (y - y_mean) / y_std
    xn = normalize(x_train)
    cn = normalize(x_cand)
    noise_n = (noise / y_std) ** 2
    k = _kernel(xn, xn, length) + (noise_n + 1e-6) * np.eye(len(xn))
    chol = np.linalg.cholesky(k)
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, yn))
    ks = _kernel(cn, xn, length)
    mu = ks @ alpha
    solved = np.linalg.solve(chol, ks.T)
    var = np.maximum(1.0 - np.sum(solved ** 2, axis=0), 1e-8)
    return mu * y_std + y_mean, np.sqrt(var) * y_std


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-9)
    improve = mu - best - xi
    z = improve / sigma
    return improve * _Phi(z) + sigma * _phi(z)
