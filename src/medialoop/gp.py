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


def predict(x_train: np.ndarray, y: np.ndarray, x_cand: np.ndarray, length: float = 0.35, noise: float | np.ndarray = 0.02):
    x_train, y, x_cand = (np.asarray(value, dtype=float) for value in (x_train, y, x_cand))
    if (x_train.ndim != 2 or x_train.shape[1] != len(FACTORS) or not len(x_train)
            or y.shape != (len(x_train),) or x_cand.ndim != 2
            or x_cand.shape[1] != len(FACTORS) or not len(x_cand)):
        raise ValueError("GP needs nonempty factor matrices and one response per training row")
    if not all(np.isfinite(value).all() for value in (x_train, y, x_cand)):
        raise ValueError("GP inputs must be finite")
    noise_values = np.asarray(noise, dtype=float)
    if noise_values.ndim == 0:
        noise_values = np.full(len(x_train), float(noise_values))
    if (not np.isfinite(length) or length <= 0 or noise_values.shape != (len(x_train),)
            or not np.isfinite(noise_values).all() or (noise_values < 0).any()):
        raise ValueError("length must be positive; noise must be a finite, nonnegative scalar or one value per observation")
    y_mean = float(y.mean())
    # Flat initial readouts still leave uncertainty away from measured points.
    y_std = max(float(y.std()), float(noise_values.max()), 1e-3)
    yn = (y - y_mean) / y_std
    xn = normalize(x_train)
    cn = normalize(x_cand)
    noise_n = (noise_values / y_std) ** 2
    k = _kernel(xn, xn, length) + np.diag(noise_n + 1e-6)
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


def upper_confidence_bound(mu: np.ndarray, sigma: np.ndarray, kappa: float = 1.5) -> np.ndarray:
    """Fixed-kappa UCB. kappa is a baseline, not a calibrated confidence level."""
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    return np.asarray(mu, dtype=float) + kappa * np.maximum(np.asarray(sigma, dtype=float), 0.0)


def thompson_draw(mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One posterior draw per candidate. Not a multi-sample Thompson average."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-9)
    return rng.normal(np.asarray(mu, dtype=float), sigma)
