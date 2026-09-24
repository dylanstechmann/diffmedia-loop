"""Sequential design: random init, then expected improvement."""

from __future__ import annotations

import numpy as np

from medialoop.gp import expected_improvement, predict
from medialoop.space import FACTORS, NAMES


def _row(vector: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(NAMES, vector)}


def _sample(rng: np.random.Generator, n: int) -> np.ndarray:
    lows = np.array([item["low"] for item in FACTORS])
    highs = np.array([item["high"] for item in FACTORS])
    return rng.uniform(lows, highs, size=(n, len(FACTORS)))


def run(objective, n_init: int, n_steps: int, seed: int, noise: float = 0.02, candidates: int = 400):
    rng = np.random.default_rng(seed)
    x = _sample(rng, n_init)
    true = np.array([objective(_row(row)) for row in x])
    y = true + rng.normal(0.0, noise, size=n_init)
    for _ in range(n_steps):
        pool = _sample(rng, candidates)
        mu, sigma = predict(x, y, pool, noise=noise)
        score = expected_improvement(mu, sigma, float(y.max()))
        pick = pool[int(np.argmax(score))]
        value = objective(_row(pick))
        x = np.vstack([x, pick])
        true = np.append(true, value)
        y = np.append(y, value + rng.normal(0.0, noise))
    return {"x": x, "true": true, "best_true": float(true.max())}


def random_search(objective, budget: int, seed: int):
    rng = np.random.default_rng(seed)
    x = _sample(rng, budget)
    true = np.array([objective(_row(row)) for row in x])
    return {"best_true": float(true.max())}


def inside_bounds(x: np.ndarray) -> bool:
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != len(FACTORS) or not np.isfinite(x).all():
        return False
    for col, item in enumerate(FACTORS):
        if np.any(x[:, col] < item["low"] - 1e-9) or np.any(x[:, col] > item["high"] + 1e-9):
            return False
    return True
