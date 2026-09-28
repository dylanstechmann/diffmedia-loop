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


def batch_acquire(
    x_train: np.ndarray,
    y_train: np.ndarray,
    pool: np.ndarray,
    batch_size: int,
    strategy: str = "kriging_believer",
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Select candidate indices using Kriging Believer or Constant Liar.

    Strategies:
    - 'kriging_believer': Imputes the GP posterior mean mu(x) at each selected point.
    - 'constant_liar_min': Imputes min(y) (pessimistic / exploratory).
    - 'constant_liar_max': Imputes max(y) (optimistic).
    - 'constant_liar_mean': Imputes mean(y).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if batch_size > len(pool):
        raise ValueError("batch_size cannot exceed candidate pool size")

    selected_indices: list[int] = []
    available_mask = np.ones(len(pool), dtype=bool)

    cur_x = np.copy(x_train)
    cur_y = np.copy(y_train)

    for b in range(batch_size):
        avail_indices = np.flatnonzero(available_mask)
        if len(avail_indices) == 0:
            break
        cand_x = pool[avail_indices]
        mu, sigma = predict(cur_x, cur_y, cand_x, noise=noise)
        ei = expected_improvement(mu, sigma, float(cur_y.max()))

        # Resolve ties reproducibly
        best_cand = np.flatnonzero(np.isclose(ei, ei.max(), rtol=1e-10, atol=1e-15))
        if rng is not None:
            local_idx = int(rng.choice(best_cand))
        else:
            local_idx = int(best_cand[0])

        chosen_idx = int(avail_indices[local_idx])
        selected_indices.append(chosen_idx)
        available_mask[chosen_idx] = False

        if b < batch_size - 1:
            pick_x = pool[chosen_idx : chosen_idx + 1]
            if strategy == "kriging_believer":
                imputed_y = float(mu[local_idx])
            elif strategy == "constant_liar_min":
                imputed_y = float(cur_y.min())
            elif strategy == "constant_liar_max":
                imputed_y = float(cur_y.max())
            elif strategy == "constant_liar_mean":
                imputed_y = float(cur_y.mean())
            else:
                raise ValueError(f"unknown batch strategy: {strategy!r}")

            cur_x = np.vstack([cur_x, pick_x])
            cur_y = np.append(cur_y, imputed_y)

    return selected_indices


def run(
    objective,
    n_init: int,
    n_steps: int,
    seed: int,
    noise: float = 0.02,
    candidates: int = 400,
    batch_size: int = 1,
    batch_strategy: str = "kriging_believer",
):
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    rng = np.random.default_rng(seed)
    x = _sample(rng, n_init)
    true = np.array([objective(_row(row)) for row in x])
    y = true + rng.normal(0.0, noise, size=n_init)
    for _ in range(n_steps):
        pool = _sample(rng, candidates)
        if batch_size == 1:
            mu, sigma = predict(x, y, pool, noise=noise)
            score = expected_improvement(mu, sigma, float(y.max()))
            pick = pool[int(np.argmax(score))]
            value = objective(_row(pick))
            x = np.vstack([x, pick])
            true = np.append(true, value)
            y = np.append(y, value + rng.normal(0.0, noise))
        else:
            picks_idx = batch_acquire(x, y, pool, batch_size=batch_size, strategy=batch_strategy, noise=noise, rng=rng)
            for idx in picks_idx:
                pick = pool[idx]
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
