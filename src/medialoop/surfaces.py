"""Hidden response surfaces for the bake-off.

These are cartoons of the pathway logic (Wnt up then Wnt down for cardiac;
dual SMAD inhibition for neural). They are not differentiation data.
A collaborator replaces `observe` with a real assay column.
"""

from __future__ import annotations

import math


def _gauss(x: float, center: float, width: float) -> float:
    z = (x - center) / width
    return math.exp(-0.5 * z * z)


def cardiac(row: dict[str, float]) -> float:
    chir = _gauss(row["CHIR99021_uM"], 8.0, 2.0)
    iwp = _gauss(row["IWP2_uM"], 4.0, 1.2)
    # dual-SMAD inhibitors push away from cardiac in this cartoon
    sb = _gauss(row["SB431542_uM"], 0.0, 3.0)
    ldn = _gauss(row["LDN193189_nM"], 0.0, 40.0)
    return chir * iwp * sb * ldn


def neural(row: dict[str, float]) -> float:
    sb = _gauss(row["SB431542_uM"], 10.0, 3.0)
    ldn = _gauss(row["LDN193189_nM"], 100.0, 40.0)
    chir = _gauss(row["CHIR99021_uM"], 0.0, 2.5)
    iwp = _gauss(row["IWP2_uM"], 0.0, 2.0)
    return sb * ldn * chir * iwp


def cardiac_shifted(row: dict[str, float]) -> float:
    """Cardiac cartoon with the optimum moved to (6, 2) µM.

    Same pathway logic and widths, different peak. It checks that a planner
    follows the surface instead of the original cartoon's peak location.
    """
    chir = _gauss(row["CHIR99021_uM"], 6.0, 2.0)
    iwp = _gauss(row["IWP2_uM"], 2.0, 1.2)
    sb = _gauss(row["SB431542_uM"], 0.0, 3.0)
    ldn = _gauss(row["LDN193189_nM"], 0.0, 40.0)
    return chir * iwp * sb * ldn


def neural_shifted(row: dict[str, float]) -> float:
    """Neural cartoon with the optimum moved to (5 µM, 250 nM)."""
    sb = _gauss(row["SB431542_uM"], 5.0, 3.0)
    ldn = _gauss(row["LDN193189_nM"], 250.0, 40.0)
    chir = _gauss(row["CHIR99021_uM"], 0.0, 2.5)
    iwp = _gauss(row["IWP2_uM"], 0.0, 2.0)
    return sb * ldn * chir * iwp


OBJECTIVES = {
    "cardiac": cardiac,
    "neural": neural,
    "cardiac_shifted": cardiac_shifted,
    "neural_shifted": neural_shifted,
}
