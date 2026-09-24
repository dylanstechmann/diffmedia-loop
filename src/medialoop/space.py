"""Factor bounds for a simulated screen.

The numbers are published in-vitro windows, used here as the box a planner
is allowed to search. They are not doses for a person and they are not a
claim that the center of the box is optimal.
"""

from __future__ import annotations

FACTORS = (
    {
        "name": "CHIR99021_uM",
        "low": 0.0,
        "high": 12.0,
        "cite": "Lian et al. 2013 used a line-specific dose inside this window (example 12 µM).",
    },
    {
        "name": "IWP2_uM",
        "low": 0.0,
        "high": 5.0,
        "cite": "Lian et al. 2013, 5 µM IWP2 in the worked GiWi example.",
    },
    {
        "name": "SB431542_uM",
        "low": 0.0,
        "high": 10.0,
        "cite": "Chambers et al. 2009, 10 µM.",
    },
    {
        "name": "LDN193189_nM",
        "low": 0.0,
        "high": 250.0,
        "cite": "100 nM is a later Noggin substitute in dual-SMAD protocols, not a Chambers 2009 condition.",
    },
)

NAMES = tuple(item["name"] for item in FACTORS)


def clip(name: str, value: float) -> float:
    for item in FACTORS:
        if item["name"] == name:
            return min(item["high"], max(item["low"], value))
    raise KeyError(name)
