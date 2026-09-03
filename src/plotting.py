"""
Shared publication-quality plotting style, used by analysis.py and
modeling.py so every figure in figures/ looks like part of the same paper.
"""

import matplotlib.pyplot as plt

RACE_PALETTE = {
    "White": "#4C72B0",
    "Black": "#DD8452",
    "Hispanic": "#55A868",
    "American Indian/Alaska Native": "#C44E52",
    "Asian": "#8172B2",
    "Pacific Islander": "#937860",
    "Two or more races": "#64B5CD",
    "Total": "#999999",
}


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })
