"""Plot utilities."""
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(
    distribution: list | np.ndarray,
    bins: int = 30,
    density: bool = False,
    color: str = None,
    x_params: dict = {"label": "X", "limits": None, "scale": "linear"},
    y_params: dict = {"limits": None, "scale": "linear"},
    figsize=(10, 5),
):
    """Plot histogram."""
    _, ax = plt.subplots(figsize=figsize)
    _ = ax.hist(distribution, bins=bins, density=density, color=color)

    if "label" in x_params:
        ax.set_xlabel(x_params["label"])
    else:
        ax.set_xlabel("X")
    if "limits" in x_params:
        ax.set_xlim(x_params["limits"])
    else:
        ax.set_xlim(distribution.min(), distribution.max())
    if "scale" in x_params:
        ax.set_xscale(x_params["scale"])

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")
    if "limits" in y_params:
        ax.set_ylim(y_params["limits"])
    if "scale" in y_params:
        ax.set_yscale(y_params["scale"])
