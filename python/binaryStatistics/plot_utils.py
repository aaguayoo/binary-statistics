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
    pi_factor: int = 0,
    file: str = None,
):
    """Plot histogram."""
    if pi_factor:
        distribution = distribution
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

    if pi_factor:
        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / pi_factor))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    if density:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("Counts")
    if "limits" in y_params:
        ax.set_ylim(y_params["limits"])
    if "scale" in y_params:
        ax.set_yscale(y_params["scale"])

    if file:
        plt.savefig(file, dpi=300)


def multiple_formatter(denominator=8, number=np.pi, latex="\\pi"):
    """Multiple formatter."""

    def gcd(a, b):
        """GCD"""
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        """Multiple formatter."""
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return f"${latex}$"
            elif num == -1:
                return f"$-{latex}$"
            else:
                return f"${num}{latex}$"
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    """Multiple class."""

    def __init__(self, denominator=2, number=np.pi, latex="\\pi"):
        """Init method."""
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        """Locator"""
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        """Formatter"""
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )
