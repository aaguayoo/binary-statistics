"""Base distribution class."""
from functools import partial

import numpy as np
from IPython.display import display
from pydantic.dataclasses import dataclass
from scipy import interpolate
from sympy import integrate, lambdify, oo, symbols


@dataclass
class BaseDistribution:
    """Base distribution class."""

    def __init__(self, dist_parameters):
        """Post init section.

        This built-in method is calle when the BaseDistribution class
        is instantiated. This function creates a self._pdf `partial` function
        which allows you to partially initialize a function.
        """
        self._pdf = partial(self.distribution, **dist_parameters)

    def __compute_pdf_norm(self, x: np.array):
        """Compute PDF norm."""
        self.norm_factor = (x.max() - x.min()) / len(x)
        if not hasattr(self, "norm"):
            self.norm = 1.0 / sum(self._pdf(value) * self.norm_factor for value in x)

    def cdf(self, x: np.array):
        """Cumulative Distribution Function."""
        self.__compute_pdf_norm(x)
        pdf_values = self._pdf(x)
        cumulative_sum = np.cumsum(pdf_values * self.norm_factor)
        return cumulative_sum / cumulative_sum[-1]

    def display_cdf(self, value: float = None):
        """Display CDF."""
        t, x = symbols("t x")
        cdf = integrate(self._pdf(t), (t, -oo, x))
        display(cdf)
        try:
            return lambdify(x, cdf)(value)
        except Exception:
            return None

    def display_pdf(self, value: float = None):
        """Display PDF."""
        x = symbols("x")
        pdf = self._pdf(x)
        display(pdf)
        try:
            return lambdify(x, pdf)(value)
        except Exception:
            return None

    def distribution(self, x, **kwargs):
        """Distribution function."""
        raise NotImplementedError(
            "Distribution function not implemented. "
            "It could be normalized or in general form."
        )

    def pdf(self, x):
        """Probability Distribution Function."""
        self.__compute_pdf_norm(x)
        return self._pdf(x) * self.norm

    def random_sample(self, x_min: float, x_max: float, size: int):
        """Random sample distribution."""
        x = np.linspace(x_min, x_max, size)
        cdf = self.cdf(x)

        cdf_inv = interpolate.interp1d(cdf, x)

        x_new = np.random.uniform(0.001, 0.999, size=size)
        return cdf_inv(x_new)
