"""Base distribution class."""

from functools import partial
from typing import Callable

import numpy as np
from pydantic.dataclasses import dataclass
from scipy import interpolate


@dataclass
class BaseDistribution:
    """Base distribution class."""

    dist_function: Callable
    dist_parameters: dict = None

    def __post_init_post_parse__(self):
        """Post init section.

        This built-in method is calle when the BaseDistribution class
        is instantiated. This function creates a self._pdf `partial` function
        which allows you to partially initialize a function.
        """
        if not self.dist_parameters:
            self.dist_parameters = {}

        self._pdf = partial(self.dist_function, **self.dist_parameters)

    def __compute_pdf_norm(self, x: np.array):
        """Compute PDF norm."""
        self.norm_factor = (x.max() - x.min()) / len(x)
        if not hasattr(self, "norm"):
            self.norm = sum(self._pdf(value) * self.norm_factor for value in x)

    def cdf(self, x: np.array):
        """Cumulative Distribution Function."""
        self.__compute_pdf_norm(x)
        pdf_values = self._pdf(x)
        cumulative_sum = np.cumsum(pdf_values * self.norm_factor)
        return cumulative_sum / cumulative_sum[-1]

    def pdf(self, x, isNormalized: bool = False):
        """Probability Distribution Function."""
        self.__compute_pdf_norm(x)
        return self._pdf(x) / self.norm

    def random_sample(self, x_min: float, x_max: float, size: int):
        """Random sample distribution."""
        x = np.linspace(x_min, x_max, size)
        cdf = self.cdf(x)

        cdf_inv = interpolate.interp1d(cdf, x)

        x_new = np.random.uniform(0.001, 0.999, size=size)
        return cdf_inv(x_new)
