"""Distributions classes."""
import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass
from sympy import Piecewise
from sympy.core.symbol import Symbol

from binaryStatistics.base_distribution import BaseDistribution


@dataclass
class Uniform(BaseDistribution):
    """Uniform distribution class."""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init BaseDistribution."""
        super().__init__(self.dist_parameters)

    def distribution(self, x, a=None, b=None):
        """Distribution method."""
        if a and b:
            self.norm = 1 / (b - a)
            if isinstance(x, Symbol):
                return Piecewise((0, x < a), (0, x > b), (1 * self.norm, True))
            if isinstance(x, int | float):
                return 1 * self.norm if a <= x <= b else 0.0
            if isinstance(x, np.ndarray):
                return np.piecewise(
                    x, [(x < a), (x <= b) * (x >= a)], [0, 1 * self.norm]
                )
        else:
            self.norm = 1
            return np.ones_like(x) if isinstance(x, np.ndarray) else 1


@dataclass
class Thermal(BaseDistribution):
    """Thermal distribution class."""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init BaseDistribution."""
        super().__init__(self.dist_parameters)

    def distribution(self, x):
        """Distribution method."""
        return 2 * x


@dataclass
class PowerLaw(BaseDistribution):
    """Power law distribution class."""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init BaseDistribution."""
        super().__init__(self.dist_parameters)

    def distribution(self, x):
        """Distribution method."""
        return 1 / x
