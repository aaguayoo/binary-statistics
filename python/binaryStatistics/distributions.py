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

    def distribution(self, x, C, alpha):
        """Distribution method."""
        return C * x**alpha


@dataclass
class Log(BaseDistribution):
    """Power law distribution class."""

    alpha = -1
    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init BaseDistribution."""
        super().__init__(self.dist_parameters)

    def distribution(self, x):
        """Distribution method."""
        return x**self.alpha


@dataclass
class VelTilde(BaseDistribution):
    """V-tilde distribution"""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init section."""
        super().__init__(self.dist_parameters)

    def distribution(self, phi, phi_0, i, e):
        """Distribution."""
        temp2 = 1.0 - (np.sin(i) ** 2) * (np.cos(phi - phi_0) ** 2)
        temp2 = temp2**0.25

        temp1 = e * np.sin(phi_0) - np.sin(phi - phi_0)
        temp1 = temp1 * temp1

        temp3 = (1.0 + e * e + 2 * e * np.cos(phi) - np.sin(i) * np.sin(i) * temp1) / (
            1.0 + e * np.cos(phi)
        )
        temp3 = np.sqrt(temp3)

        return temp2 * temp3


@dataclass
class PhiAngle(BaseDistribution):
    """Phi angle distribution."""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Init section."""
        super().__init__(self.dist_parameters)

    def distribution(self, phi, e):
        """Distribution."""
        return ((1.0 - e**2.0) ** (3.0 / 2.0)) / (
            2 * np.pi * (1 + e * np.cos(phi)) ** 2
        )

    def random_sample(self, x_min, x_max, size):
        """Random sample."""
        phi_ = []
        e = self.dist_parameters["e"]
        for index in range(size):
            LI = 0
            while LI < 1:
                E = e[index]
                c = (1 - E**2) / (2 * np.pi * (1 - E) ** 2)
                base = x_max - x_min
                Phi = np.random.uniform() * base
                pPhi = np.random.uniform() * c
                if pPhi <= self.distribution(Phi, E):
                    LI = 2
                    phi_.append(Phi)
        return phi_


@dataclass
class Sine(BaseDistribution):
    """Sine distribution."""

    dist_parameters: dict = Field(default={})

    def __post_init_post_parse__(self):
        """Initialize."""
        super().__init__(self.dist_parameters)

    def distribution(self, i):
        """Distribution."""
        return np.sin(i)
