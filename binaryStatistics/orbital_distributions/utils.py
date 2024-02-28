"""Utility functions."""
import numpy as np

from binaryStatistics.orbital_distributions.distributions import PhiAngle


def get_phi_angle(eccentricity: float, phi_0: float):
    """Get phi angle."""
    phi = PhiAngle(
        dist_parameters={
            "e": [eccentricity],
        }
    )
    return phi.random_sample(0, 2 * np.pi, size=1)[0] + phi_0
