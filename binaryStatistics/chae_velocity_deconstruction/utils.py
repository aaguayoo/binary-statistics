"""Chae velocity reconstruction utilities."""
import warnings

import numpy as np

from binaryStatistics.constants import G_pc_km_s_Modot
from binaryStatistics.orbital_distributions.distributions import Log


def get_relative_v3D(
    mass: np.ndarray, separation_3D: np.ndarray, semimajor_axis: np.ndarray
):
    """
    Compute the 3D relative velocity.

    This function calculates the 3D relative velocity using the following formula:
    v_3D = sqrt((G * mass / separation_3D) * (2 - separation_3D / semimajor_axis))

    Parameters:
        mass (np.ndarray):
            Array of masses.

        separation_3D (np.ndarray):
            Array of 3D separations.

        semimajor_axis (np.ndarray):
            Array of semimajor axes.

    Returns:
        np.ndarray: Array of computed 3D relative velocities.
    """
    G = G_pc_km_s_Modot
    return np.sqrt((G * mass / separation_3D) * (2 - separation_3D / semimajor_axis))


def get_photometric_distance(
    mass_1: np.ndarray,
    mass_2: np.ndarray,
    inner_semimajor_axis: np.ndarray,
    outer_semimajor_axis: np.ndarray,
):
    """Compute the photometric distance.

    This function calculates the photometric distance based on the masses of the binary
    system and the semimajor axes of the inner and outer orbits. It uses the following
    formula:
    - If the inner semimajor axis is less than 0.3 times the outer semimajor axis:
        distance = (mass_h * mass_c * (mass_h^(alp-1) - mass_c^(alp-1))) /
        ((mass_h + mass_c) * (mass_h^alp + mass_c^alp))
    - Otherwise:
        distance = mass_c / (mass_c + mass_h)

    Parameters:
        mass_1 (np.ndarray):
            Array of masses for the first component.

        mass_2 (np.ndarray):
            Array of masses for the second component.

        inner_semimajor_axis (np.ndarray):
            Array of semimajor axes for the inner orbit.

        outer_semimajor_axis (np.ndarray):
            Array of semimajor axes for the outer orbit.

    Returns:
        np.ndarray: Array of computed photometric distances.
    """
    if mass_1 <= mass_2:
        mass_c = mass_1
        mass_h = mass_2
    else:
        mass_c = mass_2
        mass_h = mass_1
    alp = 3.5
    return (
        (
            mass_h
            * mass_c
            * (mass_h ** (alp - 1) - mass_c ** (alp - 1))
            / ((mass_h + mass_c) * (mass_h**alp + mass_c**alp))
        )
        if inner_semimajor_axis < 0.3 * outer_semimajor_axis
        else mass_c / (mass_c + mass_h)
    )


def get_semimajor_axis(
    separation_3D: np.array, eccentricity: np.array, phi: np.array, phi_0: np.array
):
    """Compute the semimajor axis.

    This function calculates the semimajor axis of a binary system based on the 3D
    separation, eccentricity, phase angle, and reference phase angle. It uses the
    following formula:
    semimajor_axis = separation_3D * (1 + eccentricity * cos(phi - phi_0)) /
    (1 - eccentricity^2)

    Parameters:
        separation_3D (np.array):
            Array of 3D separations.

        eccentricity (np.array):
            Array of eccentricities.

        phi (np.array):
            Array of phase angles.

        phi_0 (np.array):
            Array of reference phase angles.

    Returns:
        np.array: Array of computed semimajor axes.
    """
    return (
        separation_3D
        * (1 + eccentricity * np.cos(phi - phi_0))
        / (1 - eccentricity**2)
    )


def get_inner_semimajor_axis(parallax: np.ndarray, inner_binary_mass: np.ndarray):
    """Compute the inner semimajor axis.

    This function calculates the inner semimajor axis of a binary system based on the
    parallax and the mass of the inner binary. It uses a logarithmic random sampling
    method to generate a value for the semimajor axis and then calculates the period
    using the formula P = sqrt(a_in^3 / inner_binary_mass). The function ensures that
    the calculated period is greater than 3 before returning the computed semimajor
    axis.

    Parameters:
        parallax (np.ndarray):
            Array of parallax values.

        inner_binary_mass (np.ndarray):
            Array of masses for the inner binary.

    Returns:
        np.ndarray: Array of computed inner semimajor axes.
    """
    ua = 4.84e-6  # Astronomical units

    log = Log()
    a_in = log.random_sample(0.01, parallax, size=1)[0]  # in UA
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            P = np.sqrt(a_in**3 / (inner_binary_mass))
        except Warning:
            P = np.sqrt((-1 * a_in) ** 3 / (inner_binary_mass))
    while P < 3:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                a_in = log.random_sample(0.01, parallax, size=1)[0]  # in UA
                P = np.sqrt(a_in**3 / (inner_binary_mass))
            except Warning:
                P = np.sqrt((-1 * a_in) ** 3 / (inner_binary_mass))
    return a_in * ua


def get_inner_r3D(
    inner_semimajor_axis: np.ndarray,
    eccentricity: np.ndarray,
    phi: np.ndarray,
    phi_0: np.ndarray,
):
    """Compute the inner 3D separation.

    This function calculates the inner 3D separation of a binary system based on the
    inner semimajor axis, eccentricity, phase angle, and reference phase angle. It uses
    the following formula:
    inner_r3D = inner_semimajor_axis * (1 - eccentricity^2) /
    (1 + eccentricity * cos(phi - phi_0))

    Parameters:
        inner_semimajor_axis (np.ndarray):
            Array of inner semimajor axes.

        eccentricity (np.ndarray):
            Array of eccentricities.

        phi (np.ndarray):
            Array of phase angles.

        phi_0 (np.ndarray):
            Array of reference phase angles.

    Returns:
        np.ndarray: Array of computed inner 3D separations.
    """
    return (
        inner_semimajor_axis
        * (1 - eccentricity**2)
        / (1 + eccentricity * np.cos(phi - phi_0))
    )


def get_2D_projected_separation(
    separation_3D: np.ndarray, phi: np.ndarray, orbital_angle: np.ndarray
):
    """Compute the 2D projected separation.

    This function calculates the 2D projected separation of a binary system based on the
    3D separation, phase angle, and orbital angle. It uses the following formula:
    projected_separation = separation_3D * sqrt(cos(phi)^2 +
    (sin(phi)^2) * cos(orbital_angle)^2)

    Parameters:
        separation_3D (np.ndarray):
            Array of 3D separations.

        phi (np.ndarray):
            Array of phase angles.

        orbital_angle (np.ndarray):
            Array of orbital angles.

    Returns:
        np.ndarray: Array of computed 2D projected separations.
    """
    return separation_3D * np.sqrt(
        np.cos(phi) ** 2 + (np.sin(phi) ** 2) * np.cos(orbital_angle) ** 2
    )


def get_r3D(separation_2D: np.ndarray, phi: np.ndarray, orbital_angle: np.ndarray):
    """Compute the 3D separation.

    This function calculates the 3D separation of a binary system based on the 2D
    separation, phase angle, and orbital angle. It uses the following formula:
    r_3D = separation_2D / sqrt((cos(phi)^2) + (cos(orbital_angle)^2) * (sin(phi)^2))

    Parameters:
        separation_2D (np.ndarray):
            Array of 2D separations.

        phi (np.ndarray):
            Array of phase angles.

        orbital_angle (np.ndarray):
            Array of orbital angles.

    Returns:
        np.ndarray: Array of computed 3D separations.
    """
    return separation_2D / np.sqrt(
        (np.cos(phi) ** 2) + (np.cos(orbital_angle) ** 2) * (np.sin(phi) ** 2)
    )
