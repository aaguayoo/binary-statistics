"""Velocity calculation."""
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic.dataclasses import dataclass

from binaryStatistics.constants import G_pc_km_s_Modot
from binaryStatistics.orbital_distributions.distributions import (
    Sine,
    Thermal,
    Uniform,
    VelTilde,
)
from binaryStatistics.orbital_distributions.utils import get_phi_angle
from binaryStatistics.schemas import Config


@dataclass(config=Config)
class HernandezOrbitalDistributions:
    """Class to compute v-tilde according to Hernandez (2023)"""

    csv_file: str = None
    dataframe: pd.DataFrame = None
    v_tilde_fit_data: Optional[str] = "../data/VTil.dat"
    hidden_companion_tag: Optional[str] = ""

    def __post_init_post_parse__(self):
        """Post init section."""
        if self.csv_file:
            self.dataframe = pd.read_csv(self.csv_file)

        # Initialize distributions
        self.thermal = Thermal()
        self.sine = Sine()
        self.uniform = Uniform()
        self.v_tilde = VelTilde()

    def __compute_eccentricity_distribution(self):
        """Compute eccentricity distribution."""
        self.dataframe[
            "eccentricity" + self.hidden_companion_tag
        ] = self.dataframe.apply(
            lambda x: self.thermal.random_sample(0, 1, size=1)[0], axis=1
        )

    def __compute_i_angle_distribution(self):
        """Compute i-angle distribution."""
        self.dataframe["i_angle" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: self.sine.random_sample(0, np.pi / 2, size=1)[0], axis=1
        )

    def __compute_phi_distribution(self):
        """Get phi distribution."""
        self.dataframe["phi" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: get_phi_angle(
                x["eccentricity" + self.hidden_companion_tag],
                x["phi_0" + self.hidden_companion_tag],
            ),
            axis=1,
        )

    def __compute_phi_0_distribution(self):
        """Compute phi-0 distribution."""
        self.dataframe["phi_0" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: self.uniform.random_sample(0, 2 * np.pi, size=1)[0], axis=1
        )

    def compute_orbital_distributions(self):
        """Compute $\\tilde{v}$ according to Xavier (2023)."""
        self.__compute_eccentricity_distribution()
        self.__compute_i_angle_distribution()
        self.__compute_phi_0_distribution()
        self.__compute_phi_distribution()

        return self.dataframe

    def compute_v_tilde_distribution(self):
        """Compute the v_tilde distribution.

        This method computes the v_tilde distribution for the binary system. It
        calculates the v_tilde values based on the phase angle, reference phase angle,
        inclination angle, and eccentricity. The v_tilde values are then used to compute
        the 2D velocity components. The computed values are added to the DataFrame.

        Returns:
            pd.DataFrame: The modified DataFrame with the v_tilde distribution and 2D
            velocity components.
        """
        G = G_pc_km_s_Modot
        self.dataframe[
            "v_tilde_Hernandez" + self.hidden_companion_tag
        ] = self.dataframe.apply(
            lambda x: self.v_tilde.distribution(
                x["phi" + self.hidden_companion_tag]
                - x["phi_0" + self.hidden_companion_tag],
                x["phi_0" + self.hidden_companion_tag],
                x["i_angle" + self.hidden_companion_tag],
                x["eccentricity" + self.hidden_companion_tag],
            ),
            axis=1,
        )
        self.dataframe[
            "VEL_2D_Hernandez" + self.hidden_companion_tag
        ] = self.dataframe.apply(
            lambda x: x["v_tilde_Hernandez" + self.hidden_companion_tag]
            * np.sqrt(
                G
                * x["M" + self.hidden_companion_tag]
                / x["separation" + self.hidden_companion_tag]
            ),
            axis=1,
        )
        return self.dataframe

    def plot_v_tilde_distribution(self, withFit: bool = True, plot_title: str = ""):
        """Plot the v_tilde distribution.

        This method plots the v_tilde distribution of the binary system. It creates a
        histogram of the v_tilde values and optionally overlays a fit curve based on the
        Hernandez (2023) fit data. The plot can be customized with a title and the
        option to include the fit curve.

        Parameters:
            withFit (bool, optional):
                Whether to include the fit curve. Default is True.

            plot_title (str, optional):
                Title of the plot. Default is an empty string.
        """
        _, ax = plt.subplots()
        _ = ax.hist(
            self.dataframe["v_tilde_Hernandez" + self.hidden_companion_tag],
            bins=50,
            density=True,
        )
        if withFit:
            with open(self.v_tilde_fit_data, "r") as file:
                lines = file.readlines()

                v = []
                v_dist = []
                for line in lines:
                    vel, dist = line.strip().split(" ")
                    v.append(float(vel))
                    v_dist.append(float(dist))
            ax.plot(v, v_dist, label="Hernandez (2023) fit")
            ax.set_xlim(0, np.sqrt(2))
            ax.set_title(plot_title)
            ax.set_xlabel("$\\tilde{v}(e,i,\\phi,\\phi_0)$")
            plt.legend()
