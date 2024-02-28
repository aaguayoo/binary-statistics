"""Hidden Tertiaries class."""
from typing import Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from binaryStatistics.chae_velocity_deconstruction.chae_velocity_reconstruction import (
    ChaeVelocityDeconstruction,
)
from binaryStatistics.constants import G_pc_km_s_Modot
from binaryStatistics.hidden_tertiaries.utils import set_hidden_companions_distribution
from binaryStatistics.schemas import Config


@dataclass(config=Config)
class HiddenTertiaries:
    """HiddenTertiaries class.

    Attributes:
        dataframe (Optional[pd.DataFrame]):
            Optional pandas DataFrame containing the data.

        hidden_tertiares_fraction (Optional[float]):
            Optional fraction of hidden tertiaries. Default is 0.0.

    Methods:
        __post_init_post_parse__: Post initialization section.
        show_hidden_companions_distribution: Prints the distribution of hidden
        companions.
        compute_binaries_Chae_2D_velocity_deconstruction: Computes the Chae 2D velocity
        deconstruction for the binaries.
        compute_hidden_Chae_2D_velocity_deconstruction: Computes the Chae 2D velocity
        deconstruction for the hidden companions.
        compute_binaries_Hernandez_v_tilde: Computes the Hernandez v_tilde distribution
        for the binaries.
        compute_hidden_Hernandez_v_tilde: Computes the Hernandez v_tilde distribution
        for the hidden companions.
        compute_velocity_deconstruction: Computes the velocity deconstruction.
        compute_2D_projected_relative_velocity: Computes the 2D projected relative
        velocity.

    Returns:
        pd.DataFrame: The computed dataframe.
    """

    dataframe: Optional[pd.DataFrame]
    hidden_tertiares_fraction: Optional[float] = 0.0

    def __post_init_post_parse__(self):
        """
        Post initialization section.

        This method is called after the initialization of the HiddenTertiaries class.
        It performs the following tasks:
        - Makes a copy of the DataFrame.
        - Sets the hidden companions distribution in the DataFrame using the
        set_hidden_companions_distribution function.
        - Initializes instances of the ChaeVelocityDeconstruction class for the outer
        binary and the hidden companions.
        """
        self.dataframe = self.dataframe.copy()
        self.dataframe = set_hidden_companions_distribution(
            dataframe=self.dataframe, frac=self.hidden_tertiares_fraction
        )
        self.chae_outer = ChaeVelocityDeconstruction(dataframe=self.dataframe)
        self.chae_A = ChaeVelocityDeconstruction(
            dataframe=self.dataframe, hidden_companion_tag="_A"
        )
        self.chae_B = ChaeVelocityDeconstruction(
            dataframe=self.dataframe, hidden_companion_tag="_B"
        )

    def show_hidden_companions_distribution(self):
        """Print the distribution of hidden companions.

        This method prints the distribution of hidden companions in the DataFrame. It
        counts the occurrences of each companion tag and displays the result.
        """
        print(self.dataframe["companion_tag"].value_counts())

    def compute_binaries_Chae_2D_velocity_deconstruction(self):
        """Compute the Chae 2D velocity deconstruction for the binaries.

        This method computes the Chae 2D velocity deconstruction for the binaries by
        calling the compute_velocity_deconstruction method of the
        ChaeVelocityDeconstruction class for the outer binary.

        Returns:
            pd.DataFrame: The computed dataframe.
        """
        self.dataframe = self.chae_outer.compute_velocity_deconstruction()

    def compute_hidden_Chae_2D_velocity_deconstruction(self):
        """Compute the Chae 2D velocity deconstruction for the hidden companions.

        This method computes the Chae 2D velocity deconstruction for the hidden
        companions by calling the compute_velocity_deconstruction method of the
        ChaeVelocityDeconstruction class for each hidden companion.

        Returns:
            pd.DataFrame: The computed dataframe.
        """
        self.dataframe = self.chae_A.compute_velocity_deconstruction()
        self.dataframe = self.chae_B.compute_velocity_deconstruction()

    def compute_binaries_Hernandez_v_tilde(self, plotDistribution: bool = False):
        """Compute the Hernandez v_tilde distribution for the binaries.

        This method computes the Hernandez v_tilde distribution for the binaries by
        calling the compute_v_tilde_distribution method of the
        ChaeVelocityDeconstruction class for the outer binary. If plotDistribution is
        True, it also plots the distribution.

        Parameters:
            plotDistribution (bool, optional):
                Whether to plot the distribution. Default is False.
        """
        self.dataframe = self.chae_outer.orbital.compute_v_tilde_distribution()
        if plotDistribution:
            self.chae_outer.orbital.plot_v_tilde_distribution(
                plot_title="$\\tilde{v}$ distribution for outer binary"
            )

    def compute_hidden_Hernandez_v_tilde(self, plotDistribution: bool = False):
        """Compute the Hernandez v_tilde distribution for the hidden companions.

        This method computes the Hernandez v_tilde distribution for the hidden
        companions by calling the compute_v_tilde_distribution method of the
        ChaeVelocityDeconstruction class for each hidden companion. If plotDistribution
        is True, it also plots the distribution for each hidden companion.

        Parameters:
            plotDistribution (bool, optional):
                Whether to plot the distribution. Default is False.
        """
        self.dataframe = self.chae_A.orbital.compute_v_tilde_distribution()
        self.dataframe = self.chae_B.orbital.compute_v_tilde_distribution()
        if plotDistribution:
            self.chae_A.orbital.plot_v_tilde_distribution(
                plot_title="$\\tilde{v}$ distribution for inner binary A"
            )
            self.chae_B.orbital.plot_v_tilde_distribution(
                plot_title="$\\tilde{v}$ distribution for inner binary B"
            )

    def compute_velocity_deconstruction(self):
        """
        Compute the velocity deconstruction.

        This method computes the velocity deconstruction by calling the
        compute_binaries_Chae_2D_velocity_deconstruction and
        compute_hidden_Chae_2D_velocity_deconstruction methods. It returns the computed
        dataframe.

        Returns:
            pd.DataFrame: The computed dataframe.
        """
        self.compute_binaries_Chae_2D_velocity_deconstruction()
        self.compute_hidden_Chae_2D_velocity_deconstruction()

        return self.dataframe

    def compute_2D_projected_relative_velocity(self):
        """
        Compute the 2D projected relative velocity.

        This method computes the 2D projected relative velocity for the hidden
        companions. It calculates the velocity components in the x and y directions by
        summing the contributions from each hidden companion. The computed 2D projected
        relative velocity is added to the DataFrame.
        """
        G = G_pc_km_s_Modot
        self.dataframe["VEL_2D_x"] = self.dataframe.apply(
            lambda x: x.v2D_x + x.eta_A * x.v2D_x_A + x.eta_B * x.v2D_x_B, axis=1
        )
        self.dataframe["VEL_2D_y"] = self.dataframe.apply(
            lambda x: x.v2D_y + x.eta_A * x.v2D_y_A + x.eta_B * x.v2D_y_B, axis=1
        )
        self.dataframe["VEL_2D_Chae"] = self.dataframe.apply(
            lambda x: np.sqrt(x.VEL_2D_x**2 + x.VEL_2D_y**2), axis=1
        )
        self.dataframe["v_tilde_Chae"] = self.dataframe.apply(
            lambda x: x.VEL_2D_Chae / np.sqrt(G * x.M / x.separation), axis=1
        )
