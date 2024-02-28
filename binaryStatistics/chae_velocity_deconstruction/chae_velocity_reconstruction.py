"""Hidden tertiaries companions."""
from typing import Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from binaryStatistics.chae_velocity_deconstruction.utils import (
    get_2D_projected_separation,
    get_inner_r3D,
    get_inner_semimajor_axis,
    get_photometric_distance,
    get_r3D,
    get_relative_v3D,
    get_semimajor_axis,
)
from binaryStatistics.orbital_distributions.hernandez_orbital_distributions import (
    HernandezOrbitalDistributions,
)
from binaryStatistics.schemas import Config


@dataclass(config=Config)
class ChaeVelocityDeconstruction:
    """ChaeVelocityDeconstruction class.

    This class implements the stellar binary velocity deconstruction algorithm described
    in Chae (2023).

    Attributes:
        dataframe (pd.DataFrame):
            Optional pandas DataFrame containing the data. Default is None.

        hidden_companion_tag (str):
            Optional string representing the hidden companion tag. Default is "".

    Methods:
        compute_velocity_deconstruction: Computes the velocity deconstruction.
        __compute_2D_projected_separation: Computes the 2D projected separation.
        __compute_3D_relative_velocity: Computes the 3D relative velocity.
        __compute_2D_velocity_components: Computes the 2D velocity components.
        __compute_photometric_distance: Computes the photometric distance.
        __compute_semimajor_axis: Computes the semimajor axis.

    Returns:
        pandas DataFrame: The computed dataframe.

    Examples:
        # Create an instance of ChaeVelocityDeconstruction
        chae = ChaeVelocityDeconstruction(dataframe=df, hidden_companion_tag="hidden")

        # Compute the velocity deconstruction
        result = chae.compute_velocity_deconstruction()
    """

    dataframe: Optional[pd.DataFrame] = None
    hidden_companion_tag: Optional[str] = ""

    def __post_init_post_parse__(self):
        """Post init section.

        This method is called when the ChaeVelocityDeconstruction object is initialized,
        and performs the computation of the binary orbital distributions of
        $(e, i, \\phi, \\phi_0)$, as computed by Hernandez et al. (2023).
        """
        self.orbital = HernandezOrbitalDistributions(
            dataframe=self.dataframe, hidden_companion_tag=self.hidden_companion_tag
        )
        self.dataframe = self.orbital.compute_orbital_distributions()

    def __compute_2D_velocity_components(self):
        self.dataframe["v2D_x" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: -x["v_3D" + self.hidden_companion_tag]
            * np.sin(x["phi" + self.hidden_companion_tag]),
            axis=1,
        )
        self.dataframe["v2D_y" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: -x["v_3D" + self.hidden_companion_tag]
            * np.cos(x["phi" + self.hidden_companion_tag])
            * np.cos(x["i_angle" + self.hidden_companion_tag]),
            axis=1,
        )

    def __compute_3D_relative_velocity(self):
        self.dataframe["v_3D" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: get_relative_v3D(
                x["M" + self.hidden_companion_tag],
                x["r_3D" + self.hidden_companion_tag],
                x["semimajor_axis" + self.hidden_companion_tag],
            ),
            axis=1,
        )

    def __compute_photometric_distance(self):
        self.dataframe["eta" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: get_photometric_distance(
                x["mass" + self.hidden_companion_tag + "_1"],
                x["mass" + self.hidden_companion_tag + "_2"],
                x["semimajor_axis" + self.hidden_companion_tag],
                x["semimajor_axis"],
            ),
            axis=1,
        )

    def __compute_2D_projected_separation(self):
        self.dataframe["separation" + self.hidden_companion_tag] = self.dataframe.apply(
            lambda x: get_2D_projected_separation(
                x["r_3D" + self.hidden_companion_tag],
                x["phi" + self.hidden_companion_tag],
                x["i_angle" + self.hidden_companion_tag],
            ),
            axis=1,
        )

    def __compute_semimajor_axis(self):
        if self.hidden_companion_tag:
            self.dataframe[
                "semimajor_axis" + self.hidden_companion_tag
            ] = self.dataframe.apply(
                lambda x: get_inner_semimajor_axis(
                    x["PARALLAX" + self.hidden_companion_tag],
                    x["M" + self.hidden_companion_tag],
                ),
                axis=1,
            )
            self.dataframe["r_3D" + self.hidden_companion_tag] = self.dataframe.apply(
                lambda x: get_inner_r3D(
                    x["semimajor_axis" + self.hidden_companion_tag],
                    x["eccentricity" + self.hidden_companion_tag],
                    x["phi" + self.hidden_companion_tag],
                    x["phi_0" + self.hidden_companion_tag],
                ),
                axis=1,
            )
        else:
            self.dataframe["r_3D"] = self.dataframe.apply(
                lambda x: get_r3D(x.separation, x.phi, x.i_angle),
                axis=1,
            )
            self.dataframe["semimajor_axis"] = self.dataframe.apply(
                lambda x: get_semimajor_axis(
                    x.r_3D,
                    x.eccentricity,
                    x.phi,
                    x.phi_0,
                ),
                axis=1,
            )

    def compute_velocity_deconstruction(self):
        """Compute the velocity deconstruction.

        This method computes the velocity deconstruction by performing the following
        steps:
            1. Computes the semimajor axis.
            2. Computes the 3D relative velocity.
            3. Computes the 2D velocity components.
            4. If a hidden companion tag is provided, computes the 2D projected
            separation and the photometric distance.

        Returns:
            pd.DataFrame: The computed dataframe.
        """
        self.__compute_semimajor_axis()
        self.__compute_3D_relative_velocity()
        self.__compute_2D_velocity_components()
        if self.hidden_companion_tag:
            self.__compute_2D_projected_separation()
            self.__compute_photometric_distance()

        return self.dataframe
