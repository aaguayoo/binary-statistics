"""Hidden tertiaries utilities."""
import numpy as np
import pandas as pd

# Function for simulate hidden companion mass
from binaryStatistics.orbital_distributions.distributions import PowerLaw


def get_companions_mass(dataframe: pd.DataFrame):
    """
    Compute the masses of the companions.

    This function calculates the masses of the companions in a binary system based on
    the magnitudes of the primary and secondary components. It uses a power-law
    distribution to generate a value for kappa, which represents the fraction of the
    total mass assigned to the primary component. The function then calculates the
    masses of the primary and secondary components based on the assigned kappa value
    and the magnitudes. The computed masses are returned as a dictionary.

    Parameters:
        dataframe (pd.DataFrame):
            DataFrame containing the magnitudes of the primary and secondary components.

    Returns:
        dict: Dictionary containing the computed masses of the primary and secondary
        components.
    """

    def get_kappa(delta_Mag):
        dM = delta_Mag.random_sample(29, 30, 1)
        return 1 / (1 + 10 ** (-0.4 * dM))[0]

    def get_mass(mag):
        return 10 ** (0.0725 * (4.76 - mag))

    Mag1 = dataframe.MAG1
    Mag2 = dataframe.MAG2

    gamma = -0.6
    delta_Mag = PowerLaw(
        dist_parameters={
            "C": (1 + gamma) / (12**gamma),
            "alpha": gamma,
        }
    )
    kappa = get_kappa(delta_Mag)

    if dataframe.companion_tag == "without_companion":
        return {
            "mass_A_1": get_mass(Mag1),
            "mass_A_2": 0.0,
            "mass_B_1": get_mass(Mag2),
            "mass_B_2": 0.0,
        }
    elif dataframe.companion_one == "more":
        if Mag1 > Mag2:
            host = -2.5 * np.log10(kappa) + Mag1
            comp = -2.5 * np.log10(1 - kappa) + Mag1

            return {
                "mass_A_1": get_mass(host),
                "mass_A_2": get_mass(comp),
                "mass_B_1": get_mass(Mag2),
                "mass_B_2": 0.0,
            }
        else:
            host = -2.5 * np.log10(kappa) + Mag2
            comp = -2.5 * np.log10(1 - kappa) + Mag2

            return {
                "mass_A_1": get_mass(Mag1),
                "mass_A_2": 0.0,
                "mass_B_1": get_mass(host),
                "mass_B_2": get_mass(comp),
            }

    elif dataframe.companion_one == "less":
        if Mag1 < Mag2:
            host = -2.5 * np.log10(kappa) + Mag1
            comp = -2.5 * np.log10(1 - kappa) + Mag1

            return {
                "mass_A_1": get_mass(host),
                "mass_A_2": get_mass(comp),
                "mass_B_1": get_mass(Mag2),
                "mass_B_2": 0.0,
            }
        else:
            host = -2.5 * np.log10(kappa) + Mag2
            comp = -2.5 * np.log10(1 - kappa) + Mag2

            return {
                "mass_A_1": get_mass(Mag2),
                "mass_A_2": 0.0,
                "mass_B_1": get_mass(host),
                "mass_B_2": get_mass(comp),
            }

    elif dataframe.companion_both == "both":
        kappa = get_kappa(delta_Mag)
        host_1 = -2.5 * np.log10(kappa) + Mag1
        comp_1 = -2.5 * np.log10(1 - kappa) + Mag1
        kappa = get_kappa(delta_Mag)
        host_2 = -2.5 * np.log10(kappa) + Mag2
        comp_2 = -2.5 * np.log10(1 - kappa) + Mag2

        return {
            "mass_A_1": get_mass(host_1),
            "mass_A_2": get_mass(comp_1),
            "mass_B_1": get_mass(host_2),
            "mass_B_2": get_mass(comp_2),
        }


def set_hidden_companions_distribution(dataframe: pd.DataFrame, frac: float):
    """Set the hidden companions distribution.

    This function sets the distribution of hidden companions in the binary system. It
    assigns a companion tag to each row in the DataFrame based on a given fraction. The
    companion tag can be either "with_companion" or "without_companion". If the
    companion tag is "with_companion", it further assigns a value to the companion_both
    and companion_one columns based on random choices. The function then calculates the
    masses of the primary and secondary components using the get_companions_mass
    function. Other columns in the DataFrame are modified accordingly.

    Parameters:
        dataframe (pd.DataFrame):
            DataFrame containing the data.

        frac (float):
            Fraction of rows with hidden companions.

    Returns:
        pd.DataFrame: Modified DataFrame with the hidden companions distribution set.
    """
    dataframe["companion_tag"] = dataframe.apply(
        lambda x: np.random.choice(
            ["with_companion", "without_companion"], p=[frac, 1 - frac]
        ),
        axis=1,
    )
    dataframe["companion_both"] = dataframe.apply(
        lambda x: np.random.choice(["one", "both"], p=[0.7, 0.3])
        if x.companion_tag == "with_companion"
        else None,
        axis=1,
    )
    dataframe["companion_one"] = dataframe.apply(
        lambda x: np.random.choice(["more", "less"], p=[0.4 / 0.7, 0.3 / 0.7])
        if x.companion_both == "one"
        else None,
        axis=1,
    )

    dataframe[["mass_A_1", "mass_A_2", "mass_B_1", "mass_B_2"]] = dataframe.apply(
        lambda x: get_companions_mass(x), axis=1, result_type="expand"
    )

    dataframe["vDECerr"] = dataframe["vDECerr"].apply(
        lambda x: float(x.split("(")[-1].split(",)")[0])
    )
    dataframe["d_V2D"] = (dataframe["vRA"] / dataframe["V2D"]) * dataframe["vRAerr"] + (
        dataframe["vDEC"] / dataframe["V2D"]
    ) * dataframe["vDECerr"]
    dataframe["PARALLAX_A"] = dataframe["PARALLAX1"]
    dataframe["PARALLAX_B"] = dataframe["PARALLAX2"]
    dataframe["separation"] = dataframe["r"]
    dataframe["M_A"] = dataframe["mass_A_1"] + dataframe["mass_A_2"]
    dataframe["M_B"] = dataframe["mass_B_1"] + dataframe["mass_B_2"]
    dataframe["old_mass"] = dataframe["M"]
    dataframe["M"] = (
        dataframe["mass_A_1"]
        + dataframe["mass_A_2"]
        + dataframe["mass_B_1"]
        + dataframe["mass_B_2"]
    )

    return dataframe
