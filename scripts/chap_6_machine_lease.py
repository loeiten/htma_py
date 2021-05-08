"""
Replication of HTMA_3rd_Ch6-2.xlsx and calculation of EVPI for each variable.

Notes
-----
This example contains several variables.
It could therefore be didactical to have a look at chap_7_units_production.py
which goes through an example of only one variable.
"""

from typing import Dict, Union

import numpy as np
import pandas as pd

from htma_py.continuous_evpi import (
    get_eol_from_distribution,
    get_evpi_from_samples,
    get_lin_revenue_and_loss,
    print_risk,
)
from htma_py.distribution import (
    Distribution,
    DistributionFromSamples,
    Gaussian,
    get_samples,
)
from htma_py.htma_plots.plots import (
    plot_eol_with_threshold,
    plot_loss_functions,
    plot_pdf_cdf_and_incremental_probability,
    plot_sample_histogram_with_threshold,
)


def main() -> None:
    """Calculate the risk and EVPI for the machine lease example."""
    n_points_lin_array = int(2e3)
    n_samples = int(2e3)
    threshold_payoff = 4e5

    # Specify the distributions
    # NOTE: We are specifying type in order not to get mypy invariant/covariant error
    distributions: Dict[str, Distribution] = {
        "Maintenance savings": Gaussian(10, 20),
        "Labor savings": Gaussian(-2, 8),
        "Raw materials savings": Gaussian(3, 9),
        "Production level": Gaussian(1.5e4, 3.5e4),
    }

    # Sample and calculate the annual savings using Monte Carlo
    # (i.e. perform arithmetics on the samples)
    samples_df = get_samples(distributions, n_samples=n_samples)
    samples_df.loc[:, "Annual savings"] = calculate_annual_savings(
        samples_df.loc[:, "Maintenance savings"],
        samples_df.loc[:, "Labor savings"],
        samples_df.loc[:, "Raw materials savings"],
        samples_df.loc[:, "Production level"],
    )
    print(
        "Samples from the distributions "
        "(we are letting all variables be floats for simplicity):"
    )
    # NOTE: When printing to ipython, it's better to print using .style
    #       https://stackoverflow.com/a/46370761/2786884
    pd.set_option("display.max_columns", 6)
    print(samples_df, end="\n" * 2)

    # Print risk
    print_risk(samples_df.loc[:, "Annual savings"], threshold_payoff)

    # Calculate the overall EVPI
    overall_evpi = get_evpi_from_samples(
        samples_df.loc[:, "Annual savings"], threshold_payoff, n_points_lin_array, 0
    )
    print(
        f"The overall EVPI is {overall_evpi:.0f}\tOverall threshold: "
        f"{threshold_payoff:.0f}"
    )

    # Print EVPI
    print_method_1_evpi(distributions, samples_df, threshold_payoff, n_points_lin_array)
    print_method_2_evpi(
        overall_evpi, distributions, samples_df, threshold_payoff, n_points_lin_array
    )

    # Obtain the linear arrays
    lin_revenue_array, lin_loss_array = get_lin_revenue_and_loss(
        samples_df.loc[:, "Annual savings"].min(),
        samples_df.loc[:, "Annual savings"].max(),
        threshold_payoff,
        n_points_lin_array,
    )

    # Plot
    annual_savings_dist = DistributionFromSamples(
        samples_df.loc[:, "Annual savings"], min_x_value=0
    )
    plot_sample_histogram_with_threshold(
        samples_df.loc[:, "Annual savings"],
        threshold_payoff,
        "Annual savings",
        "$",
        "machine_lease",
    )
    plot_loss_functions(
        lin_revenue_array, lin_loss_array, threshold_payoff, "machine_lease"
    )
    plot_pdf_cdf_and_incremental_probability(
        annual_savings_dist, lin_revenue_array, threshold_payoff, "machine_lease"
    )
    plot_eol_with_threshold(
        get_eol_from_distribution(
            annual_savings_dist, lin_revenue_array, lin_loss_array
        ),
        lin_revenue_array,
        threshold_payoff,
        "machine_lease",
    )


def calculate_annual_savings(
    maintenance_savings: Union[float, np.array],
    labor_savings: Union[float, np.array],
    raw_material_savings: Union[float, np.array],
    production_level: Union[float, np.array],
) -> np.array:
    """
    Calculate the annual savings.

    Parameters
    ----------
    maintenance_savings : float or np.array
        The maintenance savings
    labor_savings : float or np.array
        The labor savings
    raw_material_savings : float or np.array
        The raw material savings
    production_level : float or np.array
        The production level

    Returns
    -------
    annual_savings : float or np.array
        The annual savings
    """
    annual_savings: np.array = (
        maintenance_savings + labor_savings + raw_material_savings
    ) * production_level
    return annual_savings


def print_method_1_evpi(
    distributions: Dict[str, Distribution],
    samples_df: pd.DataFrame,
    threshold_payoff: float,
    n_points_lin_array: int,
) -> None:
    """
    Calculate EVPI using method 1.

    In method 1 all variables are fixed to the mean and only one variable is allowed
    to change (i.e. be sampled)
    Note that this method is less precise than method 2.

    Parameters
    ----------
    distributions : dict of str, Distribution
        The distributions of the variables
    samples_df : pd.DataFrame
        The samples of the variables
    threshold_payoff : float
        The payoff threshold
    n_points_lin_array : int
        Number of points used for calculation of EVPI
    """
    print(
        "\n" + "=" * 80 + "\nUsing method 1 where all variables are kept at the mean:"
    )
    for evpi_variable_name in samples_df.columns.values:
        if evpi_variable_name == "Annual savings":
            continue

        var_dict = dict()

        for variable_name in samples_df.columns.values:
            if variable_name != "Annual savings":
                if variable_name != evpi_variable_name:
                    var_dict[variable_name] = distributions[variable_name].mean
                else:
                    var_dict[variable_name] = samples_df.loc[:, variable_name]

        annual_savings = calculate_annual_savings(
            var_dict["Maintenance savings"],
            var_dict["Labor savings"],
            var_dict["Raw materials savings"],
            var_dict["Production level"],
        )
        evpi = get_evpi_from_samples(
            annual_savings, threshold_payoff, n_points_lin_array, 0
        )
        print(f"'{evpi_variable_name}' EVPI is {evpi:.0f}")


def print_method_2_evpi(
    overall_evpi: float,
    distributions: Dict[str, Distribution],
    samples_df: pd.DataFrame,
    threshold_payoff: float,
    n_points_lin_array: int,
) -> None:
    """
    Calculate EVPI using method 2.

    In method 2 one variable is fixed to the mean and all other variables are allowed
    to change (i.e. be sampled)
    The difference between the overall EVPI and the newly calculated EVPI is denoted
    the individual EVPI of that variable
    Note that this method is more precise than method 1.

    Parameters
    ----------
    overall_evpi : float
        The overall EVPI of the system
    distributions : dict of str, Distribution
        The distributions of the variables
    samples_df : pd.DataFrame
        The samples of the variables
    threshold_payoff : float
        The payoff threshold
    n_points_lin_array : int
        Number of points used for calculation of EVPI
    """
    print(
        "\n"
        + "=" * 80
        + "\nUsing method 2 where only one variable is assumed known (kept at mean):"
    )
    for evpi_variable_name in samples_df.columns.values:
        if evpi_variable_name == "Annual savings":
            continue

        var_dict = dict()

        for variable_name in samples_df.columns.values:
            if variable_name != "Annual savings":
                if variable_name != evpi_variable_name:
                    var_dict[variable_name] = samples_df.loc[:, variable_name]
                else:
                    var_dict[variable_name] = distributions[variable_name].mean

        annual_savings = calculate_annual_savings(
            var_dict["Maintenance savings"],
            var_dict["Labor savings"],
            var_dict["Raw materials savings"],
            var_dict["Production level"],
        )
        evpi = overall_evpi - get_evpi_from_samples(
            annual_savings, threshold_payoff, n_points_lin_array, 0
        )
        print(f"Individual '{evpi_variable_name}' EVPI is {evpi:.0f}")

    print()


if __name__ == "__main__":
    main()
