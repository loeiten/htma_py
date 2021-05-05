"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx."""

from typing import Dict

import pandas as pd

from htma_py.continuous_evpi import calculate_evpi, get_lin_payoff_and_loss, print_risk
from htma_py.distribution import Distribution, Gaussian, get_samples
from htma_py.plot import plot_histogram
from htma_py.utils.paths import get_plot_path


def plot_units_histogram(samples_df: pd.DataFrame, threshold: float) -> None:
    """
    Plot the units histogram.

    Parameters
    ----------
    samples_df : pd.DataFrame
        The dataframes containing the samples
    threshold: float
        The threshold for the units
    """
    fig, axis = plot_histogram(samples_df.loc[:, "Units"], "Units")
    # Mark the threshold
    axis.axvline(x=threshold, linestyle="--", color="r", ymin=0, ymax=1)
    save_path = get_plot_path().joinpath("risk_units_sold.png").absolute()
    # Save
    fig.savefig(
        save_path,
        transparent=True,
        bbox_inches="tight",
    )
    print(f"Saved plot to {save_path}")


def main() -> None:
    """Calculate the EVPI for the units example."""
    price_per_unit = 25
    threshold_units = 2.0e5
    lower_90_ci = 1.5e5
    upper_90_ci = 3.0e5

    # Note: These numbers are used in teh excel example
    units_min = 0
    units_max = 452963
    n_points_lin_array = int(2e3)

    # Calculate payoff parameters
    threshold_payoff = price_per_unit * threshold_units

    # Obtain the linear arrays
    lin_payoff_array, lin_loss_array = get_lin_payoff_and_loss(
        units_min * price_per_unit,
        units_max * price_per_unit,
        threshold_payoff,
        n_points_lin_array,
    )

    # Create the distributions
    units_gauss_dist = Gaussian(lower_90_ci, upper_90_ci)
    payoff_gauss_dist = Gaussian(
        lower_90_ci * price_per_unit, upper_90_ci * price_per_unit
    )
    # NOTE: We are specifying type in order not to get mypy invariant/covariant error
    distributions: Dict[str, Distribution] = {
        "Units": units_gauss_dist,
        "Payoff": payoff_gauss_dist,
    }

    # Get samples
    # NOTE: We are just choosing n_points_lin_array out of laziness.
    #       We could have used any other number
    samples_df = get_samples(distributions, n_samples=n_points_lin_array)

    print(
        "Samples from 'Units' distribution "
        "(we are letting units be floats for simplicity):"
    )
    # NOTE: We have sampled "Units" and "Payoff" individually, so it would be
    #       misleading to print them together
    # NOTE: When printing to ipython, it's better to print using .style
    #       https://stackoverflow.com/a/46370761/2786884
    print(samples_df.loc[:, "Units"], end="\n" * 2)

    # NOTE: One could also state the risk from the variable itself like so:
    #       print_risk(samples_df.loc[:, "Units"], threshold_units)
    print_risk(samples_df.loc[:, "Payoff"], threshold_payoff)

    # NOTE: One could also calculate the EVPI from the variable like so:
    #       lin_units_array = np.linspace(units_min, units_max, n_points_lin_array)
    #       calculate_evpi(distributions["Units"], lin_loss_array, lin_units_array)
    #       However, when dealing with multiple variables, estimating EVPI from the
    #       payoff is the simplest
    evpi = calculate_evpi(distributions["Payoff"], lin_payoff_array, lin_loss_array)
    print(f"Expected value of perfect information for 'Payoff' variable: {evpi:.1f}")

    plot_units_histogram(samples_df, threshold_units)


if __name__ == "__main__":
    main()
