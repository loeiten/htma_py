"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx."""

from typing import Dict

import numpy as np
from matplotlib.patches import Rectangle

from htma_py.continuous_evpi import (
    calculate_evpi,
    get_eol_from_distribution,
    get_lin_revenue_and_loss,
    print_risk,
)
from htma_py.distribution import Distribution, Gaussian, get_samples
from htma_py.plots import make_legend, plot_bar, plot_histogram, plot_loss, save_plot


def main() -> None:
    """Calculate the risk and EVPI for the units example."""
    price_per_unit = 25
    threshold_units = 2.0e5
    lower_90_ci = 1.5e5
    upper_90_ci = 3.0e5

    # Note: These numbers are used in teh excel example
    units_min = 0
    units_max = 452963
    n_points_lin_array = int(2e3)

    # Calculate revenue parameters
    threshold_payoff = price_per_unit * threshold_units

    # Obtain the linear arrays
    lin_revenue_array, lin_loss_array = get_lin_revenue_and_loss(
        units_min * price_per_unit,
        units_max * price_per_unit,
        threshold_payoff,
        n_points_lin_array,
    )

    # Create the distributions
    units_gauss_dist = Gaussian(lower_90_ci, upper_90_ci)
    revenue_gauss_dist = Gaussian(
        lower_90_ci * price_per_unit, upper_90_ci * price_per_unit
    )
    # NOTE: We are specifying type in order not to get mypy invariant/covariant error
    distributions: Dict[str, Distribution] = {
        "Units": units_gauss_dist,
        "Revenue": revenue_gauss_dist,
    }

    # Get samples
    # NOTE: We are just choosing n_points_lin_array out of laziness.
    #       We could have used any other number
    samples_df = get_samples(distributions, n_samples=n_points_lin_array)

    print(
        "Samples from 'Units' distribution "
        "(we are letting units be floats for simplicity):"
    )
    # NOTE: We have sampled "Units" and "Revenue" individually, so it would be
    #       misleading to print them together
    # NOTE: When printing to ipython, it's better to print using .style
    #       https://stackoverflow.com/a/46370761/2786884
    print(samples_df.loc[:, "Units"], end="\n" * 2)

    # NOTE: One could also state the risk from the variable itself like so:
    #       print_risk(samples_df.loc[:, "Units"], threshold_units)
    print_risk(samples_df.loc[:, "Revenue"], threshold_payoff)

    # NOTE: One could also calculate the EVPI from the variable like so:
    #       lin_units_array = np.linspace(units_min, units_max, n_points_lin_array)
    #       calculate_evpi(distributions["Units"], lin_loss_array, lin_units_array)
    #       However, when dealing with multiple variables, estimating EVPI from the
    #       revenue is the simplest
    evpi = calculate_evpi(distributions["Revenue"], lin_revenue_array, lin_loss_array)
    print(f"Expected value of perfect information for 'Revenue' variable: {evpi:.1f}\n")

    plot_units_histogram(samples_df.loc[:, "Units"], threshold_units)
    plot_loss_functions(
        units_min, units_max, lin_loss_array, threshold_units, price_per_unit
    )
    plot_incremental_probability(
        distributions["Revenue"], lin_revenue_array, threshold_payoff
    )
    plot_eol(
        distributions["Revenue"], lin_revenue_array, lin_loss_array, threshold_payoff
    )


def plot_units_histogram(units_samples: np.array, threshold_units: float) -> None:
    """
    Plot the units histogram.

    Parameters
    ----------
    units_samples : np.array
        Samples of the produced units
    threshold_units: float
        The threshold for the units
    """
    fig, axis, histogram_output = plot_histogram(units_samples, "Units")
    # Mark the threshold
    # Mark probabilities with risk red
    for cur_bin, patch in zip(histogram_output["bins"], histogram_output["patches"]):
        if cur_bin < threshold_units:
            patch.set_color("red")
        else:
            patch.set_color("green")

    axis.axvline(
        x=threshold_units,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    handles, labels = axis.get_legend_handles_labels()

    # Add legend for loss
    handles.append(Rectangle((0, 0), 1, 1, color="red", alpha=0.75))
    labels.append("Loss")

    # Add legend for gain
    handles.append(Rectangle((0, 0), 1, 1, color="green", alpha=0.75))
    labels.append("Gain")

    # Make legend
    legend = axis.legend(
        handles=handles, labels=labels, loc="best", fancybox=True, numpoints=1
    )
    legend.get_frame().set_alpha(0.5)

    save_plot(fig, "units_sold_histogram.png")


def plot_loss_functions(
    x_min: float,
    x_max: float,
    lin_loss_array: np.array,
    threshold_units: float,
    price_per_unit: float,
) -> None:
    """
    Plot the loss function.

    Parameters
    ----------
    x_min : float
        The minimum value for the dependant variable
    x_max : float
        The maximum value for the dependant variable
    lin_loss_array : np.array
        The loss array
    threshold_units: float
        The threshold for the units
    price_per_unit : float
        Price per unit
    """
    lin_units_array = np.linspace(x_min, x_max, lin_loss_array.size)
    fig, axis = plot_loss(lin_units_array, lin_loss_array, "Units")
    axis.axvline(
        x=threshold_units,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, "units_sold_units_loss_function.png")

    lin_revenue_array = lin_units_array * price_per_unit
    fig, axis = plot_loss(lin_revenue_array, lin_loss_array, "Revenue [$]")
    axis.axvline(
        x=threshold_units * price_per_unit,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, "units_sold_revenue_loss_function.png")

    plot_properties = {
        "x_label": "Revenue [$]",
        "y_label": "Monetary loss [$]",
        "label": "Loss function",
        "color": "red",
        "step": 40,
        "width": 1e5,
    }
    fig, axis = plot_bar(lin_revenue_array, lin_loss_array, plot_properties)
    axis.axvline(
        x=threshold_units * price_per_unit,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, "units_sold_revenue_loss_function_bar.png")


def plot_incremental_probability(
    revenue_distribution: Distribution,
    lin_revenue_array: np.array,
    threshold_payoff: float,
) -> None:
    """
    Plot incremental probability.

    Parameters
    ----------
    revenue_distribution : Distribution
        The distribution of the revenue
    lin_revenue_array : np.array
        Linear array from min to max revenue
    threshold_payoff: float
        The threshold for the payoff
    """
    incremental_prob_revenue_array = revenue_distribution.incremental_probability(
        lin_revenue_array
    )
    plot_properties = {
        "x_label": "Revenue [$]",
        "y_label": "Incremental probability",
        "label": "IP of Revenue",
        "step": 40,
        "width": 1e5,
    }
    fig, axis = plot_bar(
        lin_revenue_array, incremental_prob_revenue_array, plot_properties
    )
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, "units_incremental_probability.png")


def plot_eol(
    revenue_distribution: Distribution,
    lin_revenue_array: np.array,
    lin_loss_array: np.array,
    threshold_payoff: float,
) -> None:
    """
    Plot expected opportunity loss.

    Parameters
    ----------
    revenue_distribution : Distribution
        The distribution of the revenue
    lin_revenue_array : np.array
        Linear array from min to max revenue
    lin_loss_array : np.array
        Linear array of loss from min to max
    threshold_payoff: float
        The threshold for the payoff
    """
    eol_from_distribution = get_eol_from_distribution(
        revenue_distribution, lin_revenue_array, lin_loss_array
    )
    plot_properties = {
        "x_label": "Revenue [$]",
        "y_label": "EOL [$]",
        "label": "EOL of revenue",
        "color": "purple",
        "step": 40,
        "width": 1e5,
    }
    fig, axis = plot_bar(lin_revenue_array, eol_from_distribution, plot_properties)
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, "units_eol.png")


if __name__ == "__main__":
    main()
