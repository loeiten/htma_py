"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx."""

from typing import Dict

import numpy as np

from htma_py.continuous_evpi import calculate_evpi, get_lin_payoff_and_loss, print_risk, get_eol_from_distribution
from htma_py.distribution import Distribution, Gaussian, get_samples
from htma_py.plot import plot_histogram, plot_loss, plot_bar, save_plot

from matplotlib.patches import Rectangle


def plot_units_histogram(units_samples: np.array, threshold: float) -> None:
    """
    Plot the units histogram.

    Parameters
    ----------
    units_samples : np.array
        Samples of the produced units
    threshold: float
        The threshold for the units
    """
    fig, axis, histogram_output = plot_histogram(units_samples, "Units")
    # Mark the threshold
    # Mark probabilities with risk red
    for cur_bin, patch in zip(histogram_output["bins"], histogram_output["patches"]):
        if cur_bin < threshold:
            patch.set_color("red")
        else:
            patch.set_color("green")
    # handles = [Rectangle((0, 0), 1, 1, color=c, alpha=0.75) for c in ["green", "red"]]
    # labels = ["Gain", "Loss"]
    # legend = axis.legend(handles, labels)
    axis.axvline(x=threshold, linestyle="--", color="k", ymin=0, ymax=1, label="Threshold")
    handles, labels = axis.get_legend_handles_labels()

    # Add legend for loss
    handles.append(Rectangle((0, 0), 1, 1, color="red", alpha=0.75))
    labels.append("Loss")

    # Add legend for gain
    handles.append(Rectangle((0, 0), 1, 1, color="green", alpha=0.75))
    labels.append("Gain")

    # Make legend
    legend = axis.legend(handles=handles, labels=labels, loc="best", fancybox=True, numpoints=1)
    legend.get_frame().set_alpha(0.5)

    save_plot(fig, "units_sold_histogram.png")


def plot_loss_functions(x_min: float, x_max: float, lin_loss_array: np.array, price_per_unit: float) -> None:
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
    price_per_unit : float
        Price per unit

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    fig, _ = plot_loss(x_min, x_max, lin_loss_array, "Units")
    save_plot(fig, "units_sold_units_loss_function.png")
    fig, _ = plot_loss(x_min*price_per_unit, x_max*price_per_unit, lin_loss_array, "Payoff [$]")
    save_plot(fig, "units_sold_payoff_loss_function.png")


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
    print(f"Expected value of perfect information for 'Payoff' variable: {evpi:.1f}\n")

    plot_units_histogram(samples_df.loc[:, "Units"], threshold_units)
    plot_loss_functions(units_min, units_max, lin_loss_array, price_per_unit)
    plot_incremental_probability(distributions["Payoff"], lin_payoff_array)
    plot_units_eol_histogram(distributions["Payoff"], units_min, units_max, lin_payoff_array, lin_loss_array)


def plot_incremental_probability(payoff_distribution, lin_payoff_array):
    incremental_prob_payoff_array = payoff_distribution.incremental_probability(lin_payoff_array)
    fig, _ = plot_bar(lin_payoff_array, incremental_prob_payoff_array, "Payoff [$]", "Incremental probability", label="IP of Payoff")
    save_plot(fig, "units_incremental_probability.png")


def plot_units_eol_histogram(payoff_distribution, units_min, units_max, lin_payoff_array, lin_loss_array):
    # FIXME: Make clear what is sampled and not
    # FIXME: Plot incremental probability
    # FIXME: Change plot_loss to make it more versatile
    # FIXME: Discrepancy Payoff and units
    # FIXME: Barchart for incremental probability
    # FIXME: Payoff is the wrong word as negative numbers in payoff matrix meant risk. I'm referring to revenue
    eol_from_distribution = get_eol_from_distribution(payoff_distribution, lin_payoff_array, lin_loss_array)
    fig, _ = plot_bar(lin_payoff_array, eol_from_distribution, "Payoff [$]", "EOL [$]", label="EOL of payoff")
    save_plot(fig, "units_eol.png")


if __name__ == "__main__":
    main()
