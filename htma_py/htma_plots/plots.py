"""Functions for plotting."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from htma_py.htma_plots.plot_helpers import (
    make_axis_pretty,
    make_legend,
    save_plot,
    set_labels_and_legends,
)

if TYPE_CHECKING:
    from htma_py.distribution import Distribution  # pylint: disable=cyclic-import


def plot_line(
    x_var: np.array,
    y_var: np.array,
    line_plot_properties: Optional[Dict[str, Any]] = None,
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot a line plot.

    Parameters
    ----------
    x_var : np.array
        The minimum value for the dependant variable
    y_var : np.array
        The maximum value for the dependant variable
    line_plot_properties : dict of str
        The plot properties consisting of
        - x_label : str
            - Name to put on the x-axis
        - y_label : str
            - Name to put on the x-axis
        - label : str
            - Label to put on the legend

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    if line_plot_properties is None:
        line_plot_properties = {}
    if "label" not in line_plot_properties.keys():
        line_plot_properties["label"] = None

    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)

    axis.plot(
        x_var,
        y_var,
        label=line_plot_properties["label"],
    )
    axis = set_labels_and_legends(axis, line_plot_properties)

    return fig, axis


def plot_bar(
    x_var: np.array,
    y_var: np.array,
    bar_plot_properties: Optional[Dict[str, Any]] = None,
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot a bar plot.

    Parameters
    ----------
    x_var : np.array
        The minimum value for the dependant variable
    y_var : np.array
        The maximum value for the dependant variable
    bar_plot_properties : dict of str
        The plot properties consisting of
        - x_label : str
            - Name to put on the x-axis
        - y_label : str
            - Name to put on the x-axis
        - label : str
            - Label to put on the legend
        - color : str
            - Color to use on the bar

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    if bar_plot_properties is None:
        bar_plot_properties = {}
    if "label" not in bar_plot_properties.keys():
        bar_plot_properties["label"] = None
    if "color" not in bar_plot_properties.keys():
        bar_plot_properties["color"] = "#1f77b4"

    n_bars = 50
    step = int(len(x_var) / n_bars)
    width = x_var.max() / 100

    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)

    bars = axis.bar(
        x_var[::step],
        y_var[::step],
        color=bar_plot_properties["color"],
        label=bar_plot_properties["label"],
        width=width,
        alpha=0.75,
    )
    for patch in bars:
        patch.set_color(bar_plot_properties["color"])

    axis = set_labels_and_legends(axis, bar_plot_properties)

    return fig, axis


def plot_histogram(
    samples_from_distribution: np.array, x_label: str
) -> Tuple[figure.Figure, axes.Axes, Dict[str, Any]]:
    """
    Plot histogram from samples.

    Parameters
    ----------
    samples_from_distribution : np.array
        Size: (N,)
        The samples drawn from the distribution
    x_label : str
        Name to put on the x-axis

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    histogram_output: dict of str, Any
        The output from .hist
        - counts are the counts of hits in each bin
        - bins are the edges of each bin
        - patches are the corresponding patches object
    """
    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)
    counts, bins, patches = axis.hist(
        samples_from_distribution, bins=100, density=False, alpha=0.75
    )
    histogram_output = {"counts": counts, "bins": bins, "patches": patches}
    axis.set_xlabel(x_label)
    axis.set_ylabel("Number of hits in a bin")
    axis = make_axis_pretty(axis)
    return fig, axis, histogram_output


def plot_loss(
    lin_x_array, lin_loss_array: np.array, x_label: str
) -> Tuple[figure.Figure, axes.Axes]:
    """
    Plot the loss function.

    Parameters
    ----------
    lin_x_array : np.array
        The linear x array
    lin_loss_array : np.array
        The loss array
    x_label : str
        Name to put on the x-axis

    Returns
    -------
    fig : figure.Figure
        The figure object
    axis : axes.Axes
        The axis object
    """
    fig = plt.figure(figsize=(5, 3))
    axis = fig.add_subplot(111)

    axis.plot(lin_x_array, lin_loss_array, color="red", label="Loss function")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Monetary loss [$]")
    axis = make_axis_pretty(axis)

    # Make legend
    make_legend(axis)

    return fig, axis


def plot_sample_histogram_with_threshold(
    samples: np.array,
    threshold: float,
    sample_of: str,
    sample_units: str,
    save_name_prefix: str,
) -> None:
    """
    Plot a histogram of the samples.

    Parameters
    ----------
    save_name_prefix
    samples : np.array
        Samples to plot
    threshold: float
        The threshold for the samples
    sample_of : str
        Name of what we are sampling
    sample_units : str
        The units used for the samples
    save_name_prefix : str
        The prefix to use in the saved names
    """
    fig, axis, histogram_output = plot_histogram(
        samples, f"{sample_of.capitalize()} [{sample_units}]"
    )
    # Mark the threshold
    # Mark probabilities with risk red
    for cur_bin, patch in zip(histogram_output["bins"], histogram_output["patches"]):
        if cur_bin < threshold:
            patch.set_color("red")
        else:
            patch.set_color("green")

    axis.axvline(
        x=threshold,
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

    save_plot(fig, f"{save_name_prefix}_histogram.png")


def plot_eol_with_threshold(
    lin_eol_from_distribution: np.array,
    lin_revenue_array: np.array,
    threshold_payoff: float,
    save_name_prefix: str,
) -> None:
    """
    Plot expected opportunity loss.

    Parameters
    ----------
    lin_eol_from_distribution : np.array
        EOL obtained from the distribution
    lin_revenue_array : np.array
        Linear array from min to max revenue
    threshold_payoff: float
        The threshold for the payoff
    save_name_prefix : str
        The prefix to use in the saved names
    """
    plot_properties = {
        "x_label": "Revenue [$]",
        "y_label": "EOL [$]",
        "label": "EOL of revenue",
        "color": "purple",
    }
    fig, axis = plot_bar(lin_revenue_array, lin_eol_from_distribution, plot_properties)
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, f"{save_name_prefix}_eol.png")


def plot_pdf_cdf_and_incremental_probability(
    revenue_distribution: "Distribution",
    lin_revenue_array: np.array,
    threshold_payoff: float,
    save_name_prefix: str,
) -> None:
    """
    Plot the PDF, CDF and incremental probability.

    Parameters
    ----------
    revenue_distribution : Distribution
        The distribution of the revenue
    lin_revenue_array : np.array
        Linear array from min to max revenue
    threshold_payoff: float
        The threshold for the payoff
    save_name_prefix : str
        The prefix to use in the saved names
    """
    fig, axis = revenue_distribution.plot_pdf(lin_revenue_array, "Revenue [$]")
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    save_plot(fig, f"{save_name_prefix}_pdf.png")

    fig, axis = revenue_distribution.plot_cdf(lin_revenue_array, "Revenue [$]")
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    save_plot(fig, f"{save_name_prefix}_cdf.png")

    incremental_prob_revenue_array = revenue_distribution.incremental_probability(
        lin_revenue_array
    )
    plot_properties = {
        "x_label": "Revenue [$]",
        "y_label": "Incremental probability",
        "label": "IP of Revenue",
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
    save_plot(fig, f"{save_name_prefix}_incremental_probability.png")


def plot_loss_functions(
    lin_revenue_array: np.array,
    lin_loss_array: np.array,
    threshold_payoff: float,
    save_name_prefix: str,
) -> None:
    """
    Plot the loss function.

    Parameters
    ----------
    lin_revenue_array : np.array
        The revenue array
    lin_loss_array : np.array
        The loss array
    threshold_payoff: float
        The threshold for the units
    save_name_prefix : str
        The prefix to use in the saved names
    """
    fig, axis = plot_loss(lin_revenue_array, lin_loss_array, "Revenue [$]")
    axis.axvline(
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, f"{save_name_prefix}_revenue_loss_function.png")

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
        x=threshold_payoff,
        linestyle="--",
        color="k",
        ymin=0,
        ymax=1,
        label="Payoff threshold",
    )
    make_legend(axis)
    save_plot(fig, f"{save_name_prefix}_revenue_loss_function_bar.png")
