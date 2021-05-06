"""Functions for calculating continuous expected value of perfect information."""

from typing import Tuple

import numpy as np

from htma_py.distribution import Distribution, DistributionFromSamples


def get_evpi_from_samples(
    revenue_sample: np.array, threshold_payoff: float, n_points_lin_array: int
) -> float:
    """
    Return the expected value of perfect information given a threshold.

    Parameters
    ----------
    revenue_sample : np.array
        Shape: (n_samples,)
        Samples of the revenue
    threshold_payoff : float
        The threshold where the revenue is considered a loss
    n_points_lin_array : int
        Number of points to use in the linear array of revenues

    Returns
    -------
    evpi : float
        The expected value of perfect information
    """
    revenue_min = revenue_sample.min()
    revenue_max = revenue_sample.max()
    lin_revenue_array, lin_loss_array = get_lin_revenue_and_loss(
        revenue_min, revenue_max, threshold_payoff, n_points_lin_array
    )

    revenue_dist = DistributionFromSamples(revenue_sample, min_x_value=revenue_min)

    evpi = calculate_evpi(revenue_dist, lin_revenue_array, lin_loss_array)
    return evpi


def get_lin_revenue_and_loss(
    revenue_min: float,
    revenue_max: float,
    threshold_payoff: float,
    n_points_lin_array: int,
) -> Tuple[np.array, np.array]:
    """
    Get the linear revenue and loss arrays assuming linear loss with threshold.

    Parameters
    ----------
    revenue_min : float
        The minimum revenue
    revenue_max : float
        The maximum revenue
    threshold_payoff : float
        Threshold for where the revenue is considered a loss
        All under the threshold will be a positive value
        All above the threshold will be 0
    n_points_lin_array : int
        Number of points in the linear arrays

    Returns
    -------
    lin_revenue_array : np.array
        Size: (n_points_lin_array,)
        The revenue array
    lin_loss_array : np.array
        Size: (n_points_lin_array,)
        The loss array
    """
    lin_revenue_array = np.linspace(revenue_min, revenue_max, n_points_lin_array)
    lin_loss_array = get_lin_loss_array(lin_revenue_array, threshold_payoff)
    return lin_revenue_array, lin_loss_array


def get_lin_loss_array(
    lin_revenue_array: np.array, threshold_payoff: float
) -> np.array:
    """
    Return the loss array.

    We are here assuming a linear loss function with a threshold.
    Note that the loss function does not discriminate gain.
    I.e. loss is a positive value, and all values of gains = 0 loss

    Parameters
    ----------
    lin_revenue_array : np.array
        Shape: (n_points_lin_array,)
        A linear array with the range of revenues to calculate the loss for
    threshold_payoff : int
        The threshold for where the revenue is considered a loss

    Returns
    -------
    lin_loss_array : np.array
        Shape: (n_points_lin_array,)
        The loss array
    """
    lin_loss_array = threshold_payoff - lin_revenue_array
    lin_loss_array[lin_loss_array <= 0] = 0
    return lin_loss_array


def get_eol_from_distribution(
    distribution: Distribution, lin_x_values: np.array, lin_loss_array: np.array
) -> np.array:
    """
    Calculate the expected value of perfect information given the distribution and loss.

    Parameters
    ----------
    distribution : Distribution
        The distribution to calculate the incremental probability from
    lin_x_values : np.array
        The points to calculate the incremental probability from
    lin_loss_array : np.array
        The loss array to calculate the EVPI from

    Returns
    -------
    eol_from_distribution : np.array
        The expected opportunity loss based on the distribution

    See Also
    --------
    calculate_evpi : Preforms the same calculation as this function, but sums the EOL
    """
    incremental_prob_x_array = distribution.incremental_probability(lin_x_values)
    eol_from_distribution = lin_loss_array * incremental_prob_x_array
    return eol_from_distribution


def calculate_evpi(
    distribution: Distribution, lin_revenue_array: np.array, lin_loss_array: np.array
) -> float:
    """
    Calculate the expected value of perfect information given the distribution and loss.

    Parameters
    ----------
    distribution : Distribution
        The distribution to calculate the incremental probability from
    lin_revenue_array : np.array
        The points to calculate the incremental probability from
    lin_loss_array : np.array
        The loss array to calculate the EVPI from

    Returns
    -------
    evpi : float
        The expected value of perfect information

    See Also
    --------
    get_eol_from_distribution : Preforms this calculation without summing the EOL
    """
    incremental_prob_x_array = distribution.incremental_probability(lin_revenue_array)
    evpi: float = lin_loss_array @ incremental_prob_x_array
    return evpi


def print_risk(revenue_samples: np.array, revenue_threshold: float) -> None:
    """
    Print the risk.

    Parameters
    ----------
    revenue_samples : np.array
        The sampled revenue
    revenue_threshold : float
        The threshold
    """
    risk = (
        100
        * np.where(revenue_samples < revenue_threshold)[0].size
        / revenue_samples.size
    )
    print(f"Risk of losing money: {risk:.1f} %")
