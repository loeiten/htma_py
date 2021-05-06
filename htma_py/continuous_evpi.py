"""Functions for calculating continuous expected value of perfect information."""

from typing import Tuple

import numpy as np

from htma_py.distribution import Distribution, DistributionFromSamples


def get_evpi_from_samples(
    payoff_sample: np.array, threshold_payoff: float, n_points_lin_array: int
) -> float:
    """
    Return the expected value of perfect information given a threshold.

    Parameters
    ----------
    payoff_sample : np.array
        Shape: (n_samples,)
        Samples of the payoff
    threshold_payoff : float
        The threshold where the payoff is considered a loss
    n_points_lin_array : int
        Number of points to use in the linear array of payoffs

    Returns
    -------
    evpi : float
        The expected value of perfect information
    """
    payoff_min = payoff_sample.min()
    payoff_max = payoff_sample.max()
    lin_payoff_array, lin_loss_array = get_lin_payoff_and_loss(
        payoff_min, payoff_max, threshold_payoff, n_points_lin_array
    )

    payoff_dist = DistributionFromSamples(payoff_sample, min_x_value=payoff_min)

    evpi = calculate_evpi(payoff_dist, lin_payoff_array, lin_loss_array)
    return evpi


def get_lin_payoff_and_loss(
    payoff_min: float,
    payoff_max: float,
    threshold_payoff: float,
    n_points_lin_array: int,
) -> Tuple[np.array, np.array]:
    """
    Get the linear payoff and loss arrays assuming linear loss with threshold.

    Parameters
    ----------
    payoff_min : float
        The minimum payoff
    payoff_max : float
        The maximum payoff
    threshold_payoff : float
        Threshold for where the payoff is considered a loss
        All under the threshold will be a positive value
        All above the threshold will be 0
    n_points_lin_array : int
        Number of points in the linear arrays

    Returns
    -------
    lin_payoff_array : np.array
        Size: (n_points_lin_array,)
        The payoff array
    lin_loss_array : np.array
        Size: (n_points_lin_array,)
        The loss array
    """
    lin_payoff_array = np.linspace(payoff_min, payoff_max, n_points_lin_array)
    lin_loss_array = get_lin_loss_array(lin_payoff_array, threshold_payoff)
    return lin_payoff_array, lin_loss_array


def get_lin_loss_array(lin_payoff_array: np.array, threshold_payoff: float) -> np.array:
    """
    Return the loss array.

    We are here assuming a linear loss function with a threshold.
    Note that the loss function does not discriminate gain.
    I.e. loss is a positive value, and all values of gains = 0 loss

    Parameters
    ----------
    lin_payoff_array : np.array
        Shape: (n_points_lin_array,)
        A linear array with the range of payoffs to calculate the loss for
    threshold_payoff : int
        The threshold for where the payoff is considered a loss

    Returns
    -------
    lin_loss_array : np.array
        Shape: (n_points_lin_array,)
        The loss array
    """
    lin_loss_array = threshold_payoff - lin_payoff_array
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
    distribution: Distribution, lin_x_values: np.array, lin_loss_array: np.array
) -> float:
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
    evpi : float
        The expected value of perfect information

    See Also
    --------
    get_eol_from_distribution : Preforms the same calculation as this, but does not sum the EOL
    """
    incremental_prob_x_array = distribution.incremental_probability(lin_x_values)
    evpi: float = lin_loss_array @ incremental_prob_x_array
    return evpi


def print_risk(payoff_samples: np.array, payoff_threshold: float) -> None:
    """
    Print the risk.

    Parameters
    ----------
    payoff_samples : np.array
        The sampled payoff
    payoff_threshold : float
        The threshold
    """
    risk = (
        100 * np.where(payoff_samples < payoff_threshold)[0].size / payoff_samples.size
    )
    print(f"Risk of losing money: {risk:.1f} %")
