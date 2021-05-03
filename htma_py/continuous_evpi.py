"""Functions for calculating continuous expected value of perfect information."""

from typing import Tuple

import numpy as np

from htma_py.distribution import Distribution, DistributionFromData


def get_evpi(
    payoff_sample: np.array, threshold_payoff: float, n_points_lin_array: int
) -> float:
    """
    Return the expected value of perfect information given a threshold.

    Parameters
    ----------
    payoff_sample : (n_samples,) np.ndarray
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
    payoff_dist, payoff_lin_array = get_payoff_dist_and_lin_array(
        payoff_sample, n_points_lin_array
    )
    loss_array = get_loss_array(payoff_lin_array, threshold_payoff)
    evpi = calculate_evpi(payoff_lin_array, payoff_dist, loss_array)
    return evpi


def get_loss_array(payoff_lin_array: np.array, threshold_payoff: float) -> np.array:
    """
    Return the loss array.

    We are here assuming a linear loss function with a threshold

    Parameters
    ----------
    payoff_lin_array : (n_points_lin_array,) np.ndarray
        A linear array with the range of payoffs to calculate the loss for
    threshold_payoff : int
        The threshold for where the payoff is considered a loss

    Returns
    -------
    loss_array : (n_points_lin_array,) np.ndarray
        The loss array
    """
    loss_array = threshold_payoff - payoff_lin_array
    loss_array[loss_array <= 0] = 0
    return loss_array


def get_payoff_dist_and_lin_array(
    payoff_sample: np.array, n_points_lin_array: int
) -> Tuple[Distribution, np.array]:
    """
    Return the distribution of the payoff together with a linear array of the values.

    Parameters
    ----------
    payoff_sample : (n_samples,) np.ndarray
        Samples of the payoff
    n_points_lin_array : int
        Number of points to use in the linear array of payoffs

    Returns
    -------
    payoff_dist : Distribution
        Distribution object for the payoff
    payoff_lin_array : (n_points_lin_array,) np.ndarray
        A linear array containing the possible values for the distribution
    """
    payoff_min = payoff_sample.min()
    payoff_max = payoff_sample.max()
    payoff_lin_array = np.linspace(payoff_min, payoff_max, n_points_lin_array)
    payoff_dist = DistributionFromData(payoff_sample, min_x_value=payoff_min)
    return payoff_dist, payoff_lin_array


def calculate_evpi(
    x_values: np.array, distribution: Distribution, loss_array: np.array
) -> float:
    """
    Calculate the expected value of perfect information given the distribution and loss.

    Parameters
    ----------
    x_values : np.array
        The points to calculate the incremental probability from
    distribution : Distribution
        The distribution to calculate the incremental probability from
    loss_array : np.ndarray
        The loss array to calculate the EVPI from

    Returns
    -------
    evpi : float
        The expected value of perfect information
    """
    incremental_prob_x_array = distribution.incremental_probability(x_values)
    evpi = loss_array @ incremental_prob_x_array
    return evpi
