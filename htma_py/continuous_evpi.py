import numpy as np

from src.distribution import DistributionFromData


def get_evpi(payoff_sample, threshold_payoff, n):
    payoff_dist, payoff_lin_array = get_payoff(payoff_sample, n)
    loss_array = get_loss_array(payoff_lin_array, threshold_payoff)
    evpi = calculate_evpi(payoff_lin_array, payoff_dist, loss_array)
    return evpi


def get_loss_array(payoff_lin_array, threshold_payoff):
    loss_array = threshold_payoff - payoff_lin_array
    loss_array[loss_array <= 0] = 0
    return loss_array


def get_payoff(payoff, n):
    payoff_min = payoff.min()
    payoff_max = payoff.max()
    payoff_lin_array = np.linspace(payoff_min, payoff_max, n)
    payoff_dist = DistributionFromData(payoff, min=payoff_min)
    return payoff_dist, payoff_lin_array


def calculate_evpi(x_array, distribution, loss_array):
    incremental_prob_x_array = distribution.incremental_probability(x_array)
    evpi = loss_array @ incremental_prob_x_array
    return evpi