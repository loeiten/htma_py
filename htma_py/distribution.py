"""Contains classes for distributions."""


from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erfinv  # pylint: disable=no-name-in-module


class Distribution(ABC):
    """Abstract class for all Distributions."""

    @abstractmethod
    def sample(self, n_points: int) -> np.array:
        """
        Sample from the distribution.

        Parameters
        ----------
        n_points : int
            How many samples to draw

        Returns
        -------
        np.array
            The drawn samples
        """

    @abstractmethod
    def pdf(self, x_values: np.array) -> np.array:
        """
        Return the probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The probability density for the given x_values
        """

    @abstractmethod
    def cdf(self, x_values: np.array) -> np.array:
        """
        Return the cumulative probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the cumulative probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The cumulative probability density for the given x_values
        """

    def incremental_probability(self, x_values: np.array) -> np.array:
        """
        Return the difference in probability between neighbouring cdf values.

        Notes
        -----
        The zeroth incremental probability is set to 0 as just subtracting neighbouring
        values would result in an array with N-1 x_values

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the incremental probability for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The difference between two neighbouring cdf values
        """
        ret = np.zeros(x_values.size)
        ret[1:] = np.array(
            [
                self.cdf(x_values[i]) - self.cdf(x_values[i - 1])
                for i in range(1, x_values.size)
            ]
        )
        return ret


class Gaussian(Distribution):
    """Implementation of a Gaussian distribution."""

    def __init__(self, lower_bound: float, upper_bound: float, ci=0.9) -> None:
        """
        Set the distribution from its confidence interval.

        Parameters
        ----------
        lower_bound : float
            The lower bound of the confidence interval
        upper_bound : float
            The upper bound of the confidence interval
        ci : float
            The confidence interval
        """
        # https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
        self.standard_deviation = (upper_bound - lower_bound) / (
            2 * np.sqrt(2) * erfinv(ci)
        )
        self.mean = (upper_bound + lower_bound) / 2

    def sample(self, n_points: int = 1000) -> np.array:
        """
        Sample from the distribution.

        Parameters
        ----------
        n_points : int
            How many samples to draw

        Returns
        -------
        np.array
            Shape: (n_points,)
            The drawn samples
        """
        # RVS - Random variates
        return stats.norm.rvs(
            loc=self.mean, scale=self.standard_deviation, size=n_points
        )

    def pdf(self, x_values: np.array) -> np.array:
        """
        Return the probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The probability density for the given x_values
        """
        return stats.norm.pdf(x_values, loc=self.mean, scale=self.standard_deviation)

    def cdf(self, x_values: np.array) -> np.array:
        """
        Return the cumulative probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the cumulative probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The cumulative probability density for the given x_values
        """
        return stats.norm.cdf(x_values, loc=self.mean, scale=self.standard_deviation)


class DistributionFromSamples(Distribution):
    """
    Obtain the distribution from samples and Gaussian KDE.

    Notes
    -----
    - Histograms are great to plot from, but not to sample from
    - This works best with unimodal distributions [1]_
    - Another approach is to find best fit for distribution with Kolmogorov-Smirnoff
      test, see [2]_ and [3]_

    References
    ----------
    .. [1]
    https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#kernel-density-estimation

    .. [2] https://stackoverflow.com/q/37487830/2786884
    .. [3] https://stackoverflow.com/q/6620471/2786884
    """

    def __init__(
        self, samples: np.array, min_x_value: Union[None, float] = None
    ) -> None:
        """Compute the KDE."""
        self.__samples = samples
        if min_x_value is None:
            self.__min_x_value = samples.min()
        else:
            self.__min_x_value = min_x_value
        self.__kde = stats.gaussian_kde(self.__samples)

    def sample(self, n_points: int = 1000) -> np.array:
        """
        Sample from the distribution.

        Parameters
        ----------
        n_points : int
            How many samples to draw

        Returns
        -------
        np.array
            Shape: (n_points,)
            The drawn samples
        """
        return self.__kde.resample(n_points)

    def pdf(self, x_values: np.array) -> np.array:
        """
        Return the probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The probability density for the given x_values
        """
        return self.__kde.evaluate(x_values)

    def cdf(self, x_values: np.array) -> np.array:
        """
        Return the cumulative probability density.

        Parameters
        ----------
        x_values : np.array
            Shape: (N,)
            The values to get the cumulative probability density for

        Returns
        -------
        ret : np.array
            Shape: (N,)
            The cumulative probability density for the given x_values
        """
        return self.__kde.integrate_box_1d(self.__min_x_value, x_values)


def get_samples(
    distributions: Dict[str, Distribution], n_samples: float = 5e4
) -> pd.DataFrame:
    """
    Get samples from distributions.

    Parameters
    ----------
    distributions : dict of str, Distribution
        Dictionary containing the names as keys and Distribution objects as values
    n_samples : int
        Number of samples to sample from the distributions

    Returns
    -------
    samples_df : DataFrame
        All the samples as a DataFrame
    """
    samples: Dict[str, np.array] = dict()
    for dist_name in distributions.keys():
        samples[dist_name] = distributions[dist_name].sample(int(n_samples))

    samples_df = pd.DataFrame(samples)
    return samples_df
