from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erfinv
from scipy import stats


class Distribution(ABC):

    @abstractmethod
    def cdf(self, n_points):
        pass

    def incremental_probability(self, points):
        ret = np.zeros(points.size)
        ret[1:] = np.array([self.cdf(points[i]) - self.cdf(points[i-1]) for i in range(1, points.size)])
        return ret


class Gaussian(Distribution):
    def __init__(self, lower, upper, ci=0.9):
        # https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
        self.sd = (upper - lower) / (2 * np.sqrt(2) * erfinv(ci))
        self.mean = (upper + lower) / 2

    def sample(self, n_points=1000):
        # RVS - Random variates
        return stats.norm.rvs(loc=self.mean, scale=self.sd, size=n_points)

    def pdf(self, points):
        return stats.norm.pdf(points, loc=self.mean, scale=self.sd)

    def cdf(self, points):
        return stats.norm.cdf(points, loc=self.mean, scale=self.sd)


class DistributionFromData(Distribution):
    # Get PDE from data using KDE
    # NOTE: With histograms you must bin the data
    # NOTE: Works best with unimodal distributions
    # https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html#kernel-density-estimation

    # NOTE: It's also possible to find best fit for distribution with
    #       Kolmogorov-Smirnoff test
    #       https://stackoverflow.com/q/37487830/2786884
    #       https://stackoverflow.com/q/6620471/2786884

    def __init__(self, data, min=None):
        self.__data = data
        if min is None:
            self.__min = data.min()
        else:
            self.__min = min
        self.__kde = stats.gaussian_kde(self.__data)

    def sample(self, n_points=1000):
        return self.__kde.resample(n_points)

    def pdf(self, points):
        return self.__kde.evaluate(points)

    def cdf(self, points):
        return self.__kde.integrate_box_1d(self.__min, points)
