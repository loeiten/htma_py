"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx"""

import numpy as np
from src.distribution import Gaussian
from src.distribution import DistributionFromData


def loss(units, threshold=2.0e5, rate=25):
    return (threshold - units) * rate if units <= threshold else 0


def main():
    units_array = np.linspace(0, 452963, int(2.0e3))
    # Scalability
    # https://stackoverflow.com/a/46470401/2786884
    loss_array = np.array([loss(units) for units in units_array])

    units_gauss_dist = Gaussian(1.5e5, 3.0e5)
    incremental_prob_units_gauss = units_gauss_dist.incremental_probability(units_array)
    evpi_gauss = loss_array@incremental_prob_units_gauss
    print(f"Expected value of perfect information (calculated from Gauss): {evpi_gauss:.1f}")

    units_data_dist = DistributionFromData(units_gauss_dist.sample(int(2.0e3)), min=0)
    incremental_prob_units_data_dist = units_data_dist.incremental_probability(units_array)
    evpi_data_dist = loss_array@incremental_prob_units_data_dist
    print(f"Expected value of perfect information (calculated from data distribution): {evpi_data_dist:.1f}")


if __name__ == "__main__":
    main()
