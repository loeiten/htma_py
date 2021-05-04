"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx."""

import numpy as np

from htma_py.distribution import DistributionFromData, Gaussian


def loss(units: int, threshold: float = 2.0e5, rate: float = 25) -> float:
    """
    Return a linear loss function with threshold.

    Parameters
    ----------
    units : int
        Number of units produced
    threshold : float
        The threshold for loss
    rate : float
        The rate of loss

    Returns
    -------
    float
        The loss
    """
    return (threshold - units) * rate if units <= threshold else 0


def main() -> None:
    """Calculate the EVPI for the units example."""
    units_array = np.linspace(0, 452963, int(2.0e3))
    # Scalability
    # https://stackoverflow.com/a/46470401/2786884
    loss_array = np.array([loss(units) for units in units_array])

    units_gauss_dist = Gaussian(1.5e5, 3.0e5)
    incremental_prob_units_gauss = units_gauss_dist.incremental_probability(units_array)
    evpi_gauss = loss_array @ incremental_prob_units_gauss
    print(
        f"Expected value of perfect information (calculated from Gauss): "
        f"{evpi_gauss:.1f}"
    )

    units_data_dist = DistributionFromData(
        units_gauss_dist.sample(int(2.0e3)), min_x_value=0
    )
    incremental_prob_units_data_dist = units_data_dist.incremental_probability(
        units_array
    )
    evpi_data_dist = loss_array @ incremental_prob_units_data_dist
    print(
        f"Expected value of perfect information (calculated from data distribution): "
        f"{evpi_data_dist:.1f}"
    )


if __name__ == "__main__":
    main()
