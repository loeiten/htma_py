"""Test that the implementation follows the result of chapter 7 of the book."""

import numpy as np

from htma_py.continuous_evpi import calculate_evpi, get_lin_revenue_and_loss
from htma_py.distribution import DistributionFromSamples, Gaussian


def test_chapter_7_evpi():
    """
    Test that the implementation yields the correct EVPI.

    The EVPI is calculated both from an exact distribution and from a distribution
    generated from samples.
    """
    # Fixing random state for reproducibility
    np.random.seed(19680801)

    price_per_unit = 25
    threshold_units = 2.0e5
    lower_90_ci = 1.5e5
    upper_90_ci = 3.0e5

    # Note: These numbers are used in teh excel example
    units_min = 0
    units_max = 452963
    n_points_lin_array = int(2e3)

    # Calculate revenue parameters
    threshold_payoff = price_per_unit * threshold_units

    lin_revenue_array, lin_loss_array = get_lin_revenue_and_loss(
        units_min * price_per_unit,
        units_max * price_per_unit,
        threshold_payoff,
        n_points_lin_array,
    )

    revenue_gauss_dist = Gaussian(
        lower_90_ci * price_per_unit, upper_90_ci * price_per_unit
    )
    evpi_exact_gauss = calculate_evpi(
        revenue_gauss_dist, lin_revenue_array, lin_loss_array
    )

    # NOTE: We need a lot of samples in order for the test to pass
    units_data_dist = DistributionFromSamples(
        revenue_gauss_dist.sample(int(1e5)), min_x_value=units_min * price_per_unit
    )
    evpi_data_from_distribution = calculate_evpi(
        units_data_dist, lin_revenue_array, lin_loss_array
    )

    expected_from_excel = 208127.5
    assert np.isclose(evpi_exact_gauss, expected_from_excel)
    # Allow 10 % relative error (this can be stricter if we use higher n_samples)
    assert np.isclose(evpi_exact_gauss, evpi_data_from_distribution, rtol=1.0e-1)
