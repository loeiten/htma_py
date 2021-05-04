"""Tests that the method of calculating multiple variables is correct."""

from htma_py.continuous_evpi import get_evpi
from htma_py.distribution import Gaussian


def main() -> None:
    """Test that calculation of EVPI corresponds to the units production example."""
    # Tweaked the numbers such that we a distribution similar to
    # Gaussian(1.5e5, 3.0e5) in return
    variable_a = Gaussian(8, 10)
    variable_b = Gaussian(-1, 3)
    variable_c = Gaussian(4, 5)
    variable_d = Gaussian(1.1e4, 2.0e4)

    n_samples = 2000
    n_points_lin_array = 2000

    price_per_unit = 25
    threshold_units = 2.0e5
    threshold_payoff = price_per_unit * threshold_units

    a_sample = variable_a.sample(n_samples)
    b_sample = variable_b.sample(n_samples)
    c_sample = variable_c.sample(n_samples)
    d_sample = variable_d.sample(n_samples)

    a_mean = variable_a.mean
    b_mean = variable_b.mean
    c_mean = variable_c.mean
    d_mean = variable_d.mean

    payoff = price_per_unit * (a_sample + b_sample + c_sample) * d_sample
    overall_evpi = get_evpi(payoff, threshold_payoff, n_points_lin_array)
    print(
        f"The overall EVPI is {overall_evpi:.0f}\tOverall threshold: "
        f"{threshold_payoff:.0f}"
    )

    print(
        "\n" + "=" * 80 + "\nUsing method 1 where all variables are kept at the mean:"
    )
    payoff_a = price_per_unit * (a_sample + b_mean + c_mean) * d_mean
    threshold_a = (threshold_payoff / (price_per_unit * d_mean)) - b_mean - c_mean
    evpi_a = get_evpi(payoff_a, threshold_payoff, n_points_lin_array)
    print(f"A EVPI is {evpi_a:.0f}\tThreshold for mean A: {threshold_a:.0f}")

    payoff_b = price_per_unit * (a_mean + b_sample + c_mean) * d_mean
    threshold_b = (threshold_payoff / (price_per_unit * d_mean)) - a_mean - c_mean
    evpi_b = get_evpi(payoff_b, threshold_payoff, n_points_lin_array)
    print(f"B EVPI is {evpi_b:.0f}\tThreshold for mean B: {threshold_b:.0f}")

    payoff_c = price_per_unit * (a_mean + b_mean + c_sample) * d_mean
    threshold_c = (threshold_payoff / (price_per_unit * d_mean)) - a_mean - b_mean
    evpi_c = get_evpi(payoff_c, threshold_payoff, n_points_lin_array)
    print(f"C EVPI is {evpi_c:.0f}\tThreshold for mean C: {threshold_c:.0f}")

    payoff_d = price_per_unit * (a_mean + b_mean + c_mean) * d_sample
    threshold_d = threshold_payoff / (price_per_unit * (a_mean + b_mean + c_mean))
    evpi_d = get_evpi(payoff_d, threshold_payoff, n_points_lin_array)
    print(f"D EVPI is {evpi_d:.0f}\tThreshold for mean D: {threshold_d:.0f}")

    print(
        "\n"
        + "=" * 80
        + "\nUsing method 2 where only one variable is assumed known (kept at mean):"
    )
    payoff_a = price_per_unit * (a_mean + b_sample + c_sample) * d_sample
    evpi_a = overall_evpi - get_evpi(payoff_a, threshold_payoff, n_points_lin_array)
    print(f"Individual A EVPI: {evpi_a:.0f}")

    payoff_b = price_per_unit * (a_sample + b_mean + c_sample) * d_sample
    evpi_b = overall_evpi - get_evpi(payoff_b, threshold_payoff, n_points_lin_array)
    print(f"Individual B EVPI: {evpi_b:.0f}")

    payoff_c = price_per_unit * (a_sample + b_sample + c_mean) * d_sample
    evpi_c = overall_evpi - get_evpi(payoff_c, threshold_payoff, n_points_lin_array)
    print(f"Individual C EVPI: {evpi_c:.0f}")

    payoff_d = price_per_unit * (a_sample + b_sample + c_sample) * d_mean
    evpi_d = overall_evpi - get_evpi(payoff_d, threshold_payoff, n_points_lin_array)
    print(f"Individual D EVPI: {evpi_d:.0f}")


if __name__ == "__main__":
    main()
