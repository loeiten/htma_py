"""Tests that the method of calculating multiple variables is correct."""

from src.distribution import Gaussian
from src.continuous_evpi import get_evpi


def main():
    # Tweaked the numbers such that we a distribution similar to Gaussian(1.5e5, 3.0e5) in return
    a = Gaussian(8, 10)
    b = Gaussian(-1, 3)
    c = Gaussian(4, 5)
    d = Gaussian(1.1e4, 2.0e4)
    n = 2000

    price_per_unit = 25
    threshold_units = 2.0e5
    threshold_payoff = price_per_unit * threshold_units

    a_sample = a.sample(n)
    b_sample = b.sample(n)
    c_sample = c.sample(n)
    d_sample = d.sample(n)
    
    a_mean = a.mean
    b_mean = b.mean
    c_mean = c.mean
    d_mean = d.mean

    payoff = price_per_unit * (a_sample + b_sample + c_sample) * d_sample
    overall_evpi = get_evpi(payoff, threshold_payoff, n)
    print(f"The overall EVPI is {overall_evpi:.0f}\tOverall threshold: {threshold_payoff:.0f}")

    print("\n" + "="*80 + "\nUsing method 1 where all variables are kept at the mean:")
    payoff_a = price_per_unit * (a_sample + b_mean + c_mean) * d_mean
    threshold_a = (threshold_payoff/(price_per_unit*d_mean)) - b_mean - c_mean
    evpi_a = get_evpi(payoff_a, threshold_payoff, n)
    print(f"A EVPI is {evpi_a:.0f}\tThreshold for mean A: {threshold_a:.0f}")

    payoff_b = price_per_unit * (a_mean + b_sample + c_mean) * d_mean
    threshold_b = (threshold_payoff/(price_per_unit*d_mean)) - a_mean - c_mean
    evpi_b = get_evpi(payoff_b, threshold_payoff, n)
    print(f"B EVPI is {evpi_b:.0f}\tThreshold for mean B: {threshold_b:.0f}")

    payoff_c = price_per_unit * (a_mean + b_mean + c_sample) * d_mean
    threshold_c = (threshold_payoff/(price_per_unit*d_mean)) - a_mean - b_mean
    evpi_c = get_evpi(payoff_c, threshold_payoff, n)
    print(f"C EVPI is {evpi_c:.0f}\tThreshold for mean C: {threshold_c:.0f}")

    payoff_d = price_per_unit * (a_mean + b_mean + c_mean) * d_sample
    threshold_d = (threshold_payoff/(price_per_unit*(a_mean + b_mean + c_mean)))
    evpi_d = get_evpi(payoff_d, threshold_payoff, n)
    print(f"D EVPI is {evpi_d:.0f}\tThreshold for mean D: {threshold_d:.0f}")
    
    print("\n" + "="*80 + "\nUsing method 2 where only one variable is assumed known (kept at mean):")
    payoff_a = price_per_unit * (a_mean + b_sample + c_sample) * d_sample
    evpi_a = overall_evpi - get_evpi(payoff_a, threshold_payoff, n)
    print(f"Individual A EVPI: {evpi_a:.0f}")

    payoff_b = price_per_unit * (a_sample + b_mean + c_sample) * d_sample
    evpi_b = overall_evpi - get_evpi(payoff_b, threshold_payoff, n)
    print(f"Individual B EVPI: {evpi_b:.0f}")

    payoff_c = price_per_unit * (a_sample + b_sample + c_mean) * d_sample
    evpi_c = overall_evpi - get_evpi(payoff_c, threshold_payoff, n)
    print(f"Individual C EVPI: {evpi_c:.0f}")

    payoff_d = price_per_unit * (a_sample + b_sample + c_sample) * d_mean
    evpi_d = overall_evpi - get_evpi(payoff_d, threshold_payoff, n)
    print(f"Individual D EVPI: {evpi_d:.0f}")


if __name__ == "__main__":
    main()
