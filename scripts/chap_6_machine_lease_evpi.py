"""Calculate EVPI for each variables in the machine lease example of chapter 6."""

from htma_py.continuous_evpi import get_evpi
from htma_py.distribution import Gaussian


def main() -> None:
    """Calculate EVPI for each variable in machine lease example."""
    maintenance_savings_dist = Gaussian(10, 20)
    labor_savings_dist = Gaussian(-2, 8)
    raw_materials_savings_dist = Gaussian(3, 9)
    production_level_dist = Gaussian(1.5e4, 3.5e4)
    threshold_payoff = 4e5

    n_points_sample = 2000
    n_points_lin_array = 2000

    ms_sample = maintenance_savings_dist.sample(n_points_sample)
    ls_sample = labor_savings_dist.sample(n_points_sample)
    rms_sample = raw_materials_savings_dist.sample(n_points_sample)
    pl_sample = production_level_dist.sample(n_points_sample)

    ms_mean = maintenance_savings_dist.mean
    ls_mean = labor_savings_dist.mean
    rms_mean = raw_materials_savings_dist.mean
    pl_mean = production_level_dist.mean

    payoff = (ms_sample + ls_sample + rms_sample) * pl_sample
    overall_evpi = get_evpi(payoff, threshold_payoff, n_points_lin_array)
    print(
        f"The overall EVPI is {overall_evpi:.0f}\tOverall threshold: "
        f"{threshold_payoff:.0f}"
    )

    print(
        "\n" + "=" * 80 + "\nUsing method 1 where all variables are kept at the mean:"
    )
    payoff_ms = (ms_sample + ls_mean + rms_mean) * pl_mean
    threshold_ms = (threshold_payoff / (pl_mean)) - ls_mean - rms_mean
    evpi_ms = get_evpi(payoff_ms, threshold_payoff, n_points_lin_array)
    print(f"MS EVPI is {evpi_ms:.0f}\tThreshold for mean MS: {threshold_ms:.0f}")

    payoff_ls = (ms_mean + ls_sample + rms_mean) * pl_mean
    threshold_ls = (threshold_payoff / (pl_mean)) - ms_mean - rms_mean
    evpi_ls = get_evpi(payoff_ls, threshold_payoff, n_points_lin_array)
    print(f"LS EVPI is {evpi_ls:.0f}\tThreshold for mean LS: {threshold_ls:.0f}")

    payoff_rms = (ms_mean + ls_mean + rms_sample) * pl_mean
    threshold_rms = (threshold_payoff / (pl_mean)) - ms_mean - ls_mean
    evpi_rms = get_evpi(payoff_rms, threshold_payoff, n_points_lin_array)
    print(f"RMS EVPI is {evpi_rms:.0f}\tThreshold for mean RMS: {threshold_rms:.0f}")

    payoff_pl = (ms_mean + ls_mean + rms_mean) * pl_sample
    threshold_pl = threshold_payoff / ((ms_mean + ls_mean + rms_mean))
    evpi_pl = get_evpi(payoff_pl, threshold_payoff, n_points_lin_array)
    print(f"PL EVPI is {evpi_pl:.0f}\tThreshold for mean PL: {threshold_pl:.0f}")

    print(
        "\n"
        + "=" * 80
        + "\nUsing method 2 where only one variable is assumed known (kept at mean):"
    )
    payoff_ms = (ms_mean + ls_sample + rms_sample) * pl_sample
    evpi_ms = overall_evpi - get_evpi(payoff_ms, threshold_payoff, n_points_lin_array)
    print(f"Individual MS EVPI: {evpi_ms:.0f}")

    payoff_ls = (ms_sample + ls_mean + rms_sample) * pl_sample
    evpi_ls = overall_evpi - get_evpi(payoff_ls, threshold_payoff, n_points_lin_array)
    print(f"Individual LS EVPI: {evpi_ls:.0f}")

    payoff_rms = (ms_sample + ls_sample + rms_mean) * pl_sample
    evpi_rms = overall_evpi - get_evpi(payoff_rms, threshold_payoff, n_points_lin_array)
    print(f"Individual RMS EVPI: {evpi_rms:.0f}")

    payoff_pl = (ms_sample + ls_sample + rms_sample) * pl_mean
    evpi_pl = overall_evpi - get_evpi(payoff_pl, threshold_payoff, n_points_lin_array)
    print(f"Individual PL EVPI: {evpi_pl:.0f}")


if __name__ == "__main__":
    main()
