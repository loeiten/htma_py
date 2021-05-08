"""Replication of HTMA_3rd.ed_Ch.7_Examples-8-Nov-19.xlsx."""

from typing import Dict

from htma_py.continuous_evpi import (
    calculate_evpi,
    get_eol_from_distribution,
    get_lin_revenue_and_loss,
    print_risk,
)
from htma_py.distribution import Distribution, Gaussian, get_samples
from htma_py.htma_plots.plots import (
    plot_eol_with_threshold,
    plot_loss_functions,
    plot_pdf_cdf_and_incremental_probability,
    plot_sample_histogram_with_threshold,
)


def main() -> None:
    """Calculate the risk and EVPI for the units example."""
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

    # Obtain the linear arrays
    lin_revenue_array, lin_loss_array = get_lin_revenue_and_loss(
        units_min * price_per_unit,
        units_max * price_per_unit,
        threshold_payoff,
        n_points_lin_array,
    )

    # Create the distributions
    units_gauss_dist = Gaussian(lower_90_ci, upper_90_ci)
    revenue_gauss_dist = Gaussian(
        lower_90_ci * price_per_unit, upper_90_ci * price_per_unit
    )
    # NOTE: We are specifying type in order not to get mypy invariant/covariant error
    distributions: Dict[str, Distribution] = {
        "Units": units_gauss_dist,
        "Revenue": revenue_gauss_dist,
    }

    # Get samples
    # NOTE: We are just choosing n_points_lin_array out of laziness.
    #       We could have used any other number
    samples_df = get_samples(distributions, n_samples=n_points_lin_array)

    print(
        "Samples from 'Units' distribution "
        "(we are letting units be floats for simplicity):"
    )
    # NOTE: We have sampled "Units" and "Revenue" individually, so it would be
    #       misleading to print them together
    # NOTE: When printing to ipython, it's better to print using .style
    #       https://stackoverflow.com/a/46370761/2786884
    print(samples_df.loc[:, "Units"], end="\n" * 2)

    # NOTE: One could also state the risk from the variable itself like so:
    #       print_risk(samples_df.loc[:, "Units"], threshold_units)
    print_risk(samples_df.loc[:, "Revenue"], threshold_payoff)

    # NOTE: One could also calculate the EVPI from the variable like so:
    #       lin_units_array = np.linspace(units_min, units_max, n_points_lin_array)
    #       calculate_evpi(distributions["Units"], lin_loss_array, lin_units_array)
    #       However, when dealing with multiple variables, estimating EVPI from the
    #       revenue is the simplest
    evpi = calculate_evpi(distributions["Revenue"], lin_revenue_array, lin_loss_array)
    print(f"Expected value of perfect information for 'Revenue' variable: {evpi:.1f}\n")

    plot_sample_histogram_with_threshold(
        samples_df.loc[:, "Units"], threshold_units, "Units", "#", "units"
    )
    plot_loss_functions(lin_revenue_array, lin_loss_array, threshold_payoff, "units")
    plot_pdf_cdf_and_incremental_probability(
        distributions["Revenue"], lin_revenue_array, threshold_payoff, "units"
    )
    plot_eol_with_threshold(
        get_eol_from_distribution(
            distributions["Revenue"], lin_revenue_array, lin_loss_array
        ),
        lin_revenue_array,
        threshold_payoff,
        "units",
    )


if __name__ == "__main__":
    main()
