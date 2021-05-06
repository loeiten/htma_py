"""Tests that the scripts run without crashing."""

from scripts.chap_5_calibration_probability import main as chap_5_main
from scripts.chap_6_machine_lease import main as chap_6_main
from scripts.chap_7_units_production import main as chap_7_main
from scripts.discrete_eol import main as eol_main


def test_chap_5_calibration_probability():
    """Test the main function of chap_5_calibration_probability."""
    chap_5_main()


def test_chap_6_machine_lease():
    """Test the main function of chap_6_machine_lease."""
    chap_6_main()


def test_chap_7_units_production():
    """Test the main function of chap_7_units_production."""
    chap_7_main()


def test_discrete_eol():
    """Test the main function of discrete_eol."""
    eol_main()
