"""Test module for the functions in the `utility/dict.py` module.

This module contains unit tests for the functions implemented in the `dict.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

from app.utility.numbers import *


class TestGetPowerHtml:

    def test_single_value_zero(self):
        """Test when the value is zero."""

        result = get_power_html(0)
        assert result == "0"

    def test_single_value_positive(self):
        """Test when the value is a positive number with fixed decimal precision."""

        result = get_power_html(1234.5646, 2)
        assert result == "1.23 &#10005; 10<sup>3</sup>"

    def test_single_value_negative(self):
        """Test when the value is a negative number."""

        result = get_power_html(-1234.56, 1)
        assert result == "-1.2 &#10005; 10<sup>3</sup>"

    def test_single_value_exponent_only(self):
        """Test when the value is a positive number with exponent only."""
        result = get_power_html(123456789, None)
        assert result == "&#10005; 10<sup>8</sup>"

    def test_single_value_all_significant_digits(self):
        """Test when n is -1 to show all significant digits."""
        result = get_power_html(1234.56789, -1)
        assert result == "1.23457 &#10005; 10<sup>3</sup>"

    def test_single_value_with_small_exponent(self):
        """Test when the value has a small exponent (e.g., a number less than 1)."""
        result = get_power_html(0.000123, 3)
        assert result == "1.230 &#10005; 10<sup>-4</sup>"

    def test_list_input(self):
        """Test when the input is a list of values."""
        values = [1234.56, 0, 987654.321]
        result = get_power_html(values, 2)
        expected = ["1.23 &#10005; 10<sup>3</sup>", "0", "9.88 &#10005; 10<sup>5</sup>"]
        assert result == expected

    def test_empty_list(self):
        """Test when the input is an empty list."""
        result = get_power_html([], 2)
        assert result == []

    def test_base0(self):

        result = get_power_html(1.4, 1)
        assert result == "1.4"

        result = get_power_html(-1.4, 1)
        assert result == "-1.4"


class TestToScientific:
    """Test cases for the to_scientific function"""

    def test_single_float(self) -> None:
        """Test single float to scientific notation"""
        result = to_scientific(1.4e-4)
        assert result == "1.4E-04"

    def test_single_integer(self) -> None:
        """Test single integer to scientific notation"""
        result = to_scientific(5000)
        assert result == "5E+03"

    def test_none_input(self) -> None:
        """Test None input"""
        result = to_scientific(None)
        assert result == ""

    def test_single_negative_float(self) -> None:
        """Test single negative float to scientific notation"""
        result = to_scientific(-1.4e-4)
        assert result == "-1.4E-04"

    def test_single_float_no_trailing_zeros(self) -> None:
        """Test float with no trailing zeros"""
        result = to_scientific(1.0e5)
        assert result == "1E+05"

    def test_list_of_floats(self) -> None:
        """Test list of floats to scientific notation"""
        result = to_scientific([1e-4, 1e-5])
        assert result == "1E-04, 1E-05"

    def test_list_of_mixed_numbers(self) -> None:
        """Test list of integers and floats to scientific notation"""
        result = to_scientific([100, 1.4e-3, 5])
        assert result == "1E+02, 1.4E-03, 5E+00"

    def test_empty_list(self) -> None:
        """Test empty list input"""
        result = to_scientific([])
        assert result == ""

    def test_large_number(self) -> None:
        """Test large number to scientific notation"""
        result = to_scientific(1e100)
        assert result == "1E+100"

    def test_edge_case_zero(self) -> None:
        """Test zero as an edge case"""
        result = to_scientific(0)
        assert result == "0"

    def test_single_integer_with_zero_trailing(self) -> None:
        """Test integer with trailing zeros"""
        result = to_scientific(5000000)
        assert result == "5E+06"


class TestGetConcentrationsHtml:

    def test_basic_case(self) -> None:
        """Test with a basic list of initial carrier concentrations."""
        N0s = [1.6e16, 2e17, 3e18]
        result = get_concentrations_html(N0s)
        expected = [
            "N<sub>0</sub> = 1.6 &#10005; 10<sup>16</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 2 &#10005; 10<sup>17</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 3 &#10005; 10<sup>18</sup> cm<sup>-3</sup>",
        ]
        assert result == expected
