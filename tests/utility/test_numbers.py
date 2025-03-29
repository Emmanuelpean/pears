from app.utility.numbers import *


class TestGetPowerText:

    def test_zero_value(self) -> None:
        """Test with zero value."""
        result = get_power_html(0)
        assert result == "0"

    def test_default_decimals(self) -> None:
        """Test with default number of decimals (3)."""
        result = get_power_html(1.3e5)
        assert result == "1.300 &#10005; 10<sup>5</sup>"

    def test_specific_decimals(self) -> None:
        """Test with a specific number of decimals."""
        result = get_power_html(1.3e13, 2)
        assert result == "1.30 &#10005; 10<sup>13</sup>"

    def test_no_mantissa(self) -> None:
        """Test with None for n to get just the exponent part."""
        result = get_power_html(1.34e13, None)
        assert result == "&#10005; 10<sup>13</sup>"

    def test_auto_format(self) -> None:
        """Test with -1 for n to use %g format."""
        result = get_power_html(1.34e13, -1)
        assert result == "1.34 &#10005; 10<sup>13</sup>"

    def test_different_decimals(self) -> None:
        """Test with different decimal precisions."""
        result = get_power_html(1.56789e7, 4)
        assert result == "1.5679 &#10005; 10<sup>7</sup>"

        result = get_power_html(1.56789e7, 1)
        assert result == "1.6 &#10005; 10<sup>7</sup>"

        result = get_power_html(1.56789e7, 0)
        assert result == "2 &#10005; 10<sup>7</sup>"

    def test_small_numbers(self) -> None:
        """Test with small numbers (negative exponents)."""
        result = get_power_html(1.23e-5, 2)
        assert result == "1.23 &#10005; 10<sup>-5</sup>"

    def test_rounding(self) -> None:
        """Test rounding behavior."""
        # This should round up
        result = get_power_html(1.9999e6, 3)
        assert result == "2.000 &#10005; 10<sup>6</sup>"

    def test_integer_values(self) -> None:
        """Test with integer input values."""
        result = get_power_html(1000, 2)
        assert result == "1.00 &#10005; 10<sup>3</sup>"

    def test_near_power_boundaries(self) -> None:
        """Test values near power of 10 boundaries."""

        # Exactly 10^3
        result = get_power_html(1000, 1)
        assert result == "1.0 &#10005; 10<sup>3</sup>"

        # Just above 10^3
        result = get_power_html(1001, 1)
        assert result == "1.0 &#10005; 10<sup>3</sup>"

    def test_large_numbers(self) -> None:
        """Test with very large numbers."""
        result = get_power_html(1.23e100, 2)
        assert result == "1.23 &#10005; 10<sup>100</sup>"


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


class TestGetPowerLabels:

    def test_basic_case(self) -> None:
        """Test with a basic list of initial carrier concentrations."""
        N0s = [1.6e16, 2e17, 3e18]
        result = get_power_labels(N0s)
        expected = [
            "N<sub>0</sub> = 1.6 &#10005; 10<sup>16</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 2 &#10005; 10<sup>17</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 3 &#10005; 10<sup>18</sup> cm<sup>-3</sup>",
        ]
        assert result == expected
