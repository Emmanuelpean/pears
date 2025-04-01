"""Test module for the functions in the `fitting.py` module.

This module contains unit tests for the functions implemented in the `fitting.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import numpy as np
import pytest

from app.utility.data import are_close
from app.fitting import Fit, normalize_to_unit


class TestNormalizeToUnit:

    def test_zero(self) -> None:
        """Test normalization of zero."""
        value, exponent = normalize_to_unit(0.0)
        assert value == 0.0
        assert exponent == 0

    def test_one(self) -> None:
        """Test normalization of 1."""
        value, exponent = normalize_to_unit(1.0)
        assert value == 1.0
        assert exponent == 0

    def test_negative_one(self) -> None:
        """Test normalization of -1."""
        value, exponent = normalize_to_unit(-1.0)
        assert value == -1.0
        assert exponent == 0

    def test_smaller_than_one(self) -> None:
        """Test normalization of number smaller than 1 but greater than 0.1."""
        value, exponent = normalize_to_unit(0.5)
        assert value == 0.5
        assert exponent == 0

    def test_smaller_than_point_one(self) -> None:
        """Test normalization of number smaller than 0.1."""
        value, exponent = normalize_to_unit(0.05)
        assert value == 0.5
        assert exponent == -1

    def test_larger_than_one(self) -> None:
        """Test normalization of number larger than 1."""
        value, exponent = normalize_to_unit(42.0)
        assert are_close(value, 0.42)
        assert exponent == 2

    def test_very_large_number(self) -> None:
        """Test normalization of a very large number."""
        value, exponent = normalize_to_unit(1.433364345e9)
        assert are_close(value, 0.1433364345)
        assert exponent == 10

    def test_very_small_number(self) -> None:
        """Test normalization of a very small number."""
        value, exponent = normalize_to_unit(3.5e-8)
        assert are_close(value, 0.35)
        assert exponent == -7

    def test_negative_small_number(self) -> None:
        """Test normalization of a negative small number."""
        value, exponent = normalize_to_unit(-0.0025)
        assert are_close(value, -0.25)
        assert exponent == -2

    def test_negative_large_number(self) -> None:
        """Test normalization of a negative large number."""
        value, exponent = normalize_to_unit(-12345.0)
        assert are_close(value, -0.12345)
        assert exponent == 5

    def test_exactly_point_one(self) -> None:
        """Test normalization of exactly 0.1."""
        value, exponent = normalize_to_unit(0.1)
        assert value == 0.1
        assert exponent == 0

    def test_almost_point_one(self) -> None:
        """Test normalization of a number very close to 0.1."""
        value, exponent = normalize_to_unit(0.099999)
        assert are_close(value, 0.99999)
        assert exponent == -1

    def test_scientific_notation_positive(self) -> None:
        """Test with number in scientific notation (positive exponent)."""
        value, exponent = normalize_to_unit(2.5e4)
        assert are_close(value, 0.25)
        assert exponent == 5

    def test_scientific_notation_negative(self) -> None:
        """Test with number in scientific notation (negative exponent)."""
        value, exponent = normalize_to_unit(14e-6)
        assert are_close(value, 0.14)
        assert exponent == -4


class TestFit:

    @pytest.fixture
    def gaussian_data(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], callable, dict[str, float], list[str], list[dict[str, float]]]:
        """Gaussian data"""

        def gaussian(x: np.ndarray, a: float, x0: float, c: float) -> np.ndarray:
            """Gaussian function"""
            return a * np.exp(-((x - x0) ** 2) / (2 * c))

        xs_data = [np.linspace(-2, 5, 101)] * 3
        ys_data = [
            gaussian(xs_data[0], 1e8, 1, 0.5),
            gaussian(xs_data[0], 3e8, 1, 0.25),
            gaussian(xs_data[0], 5e8, 1, 0.25),
        ]
        p0 = dict(a=1e8, x0=0, c=0.3)
        detached_parameters = ["a"]
        fixed_parameters = [dict(c=0.5), dict(c=0.25), dict(c=0.25)]
        return xs_data, ys_data, gaussian, p0, detached_parameters, fixed_parameters

    @pytest.fixture
    def gaussian_fit(self, gaussian_data) -> Fit:
        """Gaussian fit object"""

        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = gaussian_data
        return Fit(xs_data, ys_data, function, p0, detached_parameters, fixed_parameters)

    def test_gaussian_creation(self, gaussian_fit) -> None:

        assert gaussian_fit.p0_mantissa == {"a": 1, "x0": 0.0}
        assert gaussian_fit.p0_factors == {"a": 8, "x0": 0}
        assert gaussian_fit.p0_list == [1.0, 0.0, 1.0, 1.0]
        assert gaussian_fit.bounds == {"a": [0, np.inf], "x0": [0, np.inf]}
        assert gaussian_fit.bounds_list == [[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf]]

    def test_gaussian_incorrect_fixed_parameters(self, gaussian_data) -> None:

        xs_data, ys_data, gaussian, p0, detached_parameters = gaussian_data[:-1]
        fixed_parameters = [dict(c=0.5), dict()]
        with pytest.raises(AssertionError):
            Fit(xs_data, ys_data, gaussian, p0, detached_parameters, fixed_parameters)

    def test_gaussian_list_to_dicts(self, gaussian_fit) -> None:

        expected = [
            {"a": 100000000.0, "x0": 0.0},
            {"a": 100000000.0, "x0": 0.0},
            {"a": 100000000.0, "x0": 0.0},
        ]
        assert are_close(gaussian_fit.list_to_dicts(gaussian_fit.p0_list), expected)

    def test_gaussian_list_to_dicts_no_fixed(self, gaussian_data) -> None:

        expected = [{"a": 100000000.0, "x0": 0.0}, {"a": 100000000.0, "x0": 0.0}, {"a": 100000000.0, "x0": 0.0}]
        xs_data, ys_data, function, p0, detached_parameters, fixed_parameters = gaussian_data
        fit = Fit(xs_data, ys_data, function, p0, [], fixed_parameters)
        assert are_close(fit.list_to_dicts(fit.p0_list), expected)

    def test_gaussian_error_function(self, gaussian_fit) -> None:

        expected = np.array([1819222.90846475, 2392860.49712516])
        assert are_close(gaussian_fit.error_function(gaussian_fit.p0_list)[:2], expected)

    def test_gaussian_fit(self, gaussian_fit) -> None:

        expected = [
            {"a": 100000000.0, "x0": 1.0, "c": 0.5},
            {"a": 300000000.0, "x0": 1.0, "c": 0.25},
            {"a": 500000000.0, "x0": 1.0, "c": 0.25},
        ]
        assert are_close(gaussian_fit.fit(), expected)

    def test_gaussian_calculate_fits(self, gaussian_fit) -> None:

        expected = np.array([12340.98040867, 18690.68861775])
        assert are_close(gaussian_fit.calculate_fits(gaussian_fit.fit())[0][:2], expected)

    def test_gaussian_calculate_rss(self, gaussian_fit) -> None:

        fits = gaussian_fit.calculate_fits(gaussian_fit.fit())
        assert are_close(gaussian_fit.calculate_rss(fits), 1)
