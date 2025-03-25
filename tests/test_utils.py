import pytest

from app.utils import *


class TestMergeDicts:

    def test_empty_input(self):
        """Test with empty input."""
        assert merge_dicts() == {}

    def test_single_dict(self):
        """Test with a single dictionary."""
        assert merge_dicts({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_two_dicts_no_overlap(self):
        """Test with two dictionaries that don't have overlapping keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        assert merge_dicts(dict1, dict2) == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_two_dicts_with_overlap(self):
        """Test with two dictionaries that have overlapping keys."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        assert merge_dicts(dict1, dict2) == {"a": 1, "b": 2, "c": 4}

    def test_nested_dictionaries(self):
        """Test with nested dictionaries."""
        dict1 = {"a": {"nested": 1}}
        dict2 = {"a": {"different": 2}, "b": 3}
        # The entire nested dict should be kept from dict1
        assert merge_dicts(dict1, dict2) == {"a": {"nested": 1}, "b": 3}


class TestNormalizeToUnit:

    def test_zero(self):
        """Test normalization of zero."""
        value, exponent = normalize_to_unit(0.0)
        assert value == 0.0
        assert exponent == 0

    def test_one(self):
        """Test normalization of 1."""
        value, exponent = normalize_to_unit(1.0)
        assert value == 1.0
        assert exponent == 0

    def test_negative_one(self):
        """Test normalization of -1."""
        value, exponent = normalize_to_unit(-1.0)
        assert value == -1.0
        assert exponent == 0

    def test_smaller_than_one(self):
        """Test normalization of number smaller than 1 but greater than 0.1."""
        value, exponent = normalize_to_unit(0.5)
        assert value == 0.5
        assert exponent == 0

    def test_smaller_than_point_one(self):
        """Test normalization of number smaller than 0.1."""
        value, exponent = normalize_to_unit(0.05)
        assert value == 0.5
        assert exponent == -1

    def test_larger_than_one(self):
        """Test normalization of number larger than 1."""
        value, exponent = normalize_to_unit(42.0)
        assert pytest.approx(value) == 0.42
        assert exponent == 2

    def test_very_large_number(self):
        """Test normalization of a very large number."""
        value, exponent = normalize_to_unit(1.433364345e9)
        assert pytest.approx(value) == 0.1433364345
        assert exponent == 10

    def test_very_small_number(self):
        """Test normalization of a very small number."""
        value, exponent = normalize_to_unit(3.5e-8)
        assert pytest.approx(value) == 0.35
        assert exponent == -7

    def test_negative_small_number(self):
        """Test normalization of a negative small number."""
        value, exponent = normalize_to_unit(-0.0025)
        assert pytest.approx(value) == -0.25
        assert exponent == -2

    def test_negative_large_number(self):
        """Test normalization of a negative large number."""
        value, exponent = normalize_to_unit(-12345.0)
        assert pytest.approx(value) == -0.12345
        assert exponent == 5

    def test_exactly_point_one(self):
        """Test normalization of exactly 0.1."""
        value, exponent = normalize_to_unit(0.1)
        assert value == 0.1
        assert exponent == 0

    def test_almost_point_one(self):
        """Test normalization of a number very close to 0.1."""
        value, exponent = normalize_to_unit(0.099999)
        assert pytest.approx(value) == 0.99999
        assert exponent == -1

    def test_scientific_notation_positive(self):
        """Test with number in scientific notation (positive exponent)."""
        value, exponent = normalize_to_unit(2.5e4)
        assert pytest.approx(value) == 0.25
        assert exponent == 5

    def test_scientific_notation_negative(self):
        """Test with number in scientific notation (negative exponent)."""
        value, exponent = normalize_to_unit(14e-6)
        assert pytest.approx(value) == 0.14
        assert exponent == -4


class TestFilterDicts:

    def test_empty_inequations(self):
        """Test with empty list of inequations."""
        dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        result = filter_dicts(dicts, [])
        assert result == dicts

    def test_basic_less_than(self):
        """Test basic less than operation."""
        dicts = [{"a": 4, "b": 2}, {"a": 2, "b": 3}, {"a": 8, "b": 4}, {"a": 4, "b": 5}]
        result = filter_dicts(dicts, ["b < a"])
        assert result == [{"a": 4, "b": 2}, {"a": 8, "b": 4}]

    def test_basic_greater_than(self):
        """Test basic greater than operation."""
        dicts = [{"a": 4, "b": 2}, {"a": 2, "b": 3}, {"a": 8, "b": 4}, {"a": 4, "b": 5}]
        result = filter_dicts(dicts, ["b > a"])
        assert result == [{"a": 2, "b": 3}, {"a": 4, "b": 5}]

    def test_multiple_inequations(self):
        """Test multiple inequations."""
        dicts = [{"a": 4, "b": 2, "c": 3}, {"a": 2, "b": 3, "c": 1}, {"a": 8, "b": 4, "c": 5}, {"a": 4, "b": 5, "c": 2}]
        result = filter_dicts(dicts, ["b < a", "c > b"])
        assert result == [{"a": 4, "b": 2, "c": 3}, {"a": 8, "b": 4, "c": 5}]

    def test_fixed_values_less_than(self):
        """Test with fixed values and less than operation."""
        dicts = [{"a": 4}, {"a": 2}, {"a": 8}, {"a": 4}]
        result = filter_dicts(dicts, ["a < b"], {"b": 4.1})
        assert result == [{"a": 4}, {"a": 2}, {"a": 4}]

    def test_fixed_values_greater_than(self):
        """Test with fixed values and greater than operation."""
        dicts = [{"a": 4}, {"a": 2}, {"a": 8}, {"a": 4}]
        result = filter_dicts(dicts, ["a > b"], {"b": 4.1})
        assert result == [{"a": 8}]

    def test_reversed_fixed_values(self):
        """Test with fixed values on the left side of the inequation."""
        dicts = [{"b": 4}, {"b": 2}, {"b": 8}, {"b": 4}]
        result = filter_dicts(dicts, ["a < b"], {"a": 4.1})
        assert result == [{"b": 8}]

    def test_whitespace_handling(self):
        """Test with whitespace in inequations."""
        dicts = [{"a": 4, "b": 2}, {"a": 2, "b": 3}]
        result = filter_dicts(dicts, ["b  <  a"])  # Extra whitespace
        assert result == [{"a": 4, "b": 2}]

    def test_equal_values(self):
        """Test with equal values (should not be included)."""
        dicts = [{"a": 4, "b": 4}, {"a": 2, "b": 3}]
        result = filter_dicts(dicts, ["b < a"])
        assert result == []

        result = filter_dicts(dicts, ["b > a"])
        assert result == [{"a": 2, "b": 3}]

    def test_nonexistent_key(self):
        """Test with nonexistent key (should not affect the result)."""
        dicts = [{"a": 4, "b": 2}, {"a": 2, "b": 3}]
        result = filter_dicts(dicts, ["c < a"])
        assert result == dicts

    def test_multiple_fixed_values(self):
        """Test with multiple fixed values."""
        dicts = [{"a": 4}, {"a": 2}, {"a": 8}, {"a": 4}]
        result = filter_dicts(dicts, ["a > b", "a < c"], {"b": 3, "c": 5})
        assert result == [{"a": 4}, {"a": 4}]

    def test_combined_inequations(self):
        """Test combining multiple inequations with <= and >=."""
        dicts = [{"a": 4, "b": 4, "c": 5}, {"a": 2, "b": 3, "c": 4}, {"a": 8, "b": 4, "c": 8}]
        result = filter_dicts(dicts, ["b <= a", "c >= b"])
        assert result == [{"a": 4, "b": 4, "c": 5}, {"a": 8, "b": 4, "c": 8}]

    def test_no_matching_results(self):
        """Test when no dictionaries match the condition."""
        dicts = [{"a": 1, "b": 5}, {"a": 3, "b": 7}]
        result = filter_dicts(dicts, ["a >= b"])
        assert result == []


class TestKeepFunctionKwargs:

    def test_basic_lambda(self):
        """Test with a basic lambda function."""
        function = lambda a, b: a + b
        kwargs = {"a": 1, "b": 2, "c": 3}
        result = keep_function_kwargs(function, kwargs)
        assert result == {"a": 1, "b": 2}

    def test_no_matching_args(self):
        """Test with no matching arguments."""
        function = lambda a, b: a + b
        kwargs = {"c": 3, "d": 4}
        result = keep_function_kwargs(function, kwargs)
        assert result == {}

    def test_all_matching_args(self):
        """Test with all arguments matching."""
        function = lambda a, b: a + b
        kwargs = {"a": 1, "b": 2}
        result = keep_function_kwargs(function, kwargs)
        assert result == {"a": 1, "b": 2}

    def test_empty_kwargs(self):
        """Test with empty kwargs dictionary."""
        function = lambda a, b: a + b
        kwargs = {}
        result = keep_function_kwargs(function, kwargs)
        assert result == {}

    def test_function_with_no_args(self):
        """Test with a function that has no arguments."""
        function = lambda: 42
        kwargs = {"a": 1, "b": 2}
        result = keep_function_kwargs(function, kwargs)
        assert result == {}

    def test_regular_function(self):
        """Test with a regular function definition."""

        def add(x, y, z=0):
            return x + y + z

        kwargs = {"x": 5, "y": 10, "a": 15}
        result = keep_function_kwargs(add, kwargs)
        assert result == {"x": 5, "y": 10}

    def test_function_with_args_kwargs(self):
        """Test with a function that has *args and **kwargs parameters."""

        def flexible_function(_a, _b, *_args, **_kwargs):
            pass

        kwargs = {"_a": 1, "_b": 2, "c": 3, "d": 4}
        result = keep_function_kwargs(flexible_function, kwargs)
        assert result == {"_a": 1, "_b": 2}

    def test_class_method(self):
        """Test with a class method."""

        class TestClass:
            a = 1

            def method(self, x, y):
                return x + y + self.a

        instance = TestClass()
        kwargs = {"x": 5, "y": 10, "z": 15}
        result = keep_function_kwargs(instance.method, kwargs)
        # Note: 'self' is not included in the args for instance methods when using inspect
        assert result == {"x": 5, "y": 10}

    def test_static_method(self):
        """Test with a static method."""

        class TestClass:
            @staticmethod
            def static_method(a, b):
                return a * b

        kwargs = {"a": 5, "b": 10, "c": 15}
        result = keep_function_kwargs(TestClass.static_method, kwargs)
        assert result == {"a": 5, "b": 10}


class TestGetPowerText:

    def test_zero_value(self):
        """Test with zero value."""
        result = get_power_html(0)
        assert result == "0"

    def test_default_decimals(self):
        """Test with default number of decimals (3)."""
        result = get_power_html(1.3e5)
        assert result == "1.300 &#10005; 10<sup>5</sup>"

    def test_specific_decimals(self):
        """Test with a specific number of decimals."""
        result = get_power_html(1.3e13, 2)
        assert result == "1.30 &#10005; 10<sup>13</sup>"

    def test_no_mantissa(self):
        """Test with None for n to get just the exponent part."""
        result = get_power_html(1.34e13, None)
        assert result == "10<sup>13</sup>"

    def test_auto_format(self):
        """Test with -1 for n to use %g format."""
        result = get_power_html(1.34e13, -1)
        assert result == "1.34 &#10005; 10<sup>13</sup>"

    def test_different_decimals(self):
        """Test with different decimal precisions."""
        result = get_power_html(1.56789e7, 4)
        assert result == "1.5679 &#10005; 10<sup>7</sup>"

        result = get_power_html(1.56789e7, 1)
        assert result == "1.6 &#10005; 10<sup>7</sup>"

        result = get_power_html(1.56789e7, 0)
        assert result == "2 &#10005; 10<sup>7</sup>"

    def test_small_numbers(self):
        """Test with small numbers (negative exponents)."""
        result = get_power_html(1.23e-5, 2)
        assert result == "1.23 &#10005; 10<sup>-5</sup>"

    def test_rounding(self):
        """Test rounding behavior."""
        # This should round up
        result = get_power_html(1.9999e6, 3)
        assert result == "2.000 &#10005; 10<sup>6</sup>"

    def test_integer_values(self):
        """Test with integer input values."""
        result = get_power_html(1000, 2)
        assert result == "1.00 &#10005; 10<sup>3</sup>"

    def test_near_power_boundaries(self):
        """Test values near power of 10 boundaries."""
        # Just below 10^3
        result = get_power_html(999.9, 1)
        assert result == "1.0 &#10005; 10<sup>3</sup>"

        # Exactly 10^3
        result = get_power_html(1000, 1)
        assert result == "1.0 &#10005; 10<sup>3</sup>"

        # Just above 10^3
        result = get_power_html(1001, 1)
        assert result == "1.0 &#10005; 10<sup>3</sup>"

    def test_large_numbers(self):
        """Test with very large numbers."""
        result = get_power_html(1.23e100, 2)
        assert result == "1.23 &#10005; 10<sup>100</sup>"


class TestMatrixToString:

    def test_basic_conversion(self):
        arrays = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

    def test_no_header(self):
        arrays = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
        result = matrix_to_string(arrays)
        expected = "1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

    def test_single_column(self):
        arrays = [np.array([1.2, 2, 5])]
        result = matrix_to_string(arrays, ["A"])
        expected = "A\n1.20000E+00\n2.00000E+00\n5.00000E+00"
        assert result == expected

    def test_mixed_lengths(self):
        arrays = [np.array([1.2, 2]), np.array([1.6])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,"
        assert result == expected

    def test_all_empty(self):
        arrays = [np.array([]), np.array([])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        assert result == "A,B\n"

    def test_no_trailing_comma(self):
        arrays = [np.array([1.2, 2]), np.array([1.6, 2])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00"
        assert result == expected


class TestToScientific:
    """Test cases for the to_scientific function"""

    def test_single_float(self):
        """Test single float to scientific notation"""
        result = to_scientific(1.4e-4)
        assert result == "1.4E-04"

    def test_single_integer(self):
        """Test single integer to scientific notation"""
        result = to_scientific(5000)
        assert result == "5.E+03"

    def test_none_input(self):
        """Test None input"""
        result = to_scientific(None)
        assert result == ""

    def test_single_negative_float(self):
        """Test single negative float to scientific notation"""
        result = to_scientific(-1.4e-4)
        assert result == "-1.4E-04"

    def test_single_float_no_trailing_zeros(self):
        """Test float with no trailing zeros"""
        result = to_scientific(1.0e5)
        assert result == "1.E+05"

    def test_list_of_floats(self):
        """Test list of floats to scientific notation"""
        result = to_scientific([1e-4, 1e-5])
        assert result == "1.E-04, 1.E-05"

    def test_list_of_mixed_numbers(self):
        """Test list of integers and floats to scientific notation"""
        result = to_scientific([100, 1.4e-3, 5])
        assert result == "1.E+02, 1.4E-03, 5.E+00"

    def test_empty_list(self):
        """Test empty list input"""
        result = to_scientific([])
        assert result == ""

    def test_large_number(self):
        """Test large number to scientific notation"""
        result = to_scientific(1e100)
        assert result == "1.E+100"

    def test_edge_case_zero(self):
        """Test zero as an edge case"""
        result = to_scientific(0)
        assert result == "0.E+00"

    def test_single_integer_with_zero_trailing(self):
        """Test integer with trailing zeros"""
        result = to_scientific(5000000)
        assert result == "5.E+06"


class TestGenerateDownloadLink:

    def test_basic_functionality(self):
        """Test basic functionality of generate_download_link."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        header = ["A", "B", "C"]
        text = "Download Data"
        result = generate_download_link(data, header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_no_header(self):
        """Test when no header is provided."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        text = "Download Data"
        result = generate_download_link(data, None, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_empty_matrix(self):
        """Test when an empty matrix is provided."""
        data = np.array([[], []])
        text = "Download Data"
        result = generate_download_link(data, None, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_single_row_matrix(self):
        """Test with a matrix containing a single row."""
        data = np.array([[1, 2, 3]])
        header = ["A", "B", "C"]
        text = "Download Data"
        result = generate_download_link(data, header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_single_column_matrix(self):
        """Test with a matrix containing a single column."""
        data = np.array([[1], [4]])
        header = ["A"]
        text = "Download Data"
        result = generate_download_link(data, header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_with_special_characters(self):
        """Test if the function handles special characters in the header and text."""
        data = np.array([[1, 2], [3, 4]])
        header = ["Col@1", "Col#2"]
        text = "Download with Special Characters"
        result = generate_download_link(data, header, text)

        # Extract the base64 string from the result
        base64_string = result.split("base64,")[1].split('"')[0]

        # Decode the base64 string to get the original string
        decoded_string = base64.b64decode(base64_string).decode()

        # Now check if the decoded string contains the header with special characters
        assert "Col@1" in decoded_string
        assert "Col#2" in decoded_string
        assert "Download with Special Characters" in result

    def test_no_text_provided(self):
        """Test if no text is provided (empty string)."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        header = ["A", "B", "C"]
        result = generate_download_link(data, header, "")
        assert '<a href="data:text/csv;base64,' in result
        assert 'href="data:text/csv;base64,' in result
        assert "Download" not in result  # Should not have any text if empty

    def test_large_data(self):
        """Test with large data to check performance (no specific checks)."""
        data = np.random.rand(100, 100)
        header = [f"Col{i}" for i in range(100)]
        result = generate_download_link(data, header, "Download Large Data")
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Large Data" in result

    def test_b64_encoding(self):
        """Test to ensure base64 encoding is correct."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        header = ["A", "B", "C"]
        result = generate_download_link(data, header, "Test Encoding")
        # Check if base64 encoding exists within the result
        assert "base64," in result


class TestListToDict:

    def test_basic_case(self):
        """Test with a basic list of dictionaries."""
        dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
        result = list_to_dict(dicts)
        expected = {"a": [1, 3, 5], "b": [2, 4, 6]}
        assert result == expected

    def test_single_element_dict(self):
        """Test with a list containing a single dictionary."""
        dicts = [{"a": 1, "b": 2}]
        result = list_to_dict(dicts)
        expected = {"a": [1], "b": [2]}
        assert result == expected

    def test_missing_key(self):
        """Test when a dictionary in the list is missing a key."""
        dicts = [{"a": 1, "b": 2}, {"a": 3}, {"a": 5, "b": 6}]
        with pytest.raises(KeyError):
            list_to_dict(dicts)

    def test_non_dict_elements(self):
        """Test when the input contains non-dictionary elements."""
        dicts = [{"a": 1, "b": 2}, [3, 4], {"a": 5, "b": 6}]
        with pytest.raises(TypeError):
            list_to_dict(dicts)

    def test_empty_dicts(self):
        """Test when the dictionaries in the list are empty."""
        dicts = [{}, {}, {}]
        result = list_to_dict(dicts)
        expected = {}  # No keys, so the result should be an empty dict
        assert result == expected


class TestGetPowerLabels:

    def test_basic_case(self):
        """Test with a basic list of initial carrier concentrations."""
        N0s = [1.6e16, 2e17, 3e18]
        result = get_power_labels(N0s)
        expected = [
            "N<sub>0</sub> = 1.6 &#10005; 10<sup>16</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 2 &#10005; 10<sup>17</sup> cm<sup>-3</sup>",
            "N<sub>0</sub> = 3 &#10005; 10<sup>18</sup> cm<sup>-3</sup>",
        ]
        assert result == expected


class TestProcessData:

    @pytest.fixture
    def gaussian_data(self):
        """Fixture to generate a Gaussian dataset with 20 data points."""
        # Generate Gaussian data with mean 0 and standard deviation 1
        n_points = 10
        x_data = np.linspace(-10, 5, n_points)  # x values from -5 to 5
        y_data = np.exp(-0.5 * (x_data**2))  # Gaussian function: e^(-x^2/2)
        return [x_data, x_data], [y_data, y_data * 2]

    def test_no_normalisation(self, gaussian_data):
        """Test with no normalisation."""
        xs_data, ys_data = gaussian_data

        # Process data without normalisation
        result_xs, result_ys = process_data(xs_data, ys_data, normalise=False)

        # Check if x values are reduced and shifted, and y values are reduced
        expected_xs = [np.array([0.0, 1.66666667, 3.33333333, 5.0]), np.array([0.0, 1.66666667, 3.33333333, 5.0])]
        expected_ys = [
            np.array([1.00000000e00, 2.49352209e-01, 3.86592014e-03, 3.72665317e-06]),
            np.array([2.00000000e00, 4.98704418e-01, 7.73184028e-03, 7.45330634e-06]),
        ]

        assert np.allclose(result_xs, expected_xs)
        assert np.allclose(result_ys, expected_ys)

    def test_with_normalisation(self, gaussian_data):
        """Test with normalisation."""
        xs_data, ys_data = gaussian_data

        # Process data with normalisation
        result_xs, result_ys = process_data(xs_data, ys_data, normalise=True)

        # Check if x values are reduced and shifted, and y values are normalised
        expected_xs = [np.array([0.0, 1.66666667, 3.33333333, 5.0]), np.array([0.0, 1.66666667, 3.33333333, 5.0])]
        expected_ys = [
            np.array([1.00000000e00, 2.49352209e-01, 3.86592014e-03, 3.72665317e-06]),
            np.array([1.00000000e00, 2.49352209e-01, 3.86592014e-03, 3.72665317e-06]),
        ]

        assert np.allclose(result_xs, expected_xs)
        assert np.allclose(result_ys, expected_ys)

    def test_edge_case_empty_data(self):
        """Test with empty input data."""
        xs_data = []
        ys_data = []

        # Process empty data
        result_xs, result_ys = process_data(xs_data, ys_data, normalise=False)

        # Check that the result is also empty
        assert result_xs == []
        assert result_ys == []


class TestGetDataIndex:

    def test_no_delimiter(self):
        """Test without a delimiter (default None)"""
        content = ["header", "data starts here", "1 2 3", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with float data is at index 2

    def test_with_delimiter(self):
        """Test with a specified delimiter"""
        content = ["header", "data starts here", "1,2,3", "4,5,6"]
        result = get_data_index(content, delimiter=",")
        assert result == 2  # the first line with float data is at index 2

    def test_no_data(self):
        """Test case when there are no float data lines"""
        content = ["header", "some text", "more text"]
        result = get_data_index(content)
        assert result is None  # No line contains float data

    def test_empty_list(self):
        """Test with an empty list"""
        content = []
        result = get_data_index(content)
        assert result is None  # No data in the list

    def test_mixed_data(self):
        """Test with mixed data (some numeric and some non-numeric)"""
        content = ["header", "text", "1 2 3", "text again", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with numeric data is at index 2

    def test_non_matching_delimiter(self):
        """Test with a delimiter that doesn't match any line"""
        content = ["header", "text", "1 2 3", "4 5 6"]
        result = get_data_index(content, delimiter=",")
        assert result is None  # no lines with comma as delimiter


class TestLoadData:

    def test_x_y1_y2_y3_format(self):
        """Test the X/Y1/Y2/Y3... format where all columns have the same length"""
        content = "1 2 3 4\n5 6 7 8\n9 10 11 12\n"  # X/Y1/Y2/Y3 data format
        delimiter = " "
        data_format = "X/Y1/Y2/Y3..."

        # Simulate loading data from a file
        xs_data, ys_data = load_data(content.encode(), delimiter, data_format)

        # Expected results
        expected_xs = [np.array([1.0, 5.0, 9.0]), np.array([1.0, 5.0, 9.0]), np.array([1.0, 5.0, 9.0])]
        expected_ys = [np.array([2.0, 6.0, 10.0]), np.array([3.0, 7.0, 11.0]), np.array([4.0, 8.0, 12.0])]

        assert np.array_equal(xs_data, expected_xs)
        assert np.array_equal(ys_data, expected_ys)

    def test_x1_y1_x2_y2_format(self):
        """Test the X1/Y1/X2/Y2... format where all columns have the same length"""
        content = "1,2,3,4\n5,6,7,8\n9,10,,\n"  # X/Y1/Y2/Y3 data format
        delimiter = ","
        data_format = "X1/Y1/X2/Y2..."

        # Simulate loading data from a file
        xs_data, ys_data = load_data(content.encode(), delimiter, data_format)

        # Expected results
        expected_xs = [np.array([1.0, 5.0, 9.0]), np.array([3.0, 7.0])]
        expected_ys = [np.array([2.0, 6.0, 10.0]), np.array([4.0, 8.0])]

        for x_array, expected_x_array in zip(xs_data, expected_xs):
            assert np.array_equal(x_array, expected_x_array)
        for y_array, expected_y_array in zip(ys_data, expected_ys):
            assert np.array_equal(y_array, expected_y_array)


class TestAreListsIdentical:

    def test_identical_basic_lists(self):
        list1 = [1, 2, 3, "a"]
        list2 = [1, 2, 3, "a"]
        assert are_lists_identical(list1, list2) is True

    def test_different_basic_lists(self):
        list1 = [1, 2, 3, "a"]
        list2 = [1, 2, 4, "a"]
        assert are_lists_identical(list1, list2) is False

    def test_identical_lists_with_ndarrays(self):
        list1 = [1, np.array([1, 2, 3]), "a"]
        list2 = [1, np.array([1, 2, 3]), "a"]
        assert are_lists_identical(list1, list2) is True

    def test_different_lists_with_ndarrays(self):
        list1 = [1, np.array([1, 2, 3]), "a"]
        list2 = [1, np.array([1, 2, 4]), "a"]
        assert are_lists_identical(list1, list2) is False

    def test_nested_lists(self):
        list1 = [1, [2, 3], np.array([1, 2, 3])]
        list2 = [1, [2, 3], np.array([1, 2, 3])]
        assert are_lists_identical(list1, list2) is True

    def test_different_nested_lists(self):
        list1 = [1, [2, 3], np.array([1, 2, 3])]
        list2 = [1, [2, 4], np.array([1, 2, 3])]
        assert are_lists_identical(list1, list2) is False

    def test_lists_with_different_lengths(self):
        list1 = [1, 2, 3]
        list2 = [1, 2]
        assert are_lists_identical(list1, list2) is False

    def test_empty_lists(self):
        list1 = []
        list2 = []
        assert are_lists_identical(list1, list2) is True

    def test_lists_with_non_matching_ndarrays(self):
        list1 = [np.array([1, 2, 3])]
        list2 = [np.array([1, 3, 2])]
        assert are_lists_identical(list1, list2) is False
