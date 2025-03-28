import pytest

from app.utility.data import *


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


class TestGenerateDownloadLink:

    def test_basic_functionality(self):
        """Test basic functionality of generate_download_link."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_no_header(self):
        """Test when no header is provided."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), None, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_with_special_characters(self):
        """Test if the function handles special characters in the header and text."""
        x_data = [np.array([1, 2])]
        y_data = [np.array([3, 4])]
        header = ["Col@1", "Col#2"]
        text = "Download with Special Characters"
        result = generate_download_link((x_data, y_data), header, text)

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
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        result = generate_download_link((x_data, y_data), header, "")
        assert '<a href="data:text/csv;base64,' in result
        assert 'href="data:text/csv;base64,' in result
        assert "Download" not in result  # Should not have any text if empty

    def test_large_data(self):
        """Test with large data to check performance (no specific checks)."""
        x_data = [np.random.rand(100)]
        y_data = [np.random.rand(100)]
        header = [f"Col{i}" for i in range(100)]
        result = generate_download_link((x_data, y_data), header, "Download Large Data")
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Large Data" in result

    def test_b64_encoding(self):
        """Test to ensure base64 encoding is correct."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        result = generate_download_link((x_data, y_data), header, "Test Encoding")
        # Check if base64 encoding exists within the result
        assert "base64," in result


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
    def test_identical_lists(self):
        assert are_identical([1, 2, 3], [1, 2, 3])

    def test_different_lists(self):
        assert not are_identical([1, 2, 3], [1, 2, 4])

    def test_nested_lists(self):
        assert are_identical([1, [2, 3], 4], [1, [2, 3], 4])
        assert not are_identical([1, [2, 3], 4], [1, [2, 4], 4])

    def test_numpy_arrays(self):
        assert are_identical(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert not are_identical(np.array([1.0, 2.0]), np.array([1.0, 3.0]))

    def test_allclose_with_atol(self):
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.00000001, 3.0])
        assert are_identical(arr1, arr2, rtol=1e-7)

    def test_identical_dicts(self):
        dict1 = {"a": 1, "b": [2, 3], "c": np.array([1.0, 2.0])}
        dict2 = {"a": 1, "b": [2, 3], "c": np.array([1.0, 2.0])}
        assert are_identical(dict1, dict2)

    def test_different_dicts(self):
        dict1 = {"a": 1, "b": [2, 3], "c": np.array([1.0, 2.0])}
        dict2 = {"a": 1, "b": [2, 4], "c": np.array([1.0, 2.0])}
        assert not are_identical(dict1, dict2)

    def test_dicts_with_allclose(self):
        dict1 = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
        dict2 = {"a": np.array([1.0, 2.00000001]), "b": np.array([3.0, 4.0])}
        assert are_identical(dict1, dict2, rtol=1e-7)

    def test_list_of_dicts(self):
        list1 = [{"a": 1, "b": np.array([1.0, 2.0])}, {"c": 3, "d": np.array([3.0, 4.0])}]
        list2 = [{"a": 1, "b": np.array([1.0, 2.0])}, {"c": 3, "d": np.array([3.0, 4.0])}]
        assert are_identical(list1, list2)

        list3 = [{"a": 1, "b": np.array([1.0, 2.0])}, {"c": 3, "d": np.array([3.0, 5.0])}]
        assert not are_identical(list1, list3)
