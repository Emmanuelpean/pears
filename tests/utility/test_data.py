"""Test module for the functions in the `utility/data.py` module.

This module contains unit tests for the functions implemented in the `data.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

import os
import tempfile

import pytest
from bs4 import BeautifulSoup

from app.utility.data import *


class TestMatrixToString:

    def test_basic_conversion(self) -> None:
        arrays = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

    def test_no_header(self) -> None:
        arrays = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
        result = matrix_to_string(arrays)
        expected = "1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00\n5.00000E+00,"
        assert result == expected

    def test_single_column(self) -> None:
        arrays = [np.array([1.2, 2, 5])]
        result = matrix_to_string(arrays, ["A"])
        expected = "A\n1.20000E+00\n2.00000E+00\n5.00000E+00"
        assert result == expected

    def test_mixed_lengths(self) -> None:
        arrays = [np.array([1.2, 2]), np.array([1.6])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,"
        assert result == expected

    def test_all_empty(self) -> None:
        arrays = [np.array([]), np.array([])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        assert result == "A,B\n"

    def test_no_trailing_comma(self) -> None:
        arrays = [np.array([1.2, 2]), np.array([1.6, 2])]
        header = ["A", "B"]
        result = matrix_to_string(arrays, header)
        expected = "A,B\n1.20000E+00,1.60000E+00\n2.00000E+00,2.00000E+00"
        assert result == expected


class TestGenerateDownloadLink:

    def test_basic_functionality(self) -> None:
        """Test basic functionality of generate_download_link."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), header, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_no_header(self) -> None:
        """Test when no header is provided."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        text = "Download Data"
        result = generate_download_link((x_data, y_data), None, text)
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Data" in result

    def test_with_special_characters(self) -> None:
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

    def test_no_text_provided(self) -> None:
        """Test if no text is provided (empty string)."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        result = generate_download_link((x_data, y_data), header, "")
        assert '<a href="data:text/csv;base64,' in result
        assert 'href="data:text/csv;base64,' in result
        assert "Download" not in result  # Should not have any text if empty

    def test_large_data(self) -> None:
        """Test with large data to check performance (no specific checks)."""
        x_data = [np.random.rand(100)]
        y_data = [np.random.rand(100)]
        header = [f"Col{i}" for i in range(100)]
        result = generate_download_link((x_data, y_data), header, "Download Large Data")
        assert '<a href="data:text/csv;base64,' in result
        assert "Download Large Data" in result

    def test_b64_encoding(self) -> None:
        """Test to ensure base64 encoding is correct."""
        x_data = [np.array([1, 2, 3])]
        y_data = [np.array([4, 5, 6])]
        header = ["X", "Y"]
        result = generate_download_link((x_data, y_data), header, "Test Encoding")
        # Check if base64 encoding exists within the result
        assert "base64," in result


class TestProcessData:

    @pytest.fixture
    def gaussian_data(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Fixture to generate a Gaussian dataset with 20 data points."""
        # Generate Gaussian data with mean 0 and standard deviation 1
        n_points = 10
        x_data = np.linspace(-10, 5, n_points)  # x values from -5 to 5
        y_data = np.exp(-0.5 * (x_data**2))  # Gaussian function: e^(-x^2/2)
        return [x_data, x_data], [y_data, y_data * 2]

    def test_no_normalisation(self, gaussian_data) -> None:
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

    def test_with_normalisation(self, gaussian_data) -> None:
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

    def test_edge_case_empty_data(self) -> None:
        """Test with empty input data."""
        xs_data = []
        ys_data = []

        # Process empty data
        result_xs, result_ys = process_data(xs_data, ys_data, normalise=False)

        # Check that the result is also empty
        assert result_xs == []
        assert result_ys == []


class TestGetDataIndex:

    def test_no_delimiter(self) -> None:
        """Test without a delimiter (default None)"""
        content = ["header", "data starts here", "1 2 3", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with float data is at index 2

    def test_with_delimiter(self) -> None:
        """Test with a specified delimiter"""
        content = ["header", "data starts here", "1,2,3", "4,5,6"]
        result = get_data_index(content, delimiter=",")
        assert result == 2  # the first line with float data is at index 2

    def test_no_data(self) -> None:
        """Test case when there are no float data lines"""
        content = ["header", "some text", "more text"]
        result = get_data_index(content)
        assert result is None  # No line contains float data

    def test_empty_list(self) -> None:
        """Test with an empty list"""
        content = []
        result = get_data_index(content)
        assert result is None  # No data in the list

    def test_mixed_data(self) -> None:
        """Test with mixed data (some numeric and some non-numeric)"""
        content = ["header", "text", "1 2 3", "text again", "4 5 6"]
        result = get_data_index(content)
        assert result == 2  # the first line with numeric data is at index 2

    def test_non_matching_delimiter(self) -> None:
        """Test with a delimiter that doesn't match any line"""
        content = ["header", "text", "1 2 3", "4 5 6"]
        result = get_data_index(content, delimiter=",")
        assert result is None  # no lines with comma as delimiter


class TestLoadData:

    def test_x_y1_y2_y3_format(self) -> None:
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

    def test_x1_y1_x2_y2_format(self) -> None:
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


class TestComparisonFunctions:

    def test_identical_simple_values(self) -> None:
        """Test identical simple values."""
        assert are_identical(5, 5)
        assert are_identical("test", "test")
        assert are_identical(None, None)
        assert not are_identical(5, 6)
        assert not are_identical("test", "different")

    def test_identical_lists(self) -> None:
        """Test identical lists."""
        assert are_identical([1, 2, 3], [1, 2, 3])
        assert are_identical([], [])
        assert are_identical([1, [2, 3]], [1, [2, 3]])
        assert not are_identical([1, 2, 3], [1, 2, 4])
        assert not are_identical([1, 2, 3], [1, 2])
        assert not are_identical([1, 2], [1, 2, 3])

    def test_identical_tuples(self) -> None:
        """Test identical tuples."""
        assert are_identical((1, 2, 3), (1, 2, 3))
        assert are_identical((), ())
        assert are_identical((1, (2, 3)), (1, (2, 3)))
        assert not are_identical((1, 2, 3), (1, 2, 4))

    def test_identical_dicts(self) -> None:
        """Test identical dictionaries."""
        assert are_identical({"a": 1, "b": 2}, {"a": 1, "b": 2})
        assert are_identical({}, {})
        assert are_identical({"a": 1, "b": {"c": 3}}, {"a": 1, "b": {"c": 3}})
        assert not are_identical({"a": 1, "b": 2}, {"a": 1, "b": 3})
        assert not are_identical({"a": 1, "b": 2}, {"a": 1, "c": 2})
        assert not are_identical({"a": 1}, {"a": 1, "b": 2})

    def test_identical_nested_structures(self) -> None:
        """Test identical nested structures."""
        nested1 = {"a": [1, 2, {"b": (3, 4)}]}
        nested2 = {"a": [1, 2, {"b": (3, 4)}]}
        different = {"a": [1, 2, {"b": (3, 5)}]}

        assert are_identical(nested1, nested2)
        assert not are_identical(nested1, different)

    def test_identical_numpy_arrays(self) -> None:
        """Test identical numpy arrays."""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([1, 2, 4])

        assert are_identical(arr1, arr2)
        assert not are_identical(arr1, arr3)

    def test_identical_with_rtol(self) -> None:
        """Test identical with relative tolerance for floating point values."""
        assert are_identical(1.0, 1.001, rtol=1e-2)
        assert not are_identical(1.0, 1.001, rtol=1e-4)

        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.001, 2.002, 3.003])

        assert are_identical(arr1, arr2, rtol=1e-2)
        assert not are_identical(arr1, arr2, rtol=1e-4)

    def test_identical_mixed_types(self) -> None:
        """Test identical with mixed types - should use strict equality."""
        assert are_identical(1, 1.0)  # Different types
        assert are_identical(True, 1)  # Different types

    def test_close_simple_values(self) -> None:
        """Test are_close with simple values."""
        assert are_close(1.0, 1.0009)  # Default rtol=1e-3
        assert are_close(1.0, 1.002, rtol=1e-2)
        assert not are_close(1.0, 1.01)  # Default rtol too small

    def test_close_numpy_arrays(self) -> None:
        """Test are_close with numpy arrays."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0009, 2.001, 3.0008])
        arr3 = np.array([1.01, 2.01, 3.01])

        assert are_close(arr1, arr2)  # Default rtol=1e-3 is enough
        assert not are_close(arr1, arr3)  # Default rtol too small
        assert are_close(arr1, arr3, rtol=1e-1)  # Larger rtol works

    def test_close_nested_structures(self) -> None:
        """Test are_close with nested structures."""
        nested1 = {"a": [1.0, 2.0, {"b": np.array([3.0, 4.0])}]}
        nested2 = {"a": [1.0009, 2.001, {"b": np.array([3.0008, 4.0009])}]}
        nested3 = {"a": [1.01, 2.01, {"b": np.array([3.01, 4.01])}]}

        assert are_close(nested1, nested2)  # Default rtol=1e-3 is enough
        assert not are_close(nested1, nested3)  # Default rtol too small
        assert are_close(nested1, nested3, rtol=1e-1)  # Larger rtol works

    def test_close_different_structures(self) -> None:
        """Test are_close with different structures - should return False."""
        assert not are_close([1.0, 2.0], [1.0, 2.0, 3.0])
        assert not are_close({"a": 1.0}, {"a": 1.0, "b": 2.0})
        assert not are_close({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Empty structures are identical
        assert are_close([], [])
        assert are_close({}, {})

        # NaN values
        nan_array1 = np.array([1.0, np.nan, 3.0])
        nan_array2 = np.array([1.0, np.nan, 3.0])
        assert not are_close(nan_array1, nan_array2)  # np.allclose returns False for NaN

        # Infinity values
        inf_array1 = np.array([1.0, np.inf, 3.0])
        inf_array2 = np.array([1.0, np.inf, 3.0])
        assert are_close(inf_array1, inf_array2)

        # Mixed infinity and regular values
        mixed_inf1 = np.array([1.0, np.inf, 3.0])
        mixed_inf2 = np.array([1.0001, np.inf, 3.0001])
        assert are_close(mixed_inf1, mixed_inf2)


class TestRenderImage:

    def test_valid_image_file(self) -> None:

        image_data = b"\x89PNG\r\n\x1a\n" + b"fakeimagecontent"
        expected_mime = "image/png"

        # Create a temporary image file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        html_output = render_image(tmp_path, width=200)

        encoded = base64.b64encode(image_data).decode()
        expected_html = f'<center><img src="data:{expected_mime};base64,{encoded}" width="200px"/></center>'
        assert html_output == expected_html

    def test_unknown_mime_type_raises(self) -> None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".unknown") as tmp:
            tmp.write(b"Some binary content")
            tmp_path = tmp.name

        with pytest.raises(ValueError, match=r"Could not determine MIME type"):
            render_image(tmp_path)

    def test_default_width(self) -> None:

        image_data = b"GIF87a" + b"fakegifdata"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name

        html_output = render_image(tmp_path)
        assert 'width="100px"' in html_output
        assert "data:image/gif;base64," in html_output


class TestReadTxtFile:
    """Test class for the read_txt_file function."""

    # File path for temporary test file
    TEMP_FILE = "_temp.txt"

    def teardown_method(self) -> None:
        """Teardown method that runs after each test."""

        # Clean up test file after each test
        if os.path.exists(self.TEMP_FILE):
            os.remove(self.TEMP_FILE)

        # Clear the cache
        st.cache_resource.clear()

    def test_read_existing_file(self) -> None:
        """Test reading from an existing file with valid content."""

        # Create file with some content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Hello, World!")

        # Read the content using our function
        content = read_txt_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Hello, World!"

    def test_read_multiline_file(self) -> None:
        """Test reading from a file with multiple lines."""

        # Create a file with multiline content
        with open(self.TEMP_FILE, "w") as f:
            f.write("Line 1\nLine 2\nLine 3")

        # Read the content using our function
        content = read_txt_file(self.TEMP_FILE)

        # Assert the content matches what we wrote
        assert content == "Line 1\nLine 2\nLine 3"
        # Additional check for line count
        assert len(content.splitlines()) == 3

    def test_nonexistent_file(self) -> None:
        """Test that trying to read a nonexistent file raises an error."""

        # Check that the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            read_txt_file(self.TEMP_FILE)


class TestGenerateHtmlTable:
    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Create a sample dataframe for testing."""

        data = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8], "C": [9, 10, 11, 12]}
        df = pd.DataFrame(data, index=["row1", "row2", "row3", "row4"])
        return df

    @pytest.fixture
    def dataframe_with_identical_rows(self) -> pd.DataFrame:
        """Create a dataframe with some identical rows."""

        data = {"A": [1, 2, 2, 3], "B": [5, 2, 2, 7], "C": [9, 2, 2, 11]}
        df = pd.DataFrame(data, index=["row1", "row2", "row3", "row4"])
        return df

    @pytest.fixture
    def dataframe_with_column_name(self) -> pd.DataFrame:
        """Create a dataframe with a column name."""

        data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
        df = pd.DataFrame(data, index=["row1", "row2", "row3"])
        df.columns.name = "Categories"
        return df

    def test_basic_table_generation(self, sample_dataframe) -> None:
        """Test basic HTML table generation."""

        html = generate_html_table(sample_dataframe)

        # Verify structure using BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")

        # Check table exists
        table = soup.find("table")
        assert table is not None

        # Check number of rows (header + data rows)
        rows = table.find_all("tr")
        assert len(rows) == 5  # 1 header + 4 data rows

        # Check header row
        header_cells = rows[0].find_all(["th"])
        assert len(header_cells) == 4  # corner cell + 3 columns

        # Check data cells
        data_rows = rows[1:]
        for i, row in enumerate(data_rows):
            cells = row.find_all("td")
            assert cells[0].text == f"row{i + 1}"  # Check row name

    def test_merged_cells_for_identical_values(self, dataframe_with_identical_rows) -> None:
        """Test that cells are merged when all values in a row are identical."""
        html = generate_html_table(dataframe_with_identical_rows)

        soup = BeautifulSoup(html, "html.parser")
        rows = soup.find_all("tr")

        # Check row2 and row3 have merged cells
        row2 = rows[2]  # index 2 corresponds to row2
        row3 = rows[3]  # index 3 corresponds to row3

        # Check for colspan in row2 and row3
        assert row2.find_all("td")[1].has_attr("colspan")
        assert row2.find_all("td")[1]["colspan"] == "3"
        assert row3.find_all("td")[1].has_attr("colspan")
        assert row3.find_all("td")[1]["colspan"] == "3"

        # Check normal rows don't have merged cells
        row1 = rows[1]  # index 1 corresponds to row1
        row4 = rows[4]  # index 4 corresponds to row4
        assert len(row1.find_all("td")) == 4  # 1 row name + 3 data cells
        assert len(row4.find_all("td")) == 4  # 1 row name + 3 data cells

    def test_column_name_in_corner(self, dataframe_with_column_name) -> None:
        """Test that the column name appears in the corner cell."""
        html = generate_html_table(dataframe_with_column_name)

        soup = BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == "Categories"

    def test_empty_corner_cell_with_no_column_name(self, sample_dataframe) -> None:
        """Test that the corner cell is empty when no column name is provided."""
        html = generate_html_table(sample_dataframe)

        soup = BeautifulSoup(html, "html.parser")
        corner_cell = soup.find("tr").find("th")

        assert corner_cell.text == ""

    def test_div_wrapper(self, sample_dataframe) -> None:
        """Test that the table is wrapped in a div with correct styling."""
        html = generate_html_table(sample_dataframe)

        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div")

        assert div is not None
        assert div.has_attr("style")
        assert "margin: auto" in div["style"]
        assert "display: table" in div["style"]
