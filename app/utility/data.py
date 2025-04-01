"""utils module"""

import base64
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st


def matrix_to_string(
    arrays: np.ndarray | list[np.ndarray],
    header: None | list[str] | np.ndarray = None,
) -> str:
    """Convert a matrix to a string
    :param arrays: list of ndarrays
    :param header: header"""

    max_rows = np.max([len(array) for array in arrays])
    rows = []
    delimiter = ","

    for i in range(max_rows):
        row_values = []
        for array in arrays:
            if i < len(array):
                row_values.append(f"{array[i]:.5E}")
            else:
                row_values.append("")
        rows.append(delimiter.join(row_values))

    string = "\n".join(rows)

    if header is not None:
        string = delimiter.join(header) + "\n" + string

    return string


@st.cache_resource
def render_image(
    svg_file: str,
    width: int = 100,
    itype: str = "svg",
) -> str:
    """Render a svg file.
    :param str svg_file: file path
    :param int width: width in percent
    :param str itype: image type"""

    with open(svg_file, "rb") as ofile:
        svg = base64.b64encode(ofile.read()).decode()
        return (
            f'<center><img src="data:image/{itype}+xml;base64,{svg}" id="responsive-image" width="{width}%"/></center>'
        )


def generate_download_link(
    data: tuple,
    header: None | list[str] | np.ndarray = None,
    text: str = "",
    name: str | None = None,
) -> str:
    """Generate a download link from a matrix and a header
    :param data: tuple containing x-axis and y-axis data
    :param header: list of strings corresponding to the header of each column
    :param text: text to be displayed instead of the link
    :param name: name of the file"""

    if name is None:
        name = text
    data = np.concatenate([[data[0][0]], data[1]])
    string = matrix_to_string(data, header)
    b64 = base64.b64encode(string.encode()).decode()
    return rf'<a href="data:text/csv;base64,{b64}" download="{name}.csv">{text}</a>'


def process_data(
    xs_data: list[np.ndarray],
    ys_data: list[np.ndarray],
    normalise: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Process the data
    :param xs_data: x data
    :param ys_data: y data
    :param normalise: if True, normalise the y-axis data"""

    for i in range(len(xs_data)):
        index = ys_data[i].argmax()
        xs_data[i] = xs_data[i][index:]  # reduce range x
        xs_data[i] -= xs_data[i][0]  # shift x
        ys_data[i] = ys_data[i][index:]  # reduce range y
        if normalise:
            ys_data[i] /= ys_data[i][0]  # normalise y
    return xs_data, ys_data


def are_identical(
    obj1: any,
    obj2: any,
    rtol: float | None = None,
) -> bool:
    """Check if two objects are identical.
    :param obj1: list or dictionary to compare.
    :param obj2: list or dictionary to compare.
    :param rtol: Relative tolerance for floating-point comparisons using np.allclose."""

    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if obj1.keys() != obj2.keys():
            return False
        else:
            return all(are_identical(obj1[k], obj2[k], rtol) for k in obj1)

    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        else:
            return all(are_identical(i1, i2, rtol) for i1, i2 in zip(obj1, obj2))

    else:
        if rtol is not None:
            return np.allclose(obj1, obj2, rtol=rtol)
        else:
            return np.array_equal(obj1, obj2)


def are_close(*args, rtol=1e-3) -> bool:
    """Check if two objects are similar"""

    return are_identical(*args, rtol=rtol)


def get_data_index(
    content: list[str],
    delimiter: None | str = None,
) -> None | int:
    """Retrieve the index of the line where the data starts
    :param list of str content: list of strings
    :param str or None delimiter: delimiter of the float data

    Example
    -------
    >>> get_data_index(['first line', 'second line', '1 2 3'])
    2"""

    for index, line in enumerate(content):

        if line != "":
            try:
                [float(f) for f in line.split(delimiter)]
                return index
            except ValueError:
                continue


def load_data(
    content: bytes,
    delimiter: str,
    data_format: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Process the data contained in a file
    :param bytes content: content string
    :param str delimiter: data delimiter
    :param str data_format: data column format"""

    # Find data index and load the data
    content1 = content.decode("ascii").splitlines()
    content1 = [line.strip(delimiter) for line in content1]  # remove any extra delimiter on each line
    index = get_data_index(content1, delimiter)
    data = np.transpose(np.genfromtxt(StringIO(content.decode("ascii")), delimiter=delimiter, skip_header=index))

    # Sort the data
    if data_format == "X/Y1/Y2/Y3...":
        xs_data, ys_data = np.array([data[0]] * (len(data) - 1)), data[1:]
    else:
        xs_data, ys_data = data[::2], data[1::2]

    # Check the data
    xs_data = [x_data[np.invert(np.isnan(x_data))] for x_data in xs_data]
    ys_data = [y_data[np.invert(np.isnan(y_data))] for y_data in ys_data]

    return xs_data, ys_data


@st.cache_resource
def read_txt_file(path: str) -> str:
    """Read the content of a text file and store it as a resource.
    :param path: file path"""

    with open(path) as ofile:
        return ofile.read()


def generate_html_table(df: pd.DataFrame) -> str:
    """Generate an HTML table from a pandas DataFrame with merged cells for rows
    where all values are identical. Includes row names (index), column names,
    and displays the columns name in the upper-left corner cell.
    :param df: pandas DataFrame to convert to HTML"""

    html = ['<table border="1" style="border-collapse: collapse; text-align: center;">']

    # Add header row with columns name in the corner cell
    corner_cell_content = df.columns.name if df.columns.name else ""
    header = f'<tr><th style="padding: 8px; text-align: center;">{corner_cell_content}</th>'
    for col in df.columns:
        header += f'<th style="padding: 8px; text-align: center;">{col}</th>'
    header += "</tr>"
    html.append(header)

    # Process each row
    for idx, row in df.iterrows():
        values = row.tolist()

        # Check if all values in the row are the same
        if len(set(values)) == 1:
            # All values are identical - merge cells, but keep row name
            html.append(
                f'<tr><td style="padding: 8px; font-weight: bold; text-align: center;">{idx}</td>'
                + f'<td colspan="{len(df.columns)}" style="padding: 8px; text-align: center;">{values[0]}</td></tr>'
            )
        else:
            # Normal row handling with row name
            row_html = f'<tr><td style="padding: 8px; font-weight: bold; text-align: center;">{idx}</td>'
            for val in values:
                row_html += f'<td style="padding: 8px; text-align: center;">{val}</td>'
            row_html += "</tr>"
            html.append(row_html)

    html.append("</table>")
    return '<div style="margin: auto; display: table;">' + "\n".join(html) + "</div>"
