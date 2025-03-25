"""utils module"""

import base64
import inspect
import math
import re
from collections.abc import Callable
from io import StringIO

import numpy as np
import streamlit as st


def merge_dicts(*dictionaries: dict) -> dict:
    """Merge multiple dictionaries, keeping the first occurrence of each key.
    If a key appears in more than one dictionary, the value from the first dictionary containing the key will be used.

    Example
    -------
    >>> merge_dicts({'label': 'string'}, {'c': 'r', 'label': 'something'})
    {'label': 'string', 'c': 'r'}"""

    merged_dictionary = {}
    for dictionary in dictionaries:
        for key in dictionary.keys():
            if key not in merged_dictionary:
                merged_dictionary[key] = dictionary[key]

    return merged_dictionary


def normalize_to_unit(number: float) -> tuple[float, int]:
    """Normalize a number to the range [0.1, 1], adjusting the power (exponent) accordingly.
    The function returns the normalized value and the exponent.

    Examples
    --------
    >>> normalize_to_unit(1.433364345e9)
    (0.1433364345, 10)
    >>> normalize_to_unit(14e-6)
    (0.13999999999999999, -4)
    >>> normalize_to_unit(-14e-6)
    (-0.13999999999999999, -4)"""

    exponent = 0  # exponent value
    value = number  # normalized value

    if value == 0.0:
        value = 0.0
        exponent = 0

    elif abs(value) < 1:
        while abs(value) < 0.1:
            value *= 10.0
            exponent -= 1

    elif abs(value) > 1:
        while abs(value) > 1:
            value /= 10.0
            exponent += 1

    return value, exponent


def filter_dicts(
    dicts: list[dict[str, float]],
    inequations: list[str],
    fixed: None | dict[str, float] = None,
) -> list[dict[str, float]]:
    # noinspection PyUnresolvedReferences
    """Filter a list of dictionaries given a list of inequations of the form "arg1 < arg2" or "arg1 > arg2"
    :param dicts: list of dictionaries sharing the same keys
    :param inequations: list of string (inequations)
    :param fixed: dictionary containing fixed values

    Examples
    --------
    >>> dicts1 = [{'a': a, 'b': b} for a, b in zip([4, 2, 8, 4], [2, 3, 4, 5])]
    >>> filter_dicts(dicts1, ['b < a'])
    [{'a': 4, 'b': 2}, {'a': 8, 'b': 4}]
    >>> dicts2 = [{'a': a} for a in [4, 2, 8, 4]]
    >>> filter_dicts(dicts2, ['a < b'], {'b': 4.1})
    [{'a': 4}, {'a': 2}, {'a': 4}]
    >>> dicts3 = [{'a': a} for a in [4, 2, 8, 4]]
    >>> filter_dicts(dicts3, ['b < a'], {'b': 4.1})
    [{'a': 8}]"""

    keys = dicts[0].keys()
    if fixed is None:
        fixed = {}

    def get_inequality_condition(
        value1: float | int,
        value2: float | int,
        sign: str,
    ) -> bool:
        """Get the inequality boolean depending on the value of a and b and sign
        :param value1: value 1
        :param value2: value 2
        :param sign: string containing '>', '<', '>=', '=>', '<=' or '=<'"""

        operations = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "=>": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "=<": lambda a, b: a <= b,
        }

        return operations[sign](value1, value2)

    for inequation in inequations:
        match = re.search(r"(<=|>=|=<|=>|<|>)", inequation)
        if not match:
            raise ValueError(f"Invalid inequality expression: {inequation}")
        ineq_sign = match.group(0)
        arg1, arg2 = [s.strip() for s in inequation.split(ineq_sign)]

        if arg1 in keys and arg2 in keys:
            dicts = [p for p in dicts if get_inequality_condition(p[arg1], p[arg2], ineq_sign)]
        elif arg1 in fixed:
            dicts = [p for p in dicts if get_inequality_condition(fixed[arg1], p[arg2], ineq_sign)]
        elif arg2 in fixed:
            dicts = [p for p in dicts if get_inequality_condition(p[arg1], fixed[arg2], ineq_sign)]

    return dicts


def keep_function_kwargs(
    function: Callable,
    kwargs: dict,
) -> dict:
    """Keep the keys of a dictionary that can be passed into a function.
    :param function: function
    :param kwargs: keyword arguments

    Example
    -------
    >>> function1 = lambda a, b: 1
    >>> keep_function_kwargs(function1, {'a': 1, 'c': 1})
    {'a': 1}"""

    return {key: kwargs[key] for key in kwargs if key in inspect.getfullargspec(function).args}


def get_power_html(
    value: float,
    n: None | int = 3,
) -> str:
    """Converts a value to html-formatted text with base 10
    :param value: value
    :param n: number of decimals. None to keep only the exponent bit. -1 to display all significant digits.

    Examples
    --------
    >>> get_power_html(1.3e13, 3)
    '1.300 &#10005; 10<sup>13</sup>'
    >>> get_power_html(1.34e13, -1)
    '1.34 &#10005; 10<sup>13</sup>'
    >>> get_power_html(1.34e13, None)
    '10<sup>13</sup>'
    >>> get_power_html(999.9, 2)
    '1.00 &#10005; 10<sup>3</sup>'"""

    if value == 0:
        return "0"

    base10 = math.floor(np.log10(value))
    mantissa = value / 10**base10

    # Adjust for boundary values like 999.9 to display as 1.00 x 10^3 instead of 10.0 x 10^2
    if round(mantissa, n) >= 10:
        mantissa /= 10
        base10 += 1

    if n is None:
        return f"10<sup>{base10}</sup>"
    elif n == -1:
        return f"{mantissa:g} &#10005; 10<sup>{base10}</sup>"
    else:
        return f"{mantissa:.{n}f} &#10005; 10<sup>{base10}</sup>"


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


def to_scientific(value: float | int | list[float | int] | None) -> str:
    """Convert a number to scientific notation without trailing zeros

    Examples
    --------
    >>> to_scientific(1.4e-4)
    '1.4E-04'
    >>> to_scientific(None)
    ''
    >>> to_scientific([1e-4, 1e-5])
    '1.E-04, 1.E-05'"""

    if value is None:
        return ""
    elif isinstance(value, (float, int, np.integer)):
        string = "%E" % value
        mantissa, exponent = string.split("E")
        return mantissa[0] + mantissa[1:].strip("0") + "E" + exponent
    else:
        return ", ".join([to_scientific(f) for f in value])


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
            f'<center><img src="data:image/{itype}+xml;base64,{svg}" id="responsive-image" width="{width}%%"/></center>'
        )


@st.cache_data
def generate_download_link(
    data: np.ndarray,
    header: None | list = None,
    text: str = "",
    name: str = "data",
) -> str:
    """Generate a download link from a matrix and a header
    :param data: matrix
    :param header: list of strings corresponding to the header of each column
    :param text: text to be displayed instead of the link
    :param name: name of the file"""

    string = matrix_to_string(data, header)
    b64 = base64.b64encode(string.encode()).decode()
    return rf'<a href="data:text/csv;base64,{b64}" download="{name}.csv">{text}</a>'


def list_to_dict(dicts: list[dict]) -> dict | None:
    """Convert a list of dictionaries to a dictionary of list. All dictionaries must share the same keys"""

    new_dict = dict()
    for key in dicts[0]:
        new_dict[key] = [d[key] for d in dicts]
    return new_dict


def get_power_labels(N0s: list[float]) -> list[str]:
    """Get the power labels
    :param N0s: initial carrier concentrations"""

    return ["N<sub>0</sub> = " + get_power_html(N0, -1) + " cm<sup>-3</sup>" for N0 in N0s]


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


def are_lists_identical(
    list1: list,
    list2: list,
) -> bool:
    """Check if 2 lists of arrays are different
    :param list1: list of arrays
    :param list2: list of arrays"""

    if len(list1) != len(list2):
        return False

    for item1, item2 in zip(list1, list2):
        if isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            if not np.array_equal(item1, item2):
                return False
        elif isinstance(item1, list) and isinstance(item2, list):
            if not are_lists_identical(item1, item2):
                return False
        elif item1 != item2:
            return False

    return True


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
