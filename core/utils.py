""" utils module """

import copy
import numpy as np
import inspect
import math
import streamlit as st
import base64
from io import StringIO


def merge_dicts(*dictionaries):
    """ Merge multiple dictionaries. Last argument can be a boolean determining if the dicts can erase each other's content

    Examples
    --------
    >>> dict1 = {'label': 'string'}
    >>> dict2 = {'c': 'r', 'label': 'something'}

    >>> merge_dicts(dict1, dict2, False)
    {'label': 'string', 'c': 'r'}

    >>> merge_dicts(dict1, dict2, True)
    {'label': 'something', 'c': 'r'}

    >>> merge_dicts(dict1, dict2)
    {'label': 'string', 'c': 'r'}"""

    # Argument for force rewriting the dictionaries
    if isinstance(dictionaries[-1], bool):
        force = dictionaries[-1]
        dictionaries = dictionaries[:-1]
    else:
        force = False

    merged_dictionary = {}
    for dictionary in dictionaries:
        if force:
            merged_dictionary.update(dictionary)
        else:
            for key in dictionary.keys():
                if key not in merged_dictionary:
                    merged_dictionary[key] = dictionary[key]

    return merged_dictionary


def normalise_number(a):
    """ Split the power part of a number and its float value normalised to 1

    Example
    -------
    >>> normalise_number(1.433364345e9)
    (0.1433364345, 10)

    >>> normalise_number(14e-6)
    (0.13999999999999999, -4)

    >>> normalise_number(-14e-6)
    (-0.13999999999999999, -4)"""

    i = 0  # power value
    v = a  # normalised value
    if v == 0.:
        v = 0.
        i = 0.
    elif abs(v) < 1:
        while abs(v) < 0.1:
            v *= 10.
            i -= 1
    elif abs(v) > 1:
        while abs(v) > 1:
            v /= 10.
            i += 1

    return v, i


def filter_dicts(dicts, inequations, fixed=None):
    # noinspection PyUnresolvedReferences
    """ Filter a list of dictionaries given a list of inequations of the form "arg1 < arg2" or "arg1 > arg2"
    :param list or np.ndarray dicts: list of dictionaries sharing the same keys
    :param list inequations: list of string (inequations)
    :param dict or None fixed: dictionary containing fixed values

    Example
    -------
    >>> dicts1 = [{'a': a, 'b': b} for a, b in zip([4, 2, 8, 4], [2, 3, 4, 5])]
    >>> filter_dicts(dicts1, ['b < a'])
    [{'a': 4, 'b': 2}, {'a': 8, 'b': 4}]

    >>> dicts2 = [{'a': a} for a in [4, 2, 8, 4]]
    >>> filter_dicts(dicts2, ['a < b'], {'b': 4.1})
    [{'a': 4}, {'a': 2}, {'a': 4}]

    >>> dicts3 = [{'a': a} for a in [4, 2, 8, 4]]
    >>> filter_dicts(dicts3, ['b < a'], {'b': 4.1})
    [{'a': 8}]
    """

    dictionaries = copy.deepcopy(dicts)
    keys = dictionaries[0].keys()
    if fixed is None:
        fixed = {}

    def test(a, b, sign):
        if sign == '<':
            return a < b
        else:
            return a > b

    for ineq in inequations:
        if '>' in ineq:
            ineq_sign = '>'
        else:
            ineq_sign = '<'

        k1, k2 = [s.strip() for s in ineq.split(ineq_sign)]

        if k1 in keys and k2 in keys:
            dictionaries = [p for p in dictionaries if test(p[k1], p[k2], ineq_sign)]
        elif k1 in fixed:
            dictionaries = [p for p in dictionaries if test(fixed[k1], p[k2], ineq_sign)]
        elif k2 in fixed:
            dictionaries = [p for p in dictionaries if test(p[k1], fixed[k2], ineq_sign)]

    return dictionaries


def keep_function_kwargs(function, kwargs):
    """ Keep the keys of a dictionary

    Example
    -------
    >>> function1 = lambda a, b: 1
    >>> keep_function_kwargs(function1, {'a': 1, 'c': 1})
    {'a': 1}"""

    return {key: kwargs[key] for key in kwargs if key in inspect.getfullargspec(function).args}


def get_power_text(value, n=3):
    """ Converts a value to text with base 10
    :param float value: value
    :param int or None n: number of decimals

    Example
    -------
    >>> get_power_text(1.3e13, 3)
    '1.300 &#10005; 10<sup>13</sup>'

    >>> get_power_text(1.34e13, -1)
    '1.34 &#10005; 10<sup>13</sup>'

    >>> get_power_text(1.34e13, None)
    '10<sup>13</sup>'"""

    if value == 0:
        return '0'

    base10 = math.floor(np.log10(value))

    if n is None:
        return '10<sup>%i</sup>' % base10
    elif n == -1:
        return ('%g' + ' &#10005; 10<sup>%i</sup>') % (value / 10 ** base10, base10)
    else:
        return ('%.' + str(n) + 'f &#10005; 10<sup>%i</sup>') % (np.round(value / 10 ** base10, n), base10)


def matrix_to_string(arrays, header=None):
    """ Convert a matrix array to a string
    :param np.ndarray, list arrays: list of ndarrays
    :param np.ndarray list header: header

    Example
    -------
    >>> arrays1 = [np.array([1.2, 2, 5]), np.array([1.6, 2])]
    >>> header1 = ['A', 'B']
    >>> matrix_to_string(arrays1, header1)
    'A,B\n1.20000E+00,1.60000E+00,\n2.00000E+00,2.00000E+00,\n5.00000E+00,,'  # will thrown an error due to \n
    """

    n = np.max([len(array) for array in arrays])
    rows = []
    delimiter = ','
    for i in range(n):
        row = ''
        for array in arrays:
            if i < len(array):
                row += '%.5E' % array[i] + delimiter
            else:
                row += delimiter
        rows.append(row)
    string = '\n'.join(rows)

    if header is not None:
        string = delimiter.join(header) + '\n' + string

    return string


def to_scientific(a):
    """ Convert a number to scientific notation without trailing zeros

    Examples
    --------
    >>> to_scientific(1.4e-4)
    '1.4E-04'
    >>> to_scientific(None)
    ''
    >>> to_scientific([1e-4, 1e-5])
    '1.E-04, 1.E-05'"""

    if a is None:
        return ''
    elif isinstance(a, (float, int, np.integer)):
        s = '%E' % a
        b, c = s.split('E')
        return b[0] + b[1:].strip('0') + 'E' + c
    else:
        return ', '.join([to_scientific(f) for f in a])


@st.cache
def render_image(svg_file, width=100, itype='svg'):
    """ Render a svg file
    :param str svg_file: file path
    :param int width: width in percent
    :param str itype: image type"""

    with open(svg_file, "rb") as ofile:
        svg = base64.b64encode(ofile.read()).decode()
        return '<center><img src="data:image/%s+xml;base64,%s" id="responsive-image" width="%s%%"/></center>' % (itype, svg, width)


@st.cache
def generate_downloadlink(array, header=None, text=''):
    """ Generate a download link from a matrix and a header
    :param np.ndarray, list array: matrix
    :param None, list None header: list of strings corresponding to the header of each column
    :param str text: text to be displayed instead of the link """
    
    string = matrix_to_string(array, header)
    b64 = base64.b64encode(string.encode()).decode()
    return r'<a href="data:file/csv;base64,%s" width="30">' % b64 + text + '</a>'


def list_to_dict(dicts):
    """ Convert a list of dictionaries to a dictionary of list """

    new_dict = dict()
    if isinstance(dicts, list):
        for key in dicts[0]:
            new_dict[key] = [d[key] for d in dicts]
        return new_dict


def get_power_labels(N0s):
    """ Get the power labels """
    return ['N<sub>0</sub> = ' + get_power_text(N0, -1) + ' cm<sup>-3</sup>' for N0 in N0s]


def process_data(xs_data, ys_data, quantity):
    """ Process the data
    :param list xs_data: x data
    :param list ys_data: y data
    :param str quantity: 'TRPL' or 'TRMC' """

    for i in range(len(xs_data)):
        index = ys_data[i].argmax()
        xs_data[i] = xs_data[i][index:]  # reduce range x
        xs_data[i] -= xs_data[i][0]  # shift x
        ys_data[i] = ys_data[i][index:]  # reduce range y
        if quantity == 'TRPL':
            ys_data[i] /= ys_data[i][0]  # normalise y
    return xs_data, ys_data


def get_data_index(content, delimiter=None):
    """ Retrieve the index of the line where the data starts
    :param list of str content: list of strings
    :param str or None delimiter: delimiter of the float data

    Example
    -------
    >>> get_data_index(['first line', 'second line', '1 2 3'])
    2"""

    for index, line in enumerate(content):

        if line != '':
            try:
                [float(f) for f in line.split(delimiter)]
                return index
            except ValueError:
                continue


def load_data(content, delimiter, data_format, ):
    """ Process the data contained in a file
    :param bytes content: content string
    :param str delimiter: data delimiter
    :param str data_format: data column format

    Examples
    --------
    >>> import core.resources

    >>> file1 = open(core.resources.BT_TRPL_np_file, 'rb')
    >>> data1 = load_data(file1.read(), ' ', 'X/Y1/Y2/Y3...')
    >>> file1.close()

    >>> file2 = open(core.resources.incomplete_file, 'rb')
    >>> data2 = load_data(file2.read(), ',', 'X1/Y1')
    >>> file2.close() """

    # Find data index and load the data
    content1 = content.decode('ascii').splitlines()
    content1 = [line.strip(delimiter) for line in content1]  # remove any extra delimiter on each line
    index = get_data_index(content1, delimiter)
    data = np.transpose(np.genfromtxt(StringIO(content.decode('ascii')), delimiter=delimiter, skip_header=index))

    # Sort the data
    if data_format == 'X/Y1/Y2/Y3...':
        xs_data, ys_data = np.array([data[0]] * (len(data) - 1)), data[1:]
    else:
        xs_data, ys_data = data[::2], data[1::2]

    # Check the data
    xs_data = [x_data[np.invert(np.isnan(x_data))] for x_data in xs_data]
    ys_data = [y_data[np.invert(np.isnan(y_data))] for y_data in ys_data]

    return xs_data, ys_data


if __name__ == '__main__':
    import doctest
    doctest.testmod()
