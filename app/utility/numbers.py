import math
import numpy as np


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

    if n is None:
        if base10 == 0:
            return ""
        else:
            return f"&#10005; 10<sup>{base10}</sup>"
    elif n == -1:
        return f"{mantissa:g} &#10005; 10<sup>{base10}</sup>"
    else:
        return f"{mantissa:.{n}f} &#10005; 10<sup>{base10}</sup>"


def to_scientific(value: float | int | list[float | int] | None) -> str:
    """Convert a number to scientific notation without trailing zeros

    Examples
    --------
    >>> to_scientific(1.4e-4)
    '1.4E-04'
    >>> to_scientific(None)
    ''
    >>> to_scientific([1e-4, 1e-5])
    '1E-04, 1E-05'"""

    if value is None:
        return ""
    if value == 0:
        return "0"
    elif isinstance(value, (float, int, np.integer)):
        string = "%E" % value
        mantissa, exponent = string.split("E")
        return (mantissa[0] + mantissa[1:].strip("0")).strip(".") + "E" + exponent
    else:
        return ", ".join([to_scientific(f) for f in value])


def get_power_labels(N0s: list[float]) -> list[str]:
    """Get the power labels
    :param N0s: initial carrier concentrations"""

    return ["N<sub>0</sub> = " + get_power_html(N0, -1) + " cm<sup>-3</sup>" for N0 in N0s]
