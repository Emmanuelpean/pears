import math
import numpy as np


def get_power_html(
    value: float | list[float],
    n: None | int = 3,
) -> str | list[str]:
    """Converts a value to html-formatted text with base 10
    :param value: positive value
    :param n: number of decimals. None to keep only the exponent bit. -1 to display all significant digits."""

    if isinstance(value, list):
        return [get_power_html(val, n) for val in value]

    if value == 0:
        return "0"

    base10 = math.floor(np.log10(abs(value)))
    mantissa = value / 10**base10

    if n is None:
        mantissa_str = ""

    # Dynamic number of digits
    elif n == -1:
        mantissa_str = f"{mantissa:g}"

    # Fixed number of digits
    else:
        mantissa_str = f"{mantissa:.{n}f}"

    if base10 == 0:
        return mantissa_str
    else:
        return (mantissa_str + f" &#10005; 10<sup>{base10}</sup>").strip()


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


def get_concentrations_html(N0s: list[float]) -> list[str]:
    """Get the carrier concentration labels in html format
    :param N0s: initial carrier concentrations"""

    return ["N<sub>0</sub> = " + get_power_html(N0, -1) + " cm<sup>-3</sup>" for N0 in N0s]
