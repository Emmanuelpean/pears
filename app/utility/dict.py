"""utils module"""

import re


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


def list_to_dict(dicts: list[dict]) -> dict | None:
    """Convert a list of dictionaries to a dictionary of list. All dictionaries must share the same keys"""

    new_dict = dict()
    for key in dicts[0]:
        new_dict[key] = [d[key] for d in dicts]
    return new_dict
