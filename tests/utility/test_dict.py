import pytest

from app.utility.dict import *


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
