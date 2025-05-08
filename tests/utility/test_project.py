"""Test module for the functions in the `utils/project.py` module.

This module contains unit tests for the functions implemented in the `utils/project.py` module. The purpose of these tests is to
ensure the correct functionality of each function in different scenarios and to validate that the expected outputs are
returned.

Tests should cover various edge cases, valid inputs, and any other conditions that are necessary to confirm the
robustness of the functions."""

from pathlib import Path
from unittest.mock import patch, Mock, mock_open

import pytest

from utility.project import get_last_commit_date_from_github, get_pyproject_info


class TestGetLastCommitDateFromGitHub:

    @patch("utility.project.requests.get")
    def test_successful_response(self, mock_get) -> None:

        # Arrange
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"commit": {"committer": {"date": "2024-04-27T12:34:56Z"}}}]
        mock_get.return_value = mock_response

        # Act
        result = get_last_commit_date_from_github("https://github.com/owner/repo")

        # Assert
        assert result == "27 April 2024"
        mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo/commits?sha=main")

    @patch("utility.project.requests.get")
    def test_custom_branch(self, mock_get) -> None:

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"commit": {"committer": {"date": "2024-01-15T08:00:00Z"}}}]
        mock_get.return_value = mock_response

        result = get_last_commit_date_from_github("https://github.com/owner/repo", branch="dev")
        assert result == "15 January 2024"
        mock_get.assert_called_once_with("https://api.github.com/repos/owner/repo/commits?sha=dev")

    @patch("utility.project.requests.get")
    def test_http_error(self, mock_get) -> None:

        # Arrange
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response

        # Act
        result = get_last_commit_date_from_github("https://github.com/owner/repo")

        # Assert
        assert result.startswith("Error fetching data:")


class TestGetPyprojectInfo:

    @patch("utility.project.tomllib.load")
    @patch("utility.project.Path.open", new_callable=mock_open)
    @patch("utility.project.Path.resolve")
    def test_single_key(self, mock_resolve, mock_open_file, mock_toml_load) -> None:
        # Arrange
        mock_resolve.return_value = Path("/fake/path/to/pyproject.toml")
        mock_toml_load.return_value = {"tool": {"poetry": {"name": "example-package"}}}

        # Act
        result = get_pyproject_info("tool")

        # Assert
        assert result == {"poetry": {"name": "example-package"}}
        mock_open_file.assert_called_once_with("rb")

    @patch("utility.project.tomllib.load")
    @patch("utility.project.Path.open", new_callable=mock_open)
    @patch("utility.project.Path.resolve")
    def test_multiple_keys(self, mock_resolve, mock_open_file, mock_toml_load) -> None:
        # Arrange
        mock_resolve.return_value = Path("/fake/path/to/pyproject.toml")
        mock_toml_load.return_value = {"tool": {"poetry": {"version": "1.2.3"}}}

        # Act
        result = get_pyproject_info("tool", "poetry", "version")

        # Assert
        assert result == "1.2.3"
        mock_open_file.assert_called_once_with("rb")

    @patch("utility.project.tomllib.load")
    @patch("utility.project.Path.open", new_callable=mock_open)
    @patch("utility.project.Path.resolve")
    def test_missing_key_raises_keyerror(self, mock_resolve, _mock_open_file, mock_toml_load) -> None:
        # Arrange
        mock_resolve.return_value = Path("/fake/path/to/pyproject.toml")
        mock_toml_load.return_value = {"tool": {"poetry": {}}}

        # Act & Assert
        with pytest.raises(KeyError):
            get_pyproject_info("tool", "poetry", "nonexistent_key")
