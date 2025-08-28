"""Module containing functions useful for managing the project"""

import datetime as dt
import tomllib
from pathlib import Path
from typing import Any

import requests


def get_pyproject_info(*keys: str) -> Any:
    """Get information from the pyproject file"""

    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    d = data[keys[0]]
    for key in keys[1:]:
        d = d[key]
    return d


def get_last_commit_date_from_github(
    repo_url: str,
    branch: str = "main",
) -> str:
    """Get the date of the latest commit
    :param repo_url: repository url
    :param branch: specific branch"""

    # Extract the owner and repo name from the URL
    repo_parts = repo_url.rstrip("/").split("/")[-2:]
    owner, repo = repo_parts[0], repo_parts[1]

    # GitHub API endpoint to fetch the latest commit
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}"

    try:
        # Send the request to GitHub API
        response = requests.get(api_url)
        response.raise_for_status()  # Raise error for bad status codes

        # Extract the last commit date from the JSON response
        commit_data = response.json()[0]
        commit_date = commit_data["commit"]["committer"]["date"]
        date = dt.datetime.strptime(commit_date, "%Y-%m-%dT%H:%M:%SZ")
        return date.strftime("%d %B %Y")
    except Exception as e:
        return f"Error fetching data: {e}"
