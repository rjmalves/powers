"""Git info detection tests."""

from pathlib import Path

from powers_profile.schemas import GitInfo
from powers_profile.utils import detect_git_info


def test_detect_git_info_in_repo() -> None:
    info = detect_git_info(Path("."))
    assert isinstance(info, GitInfo)
    assert info.commit_sha
    assert info.commit_short
    assert info.branch
