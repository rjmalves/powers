"""Git state schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .base import SerializableMixin


@dataclass
class GitInfo(SerializableMixin):
    """Git repository state."""

    commit_sha: str
    commit_short: str
    branch: str
    is_dirty: bool
    commit_date: str
    commit_message: str
    tags: List[str] = field(default_factory=list)

