# T-005: Implement Git Info Extraction

> **Epic**: [Epic 1: Core Framework](../00-epic-overview.md)
> **Sprint**: [Sprint 1: Foundation](./00-sprint-overview.md)
> **Dependencies**: T-001, T-003
> **Blocks**: T-008, T-010

---

## Context

### Background

Profiling runs must be indexed by git commit to enable version-to-version comparison and performance bisection. This ticket implements git repository state extraction.

### Current State

No git extraction exists. The `GitInfo` schema was defined in T-003.

---

## Specification

### Detection Requirements

| Field | Git Command | Fallback |
|-------|-------------|----------|
| commit_sha | `git rev-parse HEAD` | "unknown" |
| commit_short | `git rev-parse --short HEAD` | "unknown" |
| branch | `git rev-parse --abbrev-ref HEAD` | "detached" |
| is_dirty | `git status --porcelain` | False |
| commit_date | `git show -s --format=%cI HEAD` | "" |
| commit_message | `git show -s --format=%s HEAD` | "" |
| tags | `git tag --points-at HEAD` | [] |

### Implementation

```python
def detect_git_info(repo_path: Optional[Path] = None) -> GitInfo:
    """Detect git repository state."""
    cwd = str(repo_path) if repo_path else None
    
    def git(*args) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""
    
    commit_sha = git("rev-parse", "HEAD") or "unknown"
    commit_short = git("rev-parse", "--short", "HEAD") or "unknown"
    
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":
        branch = "detached"
    elif not branch:
        branch = "unknown"
    
    is_dirty = bool(git("status", "--porcelain"))
    commit_date = git("show", "-s", "--format=%cI", "HEAD")
    commit_message = git("show", "-s", "--format=%s", "HEAD")
    
    tags_output = git("tag", "--points-at", "HEAD")
    tags = tags_output.split("\n") if tags_output else []
    
    return GitInfo(
        commit_sha=commit_sha,
        commit_short=commit_short,
        branch=branch,
        is_dirty=is_dirty,
        commit_date=commit_date,
        commit_message=commit_message,
        tags=tags,
    )
```

### Edge Cases

- **Not a git repo**: All fields fallback gracefully
- **Detached HEAD**: branch = "detached"
- **Uncommitted changes**: is_dirty = True
- **No tags**: tags = []

---

## Acceptance Criteria

- [ ] All git fields correctly extracted
- [ ] Works in normal checkout
- [ ] Works in detached HEAD state
- [ ] is_dirty correctly detects uncommitted changes
- [ ] Tags correctly detected
- [ ] Graceful fallback when not in git repo
- [ ] No exceptions thrown

---

## Implementation Guide

### Suggested Approach

1. Create `utils/git_info.py`
2. Implement helper function for git commands
3. Implement `detect_git_info()` function
4. Handle all error cases
5. Add unit tests

### Key Files to Create

- `profiling/powers_profile/utils/git_info.py`

### Pitfalls to Avoid

- ⚠️ `git rev-parse --abbrev-ref HEAD` returns "HEAD" when detached
- ⚠️ `git status --porcelain` may include untracked files
- ⚠️ Subprocess timeout is important for hung git commands
- ⚠️ Don't assume git is installed

---

## Testing Requirements

### Unit Tests

```python
# tests/test_git_info.py
from unittest.mock import patch, MagicMock
import subprocess

def test_detect_git_info_returns_git_info():
    info = detect_git_info()
    assert isinstance(info, GitInfo)

def test_commit_sha_format():
    info = detect_git_info()
    if info.commit_sha != "unknown":
        assert len(info.commit_sha) == 40
        assert all(c in "0123456789abcdef" for c in info.commit_sha)

def test_short_sha_format():
    info = detect_git_info()
    if info.commit_short != "unknown":
        assert len(info.commit_short) == 7

def test_fallback_when_not_git_repo():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        info = detect_git_info()
        assert info.commit_sha == "unknown"
        assert info.branch == "unknown"
```

### Integration Tests

- [ ] Test in actual git repository
- [ ] Test with dirty working directory
- [ ] Test with tags

---

## Documentation Requirements

- [ ] Docstrings on all functions
- [ ] Document fallback behavior

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Simple git command wrapping

---

## Definition of Done

- [ ] All git fields extracted
- [ ] Fallbacks work
- [ ] Tests pass
- [ ] Code reviewed
