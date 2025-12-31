# [T-105] Add RSS Monitoring Utilities

> **Epic**: [Epic 5: Parallel Zero-Allocation Memory Optimization](../00-epic-overview.md)
> **Sprint**: [Sprint 8: Model Rebuild Strategy](./00-sprint-overview.md)
> **Dependencies**: None
> **Blocks**: T-107
> **Priority**: 1 (Critical)
> **Status**: 🔵 Ready

## Files to Read Before Starting

- `src/memory/mod.rs` - Existing memory module structure
- `docs/HIGHS_RSS_MEMORY_INVESTIGATION.md` - Why RSS monitoring is needed

---

## Context

### Background

To validate the model rebuild strategy and provide visibility into memory behavior, we need utilities to query and format the process's RSS (Resident Set Size).

### Requirements

- Linux support required (primary platform)
- macOS support nice-to-have (development)
- Windows support not required
- Zero allocation in hot paths (use for logging only)

---

## Specification

### New Module

Create `src/memory/rss.rs`:

```rust
//! RSS (Resident Set Size) monitoring utilities.
//!
//! Provides platform-specific functions to query process memory usage.

/// Get current process RSS (Resident Set Size) in bytes.
///
/// Uses platform-specific APIs:
/// - Linux: reads `/proc/self/statm`
/// - macOS: uses `mach_task_info` (optional)
/// - Other: returns `None`
///
/// # Returns
///
/// `Some(bytes)` on supported platforms, `None` otherwise.
///
/// # Example
///
/// ```ignore
/// if let Some(rss) = get_rss_bytes() {
///     println!("Current RSS: {} bytes", rss);
/// }
/// ```
pub fn get_rss_bytes() -> Option<usize>;

/// Format RSS bytes for human-readable logging.
///
/// Automatically selects appropriate unit (KB, MB, GB).
///
/// # Example
///
/// ```
/// use powers_rs::memory::format_rss;
/// assert_eq!(format_rss(1_500_000_000), "1.50 GB");
/// assert_eq!(format_rss(150_000_000), "150.00 MB");
/// assert_eq!(format_rss(1_500_000), "1.50 MB");
/// ```
pub fn format_rss(bytes: usize) -> String;
```

### Linux Implementation

```rust
#[cfg(target_os = "linux")]
pub fn get_rss_bytes() -> Option<usize> {
    use std::fs;
    
    // /proc/self/statm contains: size resident shared text lib data dt
    // We want "resident" (second field) which is RSS in pages
    let statm = fs::read_to_string("/proc/self/statm").ok()?;
    let parts: Vec<&str> = statm.split_whitespace().collect();
    
    // Second field is RSS in pages
    let rss_pages: usize = parts.get(1)?.parse().ok()?;
    
    // Get page size from system
    let page_size = page_size();
    
    Some(rss_pages * page_size)
}

#[cfg(target_os = "linux")]
fn page_size() -> usize {
    // SAFETY: sysconf is safe to call with _SC_PAGESIZE
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
```

### macOS Implementation (Optional)

```rust
#[cfg(target_os = "macos")]
pub fn get_rss_bytes() -> Option<usize> {
    use std::mem::MaybeUninit;
    
    // Use mach_task_basic_info to get RSS
    unsafe {
        let mut info = MaybeUninit::<libc::mach_task_basic_info>::uninit();
        let mut count = std::mem::size_of::<libc::mach_task_basic_info>() as u32 
            / std::mem::size_of::<libc::integer_t>() as u32;
        
        let result = libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr() as *mut libc::integer_t,
            &mut count,
        );
        
        if result == libc::KERN_SUCCESS {
            let info = info.assume_init();
            Some(info.resident_size as usize)
        } else {
            None
        }
    }
}
```

### Fallback Implementation

```rust
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub fn get_rss_bytes() -> Option<usize> {
    None  // Unsupported platform
}
```

### Format Function

```rust
pub fn format_rss(bytes: usize) -> String {
    const GB: usize = 1_000_000_000;
    const MB: usize = 1_000_000;
    const KB: usize = 1_000;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
```

### Module Export

Update `src/memory/mod.rs`:

```rust
mod rss;
pub use rss::{get_rss_bytes, format_rss};
```

---

## Acceptance Criteria

- [ ] `get_rss_bytes()` works on Linux
- [ ] `format_rss()` formats correctly at all scale boundaries
- [ ] Functions exported from `memory` module
- [ ] Unit tests for formatting
- [ ] No panics on unsupported platforms

---

## Implementation Guide

### Suggested Approach

1. **Create `src/memory/rss.rs`** with implementations above

2. **Update `src/memory/mod.rs`**:
   ```rust
   mod rss;
   pub use rss::{get_rss_bytes, format_rss};
   ```

3. **Add libc dependency** if not present (already in Cargo.toml for HiGHS)

4. **Add tests** for format function

### Key Files to Modify

| File | Changes |
|------|---------|
| `src/memory/rss.rs` | New file |
| `src/memory/mod.rs` | Add module export |

### Pitfalls to Avoid

- ⚠️ Don't panic if `/proc/self/statm` unavailable
- ⚠️ Use decimal units (1 GB = 10^9) not binary (1 GiB = 2^30) for consistency with DHAT output
- ⚠️ Handle integer overflow in format calculations

---

## Testing Requirements

### Unit Tests

```rust
#[test]
fn test_format_rss_bytes() {
    assert_eq!(format_rss(500), "500 B");
}

#[test]
fn test_format_rss_kilobytes() {
    assert_eq!(format_rss(1_500), "1.50 KB");
    assert_eq!(format_rss(999_999), "1000.00 KB");
}

#[test]
fn test_format_rss_megabytes() {
    assert_eq!(format_rss(1_500_000), "1.50 MB");
    assert_eq!(format_rss(150_000_000), "150.00 MB");
}

#[test]
fn test_format_rss_gigabytes() {
    assert_eq!(format_rss(1_500_000_000), "1.50 GB");
    assert_eq!(format_rss(8_000_000_000), "8.00 GB");
}

#[cfg(target_os = "linux")]
#[test]
fn test_get_rss_bytes_linux() {
    let rss = get_rss_bytes();
    assert!(rss.is_some(), "get_rss_bytes should work on Linux");
    assert!(rss.unwrap() > 0, "RSS should be positive");
}
```

---

## Documentation Requirements

- [ ] Doc comments for `get_rss_bytes()`
- [ ] Doc comments for `format_rss()`
- [ ] Module-level documentation

---

## Effort Estimate

**Points**: 2
**Confidence**: High
**Rationale**: Straightforward implementation with clear scope

---

## Definition of Done

- [ ] `rss.rs` module created
- [ ] Functions working on Linux
- [ ] Tests passing
- [ ] Exported from memory module
- [ ] Documentation complete
- [ ] PR merged
