# T-001: Create Spec Directory Structure and Frontmatter Template

## Epic

Epic 1: Foundation

## Dependencies

None — this is the first ticket.

## Description

Create the `docs/specs/` directory hierarchy and a frontmatter template that all spec files will use. Also create a `docs/specs/README.md` index file.

## Acceptance Criteria

- [ ] Directory structure exists: `docs/specs/{00-overview,01-math,02-data-model,03-architecture,04-hpc,05-config,06-deferred}/`
- [ ] `docs/specs/TEMPLATE.md` contains the standard frontmatter and section structure
- [ ] `docs/specs/README.md` contains a table listing all planned spec files with status column
- [ ] Template includes: YAML frontmatter (status, source_sections, last_reviewed, review_notes), Purpose section, main content section, Cross-References section

## Files to Create

- `docs/specs/README.md`
- `docs/specs/TEMPLATE.md`
- All subdirectories listed above (empty)

## Technical Details

### Step 1: Create directories

```bash
mkdir -p docs/specs/{00-overview,01-math,02-data-model,03-architecture,04-hpc,05-config,06-deferred}
```

### Step 2: Create TEMPLATE.md

```markdown
---
status: draft
source_sections: []
last_reviewed: null
review_notes: ""
---

# [Spec Title]

## Purpose

[1-2 sentences: what this spec covers and why it exists as a separate document]

## [Main Content Sections]

...

## Cross-References

- [Link to related spec] — [brief relationship description]
```

### Step 3: Create README.md

A table with columns: File, Category, Status, Source Sections, Description. List all 35 planned spec files.

## Definition of Done

Directory structure and templates are committed. The README index accurately lists all planned specs.
