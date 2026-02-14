# T-026: Update Root README and Create Specs README

## Epic

Epic 6: Review and Finalize

## Dependencies

- T-025 (cross-reference validation complete)

## Description

Update the root `README.md` to reference the new spec structure instead of the monolithic docs. Update the `docs/specs/README.md` index with actual status of all spec files.

## Acceptance Criteria

- [ ] Root `README.md` updated: documentation section points to `docs/specs/README.md` instead of the monolithic files
- [ ] Root `README.md` implementation phases section removed (will be recreated in a fresh implementation plan)
- [ ] `docs/specs/README.md` has a complete table of all spec files with: file path, category, status (draft), description, approximate line count
- [ ] `docs/specs/README.md` includes a "How to Review" section explaining the frontmatter status workflow
- [ ] Original monolithic files noted as "archived reference" in the README

## Files to Modify

- `README.md` (root)
- `docs/specs/README.md`

## Technical Details

### Root README changes

- Update "Documentation" section to describe the new atomic spec structure
- Replace the monolithic file reference with a link to `docs/specs/README.md`
- Keep: Overview, Key Design Decisions, Project Structure (planned crates), Getting Started, License
- Remove: Implementation Phases table (outdated — fresh plan will be created later)
- Add note: "The original monolithic documentation files are preserved in `docs/` for reference during the transition period."

### Specs README updates

- Table with columns: Category | File | Status | Lines | Description
- Group by category (00-overview through 06-deferred)
- "How to Review" section explaining:
  - Read the spec file
  - Update `status` in frontmatter: `draft` → `under-review` → `approved` or `needs-changes`
  - Add `review_notes` if changes are needed
  - Set `last_reviewed` date

## Definition of Done

Root README updated, specs README complete and accurate, all links valid.
