# T-004b: Extract System Element Modeling Overview Spec

## Epic

Epic 2: Mathematical Formulation Specs

## Dependencies

- T-001 (directory structure)
- T-002 (notation-conventions.md)

## Description

Extract the system element modeling overview (§3) into its own spec. This section describes _what each physical element is_, its decision variables, its connections to other elements, and its role in the optimization — before the reader encounters the full LP formulation. It serves as the conceptual foundation that makes the LP constraints in `lp-formulation.md` and `equipment-formulations.md` understandable.

## Acceptance Criteria

- [ ] `docs/specs/01-math/system-elements.md` extracted from MATH_FORMULATIONS §3 (3.1-3.9)
- [ ] Covers each system element with its pattern: physical meaning, decision variables, connections, role in optimization
- [ ] Elements covered: buses, transmission lines, thermal plants, hydro plants, non-controllable sources (deferred note), pumping stations, import/export contracts
- [ ] Includes the system architecture overview diagram reference (§3.1)
- [ ] Includes the summary table mapping physical elements to LP components (§3.9)
- [ ] Cross-references `lp-formulation.md` for the complete LP constraint formulation
- [ ] Cross-references `equipment-formulations.md` for detailed per-equipment constraint math
- [ ] Under 500 lines, correct frontmatter

## Files to Create

- `docs/specs/01-math/system-elements.md`

## Technical Details

Source: `MATHEMATICAL_FORMULATIONS.md` §3 (3.1-3.9), approximately lines 271-750

### Section mapping

- §3.1 System architecture overview — diagram, high-level description of how elements interact
- §3.2 Buses — regional subsystems, load balance nodes, decision variables (deficit, excess)
- §3.3 Transmission lines — interconnections, flow limits, losses, decision variables (flow)
- §3.4 Thermal plants — generation units, cost curves, operational limits, decision variables (generation)
- §3.5 Hydro plants — reservoirs, turbines, water balance, decision variables (storage, turbining, spillage, generation)
- §3.6 Non-controllable sources — wind/solar (DEFERRED) → brief description + reference to `deferred-features.md`
- §3.7 Pumping stations — pump-storage coupling, decision variables (pumping power, pumped volume)
- §3.8 Import/export contracts — energy exchanges, fixed vs flexible, decision variables (import/export)
- §3.9 Summary table — maps each physical element to its LP variables, constraints, and objective function contributions

### Reading order context

This spec should be read AFTER `sddp-algorithm.md` (understanding what SDDP solves) and BEFORE `lp-formulation.md` (seeing the complete formulation). The logical chain is:

1. `sddp-algorithm.md` — what is SDDP and how does it work?
2. **`system-elements.md`** — what physical elements exist and what do they contribute?
3. `lp-formulation.md` — the complete LP with all constraints assembled
4. `equipment-formulations.md` — detailed per-equipment constraint derivations

### Cross-references to add

- `notation-conventions.md` — for variable naming conventions and index sets
- `lp-formulation.md` — "for the assembled LP constraints, see..."
- `equipment-formulations.md` — "for detailed per-equipment formulations, see..."
- `hydro-production-models.md` — "for hydro production function alternatives (FPHA, linearized head), see..."
- `deferred-features.md` — for non-controllable sources and batteries

## Definition of Done

File created with all system element descriptions preserved, decision variable tables intact, diagram references valid, cross-references to other math specs in place, under 500 lines.
