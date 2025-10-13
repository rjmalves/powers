## POWE.RS — Copilot instructions for AI coding agents

Purpose: give an AI coding agent the minimum, concrete knowledge to make safe, useful changes in this repository.

Quick checklist (agent pre-flight)

- Run pre-checks before edits: `cargo fmt -- --check` and `cargo clippy --all-targets --all-features -- -D warnings`.
- Run full build: `cargo build --workspace --release`.
- Run tests: `cargo test --workspace` (use `-- --nocapture` for debug output).
- When adding or changing JSON inputs update the corresponding schema in `schemas/*.schema.json` and add tests under `tests/` and fixtures under `tests/fixtures/`.

Primary entry points & files

- CLI / runtime: `src/main.rs` (parses args) and `src/lib.rs::run` (factory API used by binary).
- Input parsing & types: `src/input.rs` — contains Config, SystemInput, GraphInput, Recourse, and readers.
- Input validation: `src/input_validation.rs` — authoritative validation and error reporting.
- Core algorithm: `src/sddp/` and `src/graph.rs` — SDDP construction and training.
- Tests: `tests/` contains unit & integration tests. Use fixtures in `tests/fixtures/`.
- JSON schemas: `schemas/*.schema.json` provide IDE validation and are required for any input format changes.

Project-specific conventions and patterns

- IDs are zero-based contiguous integers (0..N-1). Validation relies on sequential IDs — see `validate_id_range_comprehensive` in `src/input_validation.rs`.
- There are two public construction patterns: Factory API (`SddpAlgorithm::from_files`) for zero-argument runs and Builder API (`SddpInstanceBuilder`) for programmatic parameter sweeps. Prefer Builder API for tests and benchmarks.
- Threading uses Rayon and `num_threads` in `config.json` (None => auto-detect). Avoid over-subscribing cores.
- CSV output is controlled via `config.output_path` (omit or null to disable — useful for tests).
- Solver integration uses `highs-sys` (C bindings). Changes to solver interfaces are high-risk and require CI/bench validation.

What to change where (common tasks)

- Add new input fields/types: edit `src/input.rs`, update `schemas/*.schema.json`, add validation in `src/input_validation.rs`, and add tests in `tests/*` + fixture in `tests/fixtures/`.
- Add algorithm logic: edit `src/sddp/` and add focused unit tests under `tests/` and benchmarks under `benches/`.
- Add CLI flags or behavior: edit `src/main.rs` and `src/lib.rs` run() contract.

Agent rules (must follow)

1. Always run pre-checks (fmt + clippy) and unit tests locally before proposing changes. Fail fast on warnings (`-D warnings`).
2. Do not change `schemas/*.schema.json` without adding or updating at least one JSON fixture in `tests/fixtures/` and a test in `tests/` that demonstrates the schema change.
3. Maintain backward compatibility for public JSON formats unless the ticket explicitly permits breaking changes. When changing input formats, provide a conversion helper and tests.
4. Keep diffs small and focused. One logical change per PR. If touching core algorithm or solver interface, include a performance/regression note and run `benches/` where applicable.
5. When adding new behavior that affects outputs (CSV shape, new files), update `docs/` (reference/INPUT-SPECIFICATION.md or relevant doc) and include example `examples/` input set if feasible.

Useful commands (copyable)

```bash
# Pre-checks
cargo fmt -- --check
cargo clippy --all-targets --all-features -- -D warnings

# Build & test
cargo build --workspace --release
cargo test --workspace

# Run the local binary with example inputs
target/release/powers examples/04-cascade
```

CI notes

- CI runs `cargo test` and coverage. Avoid non-deterministic tests. Use `with_seed()` in tests that rely on randomness.
- Large changes touching performance or solver bindings should include benchmark runs under `benches/` and a short performance summary in the PR description.

If unclear or missing info

- Look at `docs/` (especially `docs/reference/INPUT-SPECIFICATION.md`) and `schemas/` for authoritative format rules.
- If you can't find a test that covers your intended change, add one — tests are the contract.

If you want me to adapt or expand this file (for example adding a Copilot toolset mapping or recommended `.vscode/tasks.json`), say which tools/commands you want exposed to the agent and I'll add an example.
