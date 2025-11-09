# Logging Implementation Tickets - Quick Reference

**Document**: See full tickets in [`LOGGING-TICKETS.md`](./LOGGING-TICKETS.md)  
**Total Tickets**: 32  
**Total Effort**: 50 days (80-100 hours part-time over 10 weeks)

---

## Sprint Breakdown

| Sprint | Week | Tickets | Focus | Estimated Days |
|--------|------|---------|-------|---------------|
| **Sprint 0** | Week 1 | LOG-001 to LOG-003 | Preparation & Baseline | 3 days |
| **Sprint 1** | Weeks 2-3 | LOG-004 to LOG-012 | Infrastructure Setup | 10 days |
| **Sprint 2** | Weeks 4-5 | LOG-013 to LOG-020 | Code Migration (Training Loop) | 10 days |
| **Sprint 3** | Weeks 6-7 | LOG-021 to LOG-026 | Advanced Features (JSON, CLI flags) | 10 days |
| **Sprint 4-5** | Weeks 8-10 | LOG-027 to LOG-032 | Cleanup & Release | 7 days |

---

## Critical Path

```
Sprint 0: Preparation
  LOG-001 (Design Review) → LOG-002 (Baseline) → LOG-003 (Feature Branch)
                                                         ↓
Sprint 1: Infrastructure
  LOG-004 (Dependencies) → LOG-005 (Module Structure) → LOG-006 (Config Types)
       ↓                                                       ↓
  LOG-007 (Terminal Formatter) ← LOG-008 (LogContext) ←─────┘
       ↓
  LOG-009 (PowersLogger) → LOG-010 (Update Config) → LOG-011 (Initialize) → LOG-012 (Test)
                                                                                    ↓
Sprint 2: Code Migration
  LOG-013 (Training Headers) → LOG-014 (Iteration Rows) → LOG-015 (Timing Detail)
       ↓
  LOG-016 (Simulation) → LOG-017 (Greetings) → LOG-018 (Error Handling)
       ↓
  LOG-019 (Debug Logging) → LOG-020 (Verify Complete)
                                   ↓
Sprint 3: Advanced Features
  LOG-021 (Inline Formatting) → LOG-022 (JSON Formatter) → LOG-023 (CLI Flags)
       ↓
  LOG-024 (File Sink) → LOG-025 (Documentation) → LOG-026 (Final Testing)
                                                         ↓
Sprint 4-5: Cleanup & Release
  LOG-027 (Remove log.rs) → LOG-028 (Remove Env Vars) → LOG-029 (Update Docs)
       ↓
  LOG-030 (JSON Schema) → LOG-031 (Final Checks) → LOG-032 (Release)
```

---

## Ticket Quick Reference

### Sprint 0: Preparation (Week 1)

| ID | Title | Effort | Priority | Blocks |
|----|-------|--------|----------|--------|
| LOG-001 | Design Review and Approval | 0.5d | Critical | All |
| LOG-002 | Capture Baseline Measurements | 0.5d | Critical | LOG-009 |
| LOG-003 | Create Feature Branch | 0.1d | Critical | Sprint 1 |

**Deliverable**: Approved design, baselines, feature branch ready

---

### Sprint 1: Infrastructure Setup (Weeks 2-3)

| ID | Title | Effort | Priority | Key Deliverable |
|----|-------|--------|----------|-----------------|
| LOG-004 | Add Log Crate Dependencies | 0.5d | Critical | `log`, `atty` added |
| LOG-005 | Create Logging Module Structure | 0.5d | Critical | `src/logging/` created |
| LOG-006 | Implement Configuration Types | 1d | Critical | `LoggingConfig` works |
| LOG-007 | Implement Terminal Formatter | 2d | High | Pass-through formatter |
| LOG-008 | Implement LogContext | 1d | High | Thread-local context |
| LOG-009 | Implement PowersLogger | 2d | Critical | Logger working |
| LOG-010 | Update Config Struct | 0.5d | High | `Config.logging` field |
| LOG-011 | Initialize Logger in Entry Point | 1d | Critical | Logger active |
| LOG-012 | Integration Testing | 1.5d | Critical | Zero visual changes |

**Deliverable**: Working logging infrastructure, CLI output unchanged

---

### Sprint 2: Code Migration - Training (Weeks 4-5)

| ID | Title | Effort | Priority | Migrates |
|----|-------|--------|----------|----------|
| LOG-013 | Migrate Training Loop Greeting | 1d | High | `training_greeting()` |
| LOG-014 | Migrate Training Iteration Rows | 2d | Critical | `training_table_row()` |
| LOG-015 | Migrate Timing Detail Logging | 1.5d | Medium | Env var removed |
| LOG-016 | Migrate Simulation Logging | 1.5d | High | Simulation functions |
| LOG-017 | Migrate Greetings/Farewells | 1d | Medium | App-level logs |
| LOG-018 | Migrate Error Handling | 1.5d | High | All `eprintln!` |
| LOG-019 | Migrate Debug Logging | 1d | Medium | `debug!` macros |
| LOG-020 | Verify No println! Remains | 0.5d | Critical | Final audit |

**Deliverable**: All training/simulation code uses structured logging

---

### Sprint 3: Advanced Features (Weeks 6-7)

| ID | Title | Effort | Priority | Adds |
|----|-------|--------|----------|------|
| LOG-021 | Remove Dependency on log.rs | 2d | Critical | Inlined formatting |
| LOG-022 | Implement JSON Formatter | 2d | High | JSON output |
| LOG-023 | Add CLI Flags | 1.5d | High | `--log-level`, `--log-format` |
| LOG-024 | Implement File Sink | 1.5d | Medium | File output |
| LOG-025 | Create User Documentation | 1.5d | High | LOGGING-GUIDE.md |
| LOG-026 | Final Testing | 2d | Critical | Validation complete |

**Deliverable**: Complete feature set with JSON, CLI flags, file output

---

### Sprint 4-5: Cleanup & Release (Weeks 8-10)

| ID | Title | Effort | Priority | Action |
|----|-------|--------|----------|--------|
| LOG-027 | Remove src/log.rs Module | 1d | High | Delete old code |
| LOG-028 | Remove Environment Variables | 0.5d | Medium | Remove env checks |
| LOG-029 | Update All Documentation | 2d | High | Final docs sweep |
| LOG-030 | Update JSON Schema | 1d | Medium | config.schema.json |
| LOG-031 | Final Pre-Release Checks | 1.5d | Critical | QA complete |
| LOG-032 | Release and Merge | 0.5d | Critical | v0.3.0 shipped |

**Deliverable**: Clean codebase, v0.3.0 released

---

## Parallel Work Opportunities

**Sprint 1**: Single developer (sequential dependencies)  
**Sprint 2**: 
- Track A: Training loop (LOG-013 to LOG-015)
- Track B: Simulation + greetings (LOG-016 to LOG-017)
- Track C: Error handling (LOG-018 to LOG-019)

**Sprint 3**:
- Track A: Formatting (LOG-021)
- Track B: JSON formatter (LOG-022)
- Track C: Documentation (LOG-025)

**Sprint 4-5**: Single developer (cleanup tasks)

---

## Risk Management

### High-Risk Tickets
- **LOG-012**: Integration testing (gate for Phase 1)
- **LOG-014**: Training iteration rows (visual output critical)
- **LOG-020**: Final migration audit (gate for Phase 2)
- **LOG-026**: Final testing (gate for Phase 3)
- **LOG-031**: Pre-release checks (gate for release)

### Mitigation
- Extensive baseline testing (LOG-002)
- Visual regression tests at each phase gate
- Performance benchmarks at each phase
- Rollback plan documented in implementation plan

---

## Testing Strategy

### Per-Sprint Testing
- **Sprint 0**: Capture baselines
- **Sprint 1**: Compare CLI output to baseline (byte-for-byte)
- **Sprint 2**: Visual regression + all tests pass
- **Sprint 3**: Feature testing + performance validation
- **Sprint 4-5**: Final QA + release checks

### Acceptance Criteria
- CLI output unchanged (or documented)
- All 189+ tests pass
- < 1% performance regression
- 100% test coverage for logging module
- No clippy warnings

---

## Documentation Deliverables

1. **Design Documents** (Already created):
   - `LOGGING-DESIGN.md` (37KB) - Architecture
   - `LOGGING-IMPLEMENTATION-PLAN.md` (27KB) - Task breakdown
   - `LOGGING-SUMMARY.md` (8KB) - Executive summary
   - `LOGGING-QUICK-REFERENCE.md` (7KB) - Developer cheat sheet

2. **Created During Implementation**:
   - `LOGGING-GUIDE.md` (LOG-025) - User guide
   - Updated `README.md` (LOG-029)
   - Updated `INPUT-SPECIFICATION.md` (LOG-029)
   - Updated `config.schema.json` (LOG-030)

---

## Success Metrics

### Code Quality
- ✅ Zero `println!`/`eprintln!` in production code
- ✅ All log macros used appropriately
- ✅ 100% test coverage for logging module
- ✅ Zero clippy warnings

### Performance
- ✅ < 1% regression in training benchmarks
- ✅ < 20KB binary size increase
- ✅ Zero overhead for disabled log levels

### Functionality
- ✅ All log levels work (ERROR to TRACE)
- ✅ Terminal format preserves visual output
- ✅ JSON format parseable
- ✅ File output works
- ✅ CLI flags override config

### Documentation
- ✅ User guide complete
- ✅ All configuration options documented
- ✅ Examples for common use cases
- ✅ Troubleshooting guide

---

## Next Steps

1. **Review**: Read [`LOGGING-TICKETS.md`](./LOGGING-TICKETS.md) in detail
2. **Discuss**: Hold design review meeting (LOG-001)
3. **Prepare**: Capture baselines (LOG-002)
4. **Start**: Create feature branch (LOG-003)
5. **Execute**: Begin Sprint 1 tickets

---

**Document Version**: 1.0  
**Created**: 2025-11-09  
**Epic Status**: Ready for Implementation
