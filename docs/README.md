# POWE.RS Documentation

**Version**: 0.2.0 (Pre-Release)  
**Last Updated**: October 6, 2025

Welcome to the POWE.RS documentation! This guide will help you understand, use, and contribute to POWE.RS.

---

## 📚 Documentation Guide

### For First-Time Users

**Start here** to get POWE.RS running:

1. **[Installation Guide](guides/INSTALLATION.md)** - Get POWE.RS installed on your system
2. **[Quick Start Tutorial](guides/QUICKSTART.md)** - Run your first optimization in 5 minutes
3. **[Input Specification](reference/INPUT-SPECIFICATION.md)** - Understand the JSON input format
4. **[Troubleshooting](guides/TROUBLESHOOTING.md)** - Common issues and solutions

### For Power System Engineers

Learn to model hydrothermal dispatch problems:

- **[Input Specification](reference/INPUT-SPECIFICATION.md)** - Complete field-by-field documentation with examples
- **[Troubleshooting](guides/TROUBLESHOOTING.md)** - Error messages with clear fixes

### For Researchers & Algorithm Developers

Understand the SDDP implementation:

- **[Algorithm Overview](algorithm/SDDP-OVERVIEW.md)** - SDDP theory and implementation approach
- **[Performance Analysis](performance/PARALLELISM.md)** - Parallel efficiency and optimization strategies
- **[Testing Guide](development/TESTING.md)** - Comprehensive testing documentation

### For Contributors

Help improve POWE.RS:

- **[Development Guide](development/TESTING.md)** - Testing, benchmarks, and code organization
- **[Architecture Documentation](architecture/)** - Design decisions and implementation details

---

## 📖 Documentation Structure

### `/guides` - User Guides

Practical guides for getting started and solving common problems:

| Document                                        | Purpose                         | Audience  |
| ----------------------------------------------- | ------------------------------- | --------- |
| [INSTALLATION.md](guides/INSTALLATION.md)       | Installation instructions       | All users |
| [QUICKSTART.md](guides/QUICKSTART.md)           | First optimization in 5 minutes | New users |
| [TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md) | Common errors and solutions     | All users |

### `/reference` - Reference Documentation

Complete technical specifications:

| Document                                                   | Purpose                         | Audience   |
| ---------------------------------------------------------- | ------------------------------- | ---------- |
| [INPUT-SPECIFICATION.md](reference/INPUT-SPECIFICATION.md) | JSON input format specification | All users  |
| [API-REFERENCE.md](reference/API-REFERENCE.md)             | Library API documentation       | Developers |

### `/algorithm` - Algorithm Documentation

Mathematical background and implementation details:

| Document                                       | Purpose                 | Audience    |
| ---------------------------------------------- | ----------------------- | ----------- |
| [SDDP-OVERVIEW.md](algorithm/SDDP-OVERVIEW.md) | SDDP algorithm overview | Researchers |

### `/performance` - Performance Documentation

Performance analysis and optimization strategies:

| Document                                         | Purpose                     | Audience       |
| ------------------------------------------------ | --------------------------- | -------------- |
| [PARALLELISM.md](performance/PARALLELISM.md)     | Parallel execution analysis | Algorithm devs |
| [CUT-SELECTION.md](performance/CUT-SELECTION.md) | Cut selection performance   | Algorithm devs |

### `/architecture` - Architecture & Design

Design decisions and implementation rationale:

| Document                                                      | Purpose                           | Audience     |
| ------------------------------------------------------------- | --------------------------------- | ------------ |
| [BATCH-CUT-SELECTION.md](architecture/BATCH-CUT-SELECTION.md) | Batch cut selection design        | Contributors |
| [PRODUCTION-API.md](architecture/PRODUCTION-API.md)           | Factory API vs Builder API design | Contributors |
| [CONSTRUCTION-API.md](architecture/CONSTRUCTION-API.md)       | SDDP construction approaches      | Contributors |

### `/development` - Development Documentation

For contributors and maintainers:

| Document                             | Purpose                          | Audience     |
| ------------------------------------ | -------------------------------- | ------------ |
| [TESTING.md](development/TESTING.md) | Testing guide and best practices | Contributors |

---

## 🚀 Quick Navigation

### I want to...

| Goal                                   | Document to Read                                           |
| -------------------------------------- | ---------------------------------------------------------- |
| Install POWE.RS                        | [INSTALLATION.md](guides/INSTALLATION.md)                  |
| Run my first optimization              | [QUICKSTART.md](guides/QUICKSTART.md)                      |
| Understand the input JSON files        | [INPUT-SPECIFICATION.md](reference/INPUT-SPECIFICATION.md) |
| Fix an error message                   | [TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)            |
| Use POWE.RS as a library               | [API-REFERENCE.md](reference/API-REFERENCE.md)             |
| Understand how SDDP works              | [SDDP-OVERVIEW.md](algorithm/SDDP-OVERVIEW.md)             |
| Understand performance characteristics | [PARALLELISM.md](performance/PARALLELISM.md)               |
| Contribute code or tests               | [TESTING.md](development/TESTING.md)                       |
| Understand design decisions            | [architecture/](architecture/)                             |

---

## 📋 Documentation Status (Pre-Release)

POWE.RS is currently in **pre-release** (v0.2.0) preparing for the first major release (v1.0.0). Documentation reflects the current stable state.

### Available Documentation

- ✅ **Installation** - Complete
- ✅ **Input Specification** - Complete with JSON schemas
- ✅ **Troubleshooting** - Common errors documented
- ✅ **Testing Guide** - Comprehensive for contributors
- ✅ **Architecture Docs** - Design decisions documented
- ⚠️ **Quick Start** - In progress
- ⚠️ **API Reference** - In progress
- ⚠️ **Algorithm Overview** - In progress

### Planned for v1.0.0

- 📝 Complete user guide with hydrothermal modeling examples
- 📝 Comprehensive API reference with examples
- 📝 Case studies with real-world problems
- 📝 Performance tuning guide
- 📝 Contribution guide

---

## 🆘 Getting Help

### Documentation Issues

If you find:

- Missing information
- Unclear explanations
- Broken links
- Errors or typos

**Please open an issue**: https://github.com/rjmalves/powers/issues

### Using POWE.RS

For questions about using POWE.RS:

1. Check [TROUBLESHOOTING.md](guides/TROUBLESHOOTING.md)
2. Search existing [GitHub Issues](https://github.com/rjmalves/powers/issues)
3. Open a new issue with your question

### Contributing

We welcome contributions! See:

- [TESTING.md](development/TESTING.md) for development setup
- Architecture docs in [architecture/](architecture/) for design context

---

## 📄 License

POWE.RS is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## 📚 External Resources

### SDDP Theory

- Pereira, M. V. F., & Pinto, L. M. V. G. (1991). "Multi-stage stochastic optimization applied to energy planning." _Mathematical Programming_, 52(1-3), 359-375.
- Shapiro, A., et al. (2011). "Risk neutral and risk averse Stochastic Dual Dynamic Programming method." _European Journal of Operational Research_, 224(2), 375-391.

### Related Software

- [SDDP.jl](https://github.com/odow/SDDP.jl) - Julia implementation with extensive features
- [HiGHS](https://highs.dev/) - The LP/MIP solver used by POWE.RS

---

**Navigation**: [↑ Back to Top](#powers-documentation) | [Repository Root](../)
