# Copilot Instructions for Enzyme.jl

## Project Overview

Enzyme.jl is the Julia bindings for [Enzyme](https://github.com/EnzymeAD/enzyme), a high-performance automatic differentiation (AD) plugin for LLVM. Enzyme performs automatic differentiation of statically analyzable LLVM code and is designed to meet or exceed the performance of state-of-the-art AD tools.

**Key Features:**
- Automatic differentiation via `autodiff` function
- Supports both forward and reverse mode AD
- Works on optimized LLVM code for high performance
- GPU support (CUDA, ROCm, Metal)
- Tight integration with Julia's LLVM infrastructure

## Repository Structure

```
Enzyme.jl/
├── src/              # Main source code
│   ├── Enzyme.jl     # Main module
│   ├── compiler/     # Compiler integration
│   ├── llvm/         # LLVM utilities
│   ├── rules/        # Differentiation rules
│   └── analyses/     # Static analysis tools
├── lib/              # Sub-packages
│   ├── EnzymeCore/   # Core functionality
│   └── EnzymeTestUtils/  # Testing utilities
├── test/             # Test suite
│   ├── core/         # Core tests
│   ├── ext/          # Extension tests
│   ├── rules/        # Rule tests
│   └── integration/  # Integration tests
├── ext/              # Package extensions
├── deps/             # Build dependencies
├── docs/             # Documentation
├── examples/         # Example code
└── benchmark/        # Performance benchmarks
```

## Development Environment Setup

### Standard Installation
```bash
] add Enzyme
```

### Development Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/EnzymeAD/Enzyme.jl.git
   cd Enzyme.jl
   ```

2. Start Julia with the project:
   ```bash
   julia --project=.
   ```

3. Install dependencies:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

### Building Custom Enzyme (for Enzyme development)

To test changes to the underlying Enzyme library:

1. Initialize the deps project:
   ```bash
   julia --project=deps
   ```
   ```julia
   ] instantiate
   ```

2. Build custom Enzyme:
   ```bash
   julia --project=deps deps/build_local.jl
   ```

   Or for a specific branch:
   ```bash
   julia --project=deps deps/build_local.jl --branch mybranch
   ```

   Or from a local Enzyme copy:
   ```bash
   julia --project=deps deps/build_local.jl ../path/to/Enzyme
   ```

This generates `LocalPreferences.toml` with the custom Enzyme_jll path.

## Code Style and Conventions

### Formatting
- **Formatter:** Runic (Julia formatter)
- **Pre-commit hook:** Available via `.pre-commit-config.yaml`
- **CI check:** Automatic formatting checks on PRs

To format code:
```bash
git runic <base-branch>
```

Or install the pre-commit hook:
```bash
pre-commit install
```

### Julia Conventions
- Follow standard Julia style guidelines
- Use meaningful variable names
- Add docstrings for exported functions
- Keep functions focused and small
- Use type annotations judiciously (where they improve clarity or performance)

## Testing

### Running Tests

Run all tests:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or using the test script:
```bash
julia --project=. test/runtests.jl
```

Run tests with multiple threads:
```bash
julia --project=. --threads=2 test/runtests.jl
```

Run specific tests:
```bash
julia --project=. test/runtests.jl <test_pattern>
```

### Test Structure
- Tests use `ParallelTestRunner` for parallel execution
- GPU tests (cuda, metal, amdgpu) are skipped by default
- Integration tests run in separate environments
- Thread tests run with `--threads=2`

### Common Test Commands
```bash
# Run basic tests
julia --project=. -e 'using Pkg; Pkg.test("Enzyme"; test_args=`basic`)'

# Run with verbose output
julia --project=. test/runtests.jl --verbose
```

## Building and CI

### CI Workflows
- **CI.yml**: Main test suite across Julia versions (1.10, 1.11, nightly) and platforms (Ubuntu, macOS, Windows)
- **Format.yml**: Automatic code formatting checks using Runic
- **Documentation.yml**: Documentation build and deployment
- **Integration.yml**: Integration tests with other packages

### Build Configuration
- Requires Julia 1.10 or higher
- Tests against packaged and local Enzyme builds
- Optional LLVM assertions for debugging
- Coverage reporting to Codecov

### Local Build Commands
```bash
# Instantiate dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run with LLVM arguments
JULIA_LLVM_ARGS='--opaque-pointers' julia --project=.
```

## Common Development Tasks

### Adding a New Differentiation Rule
1. Add rule to `src/rules/`
2. Follow existing rule patterns (check `src/internal_rules.jl`)
3. Add tests in `test/rules/`
4. Update documentation if it's a user-facing rule

### Debugging AD Issues
1. Enable assertions: Build Julia with `FORCE_ASSERTIONS=1 LLVM_ASSERTIONS=1`
2. Check LLVM IR: Use `@code_llvm` to inspect generated code
3. Use verbose test output: `--verbose` flag
4. Check enzyme logs with custom build

### Working with Extensions
Extensions are in `ext/` directory:
- EnzymeBFloat16sExt
- EnzymeChainRulesCoreExt
- EnzymeGPUArraysCoreExt
- EnzymeLogExpFunctionsExt
- EnzymeSpecialFunctionsExt
- EnzymeStaticArraysExt

Extensions load automatically when weak dependencies are loaded.

## Key Files and Their Purpose

- `src/Enzyme.jl` - Main module, exports and core API
- `src/compiler.jl` - Compiler integration and autodiff implementation
- `src/api.jl` - C API bindings to Enzyme
- `src/rules/` - Custom differentiation rules
- `lib/EnzymeCore/` - Core types and interfaces (minimal dependencies)
- `lib/EnzymeTestUtils/` - Testing utilities and test data generators

## Dependencies

### Main Dependencies
- EnzymeCore (from lib/EnzymeCore)
- Enzyme_jll (LLVM plugin binary)
- LLVM.jl (LLVM interface)
- GPUCompiler.jl (GPU compilation)

### Weak Dependencies (for extensions)
- ChainRulesCore
- StaticArrays
- SpecialFunctions
- BFloat16s
- GPUArraysCore
- LogExpFunctions

## Documentation

Build documentation locally:
```bash
julia --project=docs docs/make.jl
```

Documentation is in `docs/src/`:
- `index.md` - Main documentation
- `faq.md` - Frequently asked questions
- `dev_docs.md` - Developer documentation
- `api.md` - API reference

## Contribution Guidelines

### Pull Request Requirements
1. Include unit tests for new functionality
2. Ensure code passes Runic formatting
3. Keep changes isolated and focused
4. Add docstrings for exported functions
5. Update documentation if adding user-facing features

### Issues and Communication
- GitHub Issues: Bug reports and feature requests
- Slack: #enzyme channel on Julia Slack
- Discourse: Julia Discourse with `enzyme` tag
- Mailing list: enzyme-dev Google Group

### Code of Conduct
This project follows the [Julia Community Standards](https://julialang.org/community/standards/).

## Performance Considerations

- Enzyme works on optimized LLVM IR for best performance
- Use type-stable code for optimal AD performance
- Be aware of Julia's compilation model (first call compiles)
- GPU kernels require special handling
- Consider using `Const` annotation for constants to improve performance

## Troubleshooting

### Common Issues
1. **LLVM version mismatch**: Ensure Enzyme is built against Julia's LLVM
2. **Type instability**: Use `@code_warntype` to check for type issues
3. **Rule not found**: Check if a custom rule is needed for your function
4. **GPU errors**: Ensure proper GPU setup and compatible CUDA/ROCm versions

### Getting Help
1. Check FAQ in documentation
2. Search existing GitHub issues
3. Ask on Julia Slack or Discourse
4. Create a minimal reproducible example
5. Include Julia version, Enzyme version, and OS information
