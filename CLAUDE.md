# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests
julia --project=. -e 'import Pkg; Pkg.test()'

# Run tests matching a pattern (e.g. "basic", "rules", "iddict")
julia --project=. -e 'import Pkg; Pkg.test(; test_args=["<pattern>"])'

```

### Code Formatting
```bash
# Format changed files relative to a branch
git runic <base-branch>

# Install pre-commit hook for automatic formatting
pre-commit install
```
CI enforces Runic formatting on all PRs.
