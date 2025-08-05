# GitHub Actions Workflows

This directory contains GitHub Actions workflows for continuous integration and testing.

## Workflows

### `ci.yml` - Main CI Pipeline
**Triggers:** Push to `main`/`develop`, Pull Requests to `main`

**Jobs:**
1. **Quick Check** - Fast feedback on every PR
   - Runs tests on Python 3.11
   - Verifies CLI installation
   - Provides rapid feedback (< 2 minutes)

2. **Comprehensive Tests** - Full test matrix
   - Tests on Python 3.9, 3.10, 3.11, 3.12
   - Code quality checks (flake8, black, isort, mypy)
   - Full test suite execution
   - CLI functionality verification
   - Only runs on main branch pushes or ready PRs

3. **Security Check** - Security scanning
   - Bandit security linter
   - Safety dependency vulnerability check
   - Runs on PRs and main branch

### `tests.yml` - Legacy Test Workflow
**Triggers:** Push/PR to `main`

Simple test workflow that runs the full test matrix on all Python versions.

### `coverage.yml` - Test Coverage
**Triggers:** Push/PR to `main`

- Generates test coverage reports
- Uploads to Codecov (if configured)
- Adds coverage comments to PRs

## Workflow Features

### Performance Optimizations
- **Dependency Caching**: Pip dependencies are cached per Python version
- **Conditional Execution**: Comprehensive tests only run when needed
- **Fast Feedback**: Quick check runs first for immediate PR feedback

### Quality Gates
- **Syntax Errors**: Fail immediately on Python syntax errors
- **Code Formatting**: Enforce Black code formatting
- **Import Sorting**: Enforce isort import organization
- **Type Checking**: Run mypy type checking (warnings only)
- **Security**: Scan for security vulnerabilities

### Test Coverage
- **Unit Tests**: 66 comprehensive unit tests
- **CLI Integration**: Real CLI command testing
- **Cross-Platform**: Tests run on Ubuntu (easily extendable to Windows/macOS)
- **Multiple Python Versions**: Ensures compatibility across Python 3.9-3.12

## Status Badges

Add these badges to your README.md:

```markdown
![CI](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/CI/badge.svg)
![Tests](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/Tests/badge.svg)
![Coverage](https://github.com/kylebeggs/NASCARFantasyPredictor/workflows/Test%20Coverage/badge.svg)
```

## Local Development

To run the same checks locally:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Check code formatting
black --check nascar_fantasy_predictor/ tests/
isort --check-only nascar_fantasy_predictor/ tests/

# Run linting
flake8 nascar_fantasy_predictor/

# Type checking
mypy nascar_fantasy_predictor/ --ignore-missing-imports

# Generate coverage report
pytest tests/ --cov=nascar_fantasy_predictor --cov-report=html
```

## Customization

### Adding New Python Versions
Update the matrix in `ci.yml`:
```yaml
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12", "3.13"]
```

### Adding New Checks
Add steps to the comprehensive-test job:
```yaml
- name: Custom Check
  run: |
    custom-linter nascar_fantasy_predictor/
```

### Branch Protection
Recommended branch protection rules for `main`:
- Require status checks to pass before merging
- Require "Quick Check (Python 3.11)" and "comprehensive-test" checks
- Require branches to be up to date before merging
- Require review from code owners