# Contributing to AI Glasses Platform

Thank you for your interest in contributing to the AI Glasses Platform!

## Getting Started

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/michaltakac/juicy-ai-glasses.git
cd juicy-ai-glasses
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
# Run all unit tests
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=aiglasses --cov-report=html

# Run integration tests (mock mode)
pytest tests/integration --mock

# Run specific test file
pytest tests/unit/test_config.py -v
```

### Code Style

We use:
- **ruff** for linting
- **black** for code formatting
- **mypy** for type checking

Run all checks:
```bash
ruff check src/ tests/
black --check src/ tests/
mypy src/aiglasses
```

Format code:
```bash
black src/ tests/
```

## Project Structure

```
aiglasses/
├── src/aiglasses/          # Main source code
│   ├── foundation/         # Foundation layer services
│   ├── sdk/               # SDK for app developers
│   ├── common/            # Shared utilities
│   ├── config.py          # Configuration
│   └── cli.py             # CLI tool
├── examples/              # Example applications
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── proto/                 # Protocol buffer definitions
├── configs/              # Configuration templates
├── systemd/              # Systemd service files
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## Making Changes

### Branches

- `main` - stable release branch
- `develop` - development branch
- Feature branches: `feature/your-feature`
- Bug fixes: `fix/bug-description`

### Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Maintenance

Example:
```
feat(vision): add Hailo HAT+ support

Implements HailoVisionBackend for object detection
using the Hailo AI HAT+.

Closes #123
```

### Pull Requests

1. Create a feature branch from `develop`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Create a pull request to `develop`

### Code Review

All changes require:
- At least one approval
- Passing CI checks
- No merge conflicts

## Architecture Guidelines

### Foundation Services

When adding a new Foundation service:

1. Create the service module in `src/aiglasses/foundation/`
2. Implement `BaseService` interface
3. Add gRPC servicer
4. Add proto definition (optional, we use JSON for simplicity)
5. Add unit tests
6. Add systemd service file

### SDK APIs

When adding SDK functionality:

1. Keep APIs simple and intuitive
2. Support both mock and real modes
3. Add comprehensive docstrings
4. Add unit tests
5. Update SDK documentation

### Testing

- All new code must have tests
- Aim for >80% coverage
- Use mock mode for unit tests
- HIL tests require `--hil` flag

## Hardware-in-the-Loop Testing

For testing on real Raspberry Pi hardware:

1. Set up a Raspberry Pi 5 with Pi OS Bookworm
2. Install the AI Camera and optional AI HAT+
3. Configure as a GitHub Actions self-hosted runner
4. Run tests with `--hil` flag

## Documentation

- Update README.md for major changes
- Add docstrings to all public APIs
- Update configuration examples
- Add examples for new features

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create PR to `main`
4. After merge, create release tag
5. CI builds and publishes release

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

Thank you for contributing! 🎉


