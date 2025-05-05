# Contributing to SemiCART

Thank you for considering contributing to SemiCART! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for SemiCART.

- Use a clear and descriptive title for the issue
- Describe the exact steps to reproduce the problem
- Provide specific examples to demonstrate the steps
- Describe the behavior you observed after following the steps
- Explain which behavior you expected to see instead and why
- Include screenshots or animated GIFs if possible
- Include details about your environment (OS, Python version, etc.)

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for SemiCART.

- Use a clear and descriptive title for the issue
- Provide a detailed description of the suggested enhancement
- Explain why this enhancement would be useful to most users
- List some other applications where this enhancement exists, if applicable
- Specify which version of SemiCART you're using

### Pull Requests

- Fill in the required template
- Do not include issue numbers in the PR title
- Follow the style guidelines
- Update the documentation with details of changes if applicable
- Update the CHANGELOG.md with details of changes
- The PR should work for Python 3.7 and above

## Development Guidelines

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/WeightedAI/semicart
   cd semicart
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting:
  ```bash
  black semicart tests examples
  ```
- Sort imports with [isort](https://pycqa.github.io/isort/):
  ```bash
  isort semicart tests examples
  ```
- Lint code with [flake8](https://flake8.pycqa.org/):
  ```bash
  flake8 semicart tests examples
  ```

### Testing

- Add tests for new features
- Ensure all tests pass before submitting a PR:
  ```bash
  pytest
  ```
- Aim for high test coverage:
  ```bash
  pytest --cov=semicart
  ```

### Documentation

- Update the documentation to reflect your changes
- Follow [NumPy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html)
- Build and check the documentation:
  ```bash
  cd docs
  make html
  ```

## License

By contributing to SemiCART, you agree that your contributions will be licensed under the project's [MIT License](LICENSE). 