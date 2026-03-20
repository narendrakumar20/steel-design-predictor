# Contributing to SteelML

Thank you for your interest in contributing to SteelML! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

### Our Standards

- **Be respectful**: Treat everyone with respect and kindness
- **Be collaborative**: Work together towards common goals
- **Be constructive**: Provide helpful feedback and suggestions
- **Be inclusive**: Welcome contributors from all backgrounds

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SteelML.git
   cd SteelML
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/vamsi-op/SteelML.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

1. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .  # Install package in editable mode
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Generate demo data and train models** (optional, for testing):
   ```bash
   python generate_dataset.py
   python train_model.py
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in the codebase
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add usage examples or tutorials

### Contribution Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Write or update tests** for your changes

4. **Run tests and linting**:
   ```bash
   pytest tests/ -v
   black src/ tests/ *.py
   isort src/ tests/ *.py
   flake8 src/ tests/ *.py
   mypy src/
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```
   
   Use conventional commit messages (see below)

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** on GitHub

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Imports**: Organized with isort
- **Formatting**: Automated with Black
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public modules, classes, and functions

### Code Formatting

We use automated tools to maintain consistent code style:

```bash
# Format code
black src/ tests/ *.py

# Sort imports
isort src/ tests/ *.py

# Lint code
flake8 src/ tests/ *.py

# Type check
mypy src/
```

Pre-commit hooks will automatically run these checks before each commit.

### Docstring Style

Use Google-style docstrings:

```python
def predict_properties(composition: Dict[str, float]) -> np.ndarray:
    """
    Predict mechanical properties from steel composition.
    
    Args:
        composition: Dictionary mapping element symbols to weight percentages.
            Example: {'C': 0.42, 'Mn': 0.75, 'Si': 0.30}
    
    Returns:
        Array of predicted properties: [yield_strength, UTS, elongation]
    
    Raises:
        InvalidCompositionError: If composition is physically invalid.
        ModelNotTrainedError: If model hasn't been trained yet.
    
    Example:
        >>> composition = {'C': 0.42, 'Mn': 0.75, 'Si': 0.30}
        >>> predictions = predict_properties(composition)
        >>> print(f"YS: {predictions[0]:.1f} MPa")
    """
    pass
```

### Commit Message Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(inverse-design): add cost-aware optimization
fix(pcnn): correct physics constraint calculation
docs(readme): update installation instructions
test(utils): add tests for composition validation
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pcnn_model.py -v

# Run specific test
pytest tests/test_pcnn_model.py::test_forward_pass -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage

Example test:

```python
import pytest
from src.pcnn_model import SteelPropertyPredictor

def test_model_initialization():
    """Test that model initializes correctly."""
    predictor = SteelPropertyPredictor(input_dim=50)
    assert predictor.model is not None
    assert predictor.device is not None

def test_prediction_shape():
    """Test that predictions have correct shape."""
    predictor = SteelPropertyPredictor(input_dim=50)
    # ... setup and prediction
    assert predictions.shape == (batch_size, 3)
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run all tests** and ensure they pass
4. **Run linting** and fix any issues
5. **Update CHANGELOG** if applicable
6. **Rebase on latest main** if needed:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests passing
- [ ] No linting errors
```

### Review Process

1. **Automated checks** must pass (tests, linting, type checking)
2. **Code review** by at least one maintainer
3. **Address feedback** and make requested changes
4. **Approval** from maintainer
5. **Merge** by maintainer

## Reporting Bugs

### Before Reporting

1. **Check existing issues** to avoid duplicates
2. **Try latest version** to see if bug is already fixed
3. **Gather information** about the bug

### Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- SteelML version: [e.g., 1.0.0]

**Additional context**
Any other relevant information
```

## Suggesting Enhancements

### Enhancement Proposal Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of desired functionality

**Describe alternatives you've considered**
Alternative solutions or features

**Additional context**
Any other relevant information, mockups, examples, etc.
```

## Development Tips

### Project Structure

- `src/`: Source code
- `tests/`: Test files
- `data/`: Dataset storage
- `models/`: Trained models
- `plots/`: Generated visualizations
- `logs/`: Application logs

### Useful Commands

```bash
# Run dashboard locally
streamlit run app.py

# Generate dataset
python generate_dataset.py

# Train models
python train_model.py

# Format and lint
black src/ tests/ *.py && isort src/ tests/ *.py && flake8 src/ tests/ *.py

# Type check
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Debugging

- Use logging instead of print statements
- Set log level to DEBUG for detailed output
- Use pytest's `-s` flag to see print output during tests
- Use `breakpoint()` for interactive debugging

## Questions?

If you have questions:

1. **Check documentation** in `docs/` directory
2. **Search existing issues** on GitHub
3. **Ask in discussions** on GitHub Discussions
4. **Contact maintainers** via email

## License

By contributing to SteelML, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to SteelML! 🙏
