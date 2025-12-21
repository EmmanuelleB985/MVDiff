# Contributing to MVDiff

Thank you for your interest in contributing to MVDiff! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Documentation](#documentation)
- [Community](#community)


## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/EmmanuelleB985/mvdiff.git
   cd mvdiff
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/mvdiff/mvdiff.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

#### Bug Reports
- Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include a minimal reproducible example
- Provide system information and error messages

#### Feature Requests
- Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the problem you're trying to solve
- Describe your proposed solution

#### Code Contributions
- Bug fixes
- New features
- Performance improvements
- Refactoring
- Tests

#### Documentation
- Improving existing documentation
- Writing tutorials
- Creating examples
- Translating documentation

## Development Setup

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU support)
- Git

### Installation

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install in development mode**:
   ```bash
   make install-dev
   # Or manually:
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Make changes** in your feature branch
2. **Add tests** for new functionality
3. **Run tests** to ensure everything works:
   ```bash
   make test
   ```
4. **Format code**:
   ```bash
   make format
   ```
5. **Check linting**:
   ```bash
   make lint
   ```
6. **Commit changes** with descriptive message
7. **Push** to your fork
8. **Open Pull Request**

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:
- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Use docstrings for all public functions/classes

### Code Formatting

We use the following tools (automatically run via pre-commit):
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Example:
```python
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ExampleModule(nn.Module):
    """Example module with proper documentation.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden dimensions
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        return x
```

## Testing Guidelines

### Writing Tests

1. **Location**: Place tests in `tests/` directory
2. **Naming**: Use `test_*.py` for test files
3. **Structure**: Mirror source code structure

Example test:
```python
import pytest
import torch

from mvdiff.models import MVDiff


class TestMVDiff:
    """Test MVDiff model."""
    
    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return MVDiff(img_size=64)
    
    def test_forward_shape(self, model):
        """Test output shape."""
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64)
        output = model(x)
        assert output.shape == (batch_size, 3, 64, 64)
    
    @pytest.mark.gpu
    def test_cuda_forward(self, model):
        """Test CUDA forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = model.cuda()
        x = torch.randn(1, 3, 64, 64).cuda()
        output = model(x)
        assert output.is_cuda
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_models.py

# Run with coverage
make coverage

# Run GPU tests
make test-gpu
```

### Test Coverage

We aim for >80% test coverage. Check coverage with:
```bash
make coverage
```

## Pull Request Process

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Ensure all tests pass**:
   ```bash
   make check-all
   ```

3. **Update documentation** if needed

4. **Open Pull Request** with:
   - Clear title and description
   - Link to related issues
   - Screenshots/videos for UI changes
   - Performance benchmarks for optimizations

5. **Address review comments**

6. **Squash commits** if requested

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All checks passing
```

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Add type hints
- Update README if needed

### API Documentation

```python
def generate_views(
    input_image: torch.Tensor,
    num_views: int = 16,
    num_steps: int = 50,
    guidance_scale: float = 3.0,
    **kwargs
) -> torch.Tensor:
    """Generate multiple views from a single image.
    
    Args:
        input_image: Input image tensor of shape [B, C, H, W]
        num_views: Number of views to generate (default: 16)
        num_steps: Number of diffusion steps (default: 50)
        guidance_scale: Guidance scale for generation (default: 3.0)
        **kwargs: Additional generation parameters
        
    Returns:
        Generated views tensor of shape [B, num_views, C, H, W]
        
    Raises:
        ValueError: If input_image has incorrect shape
        RuntimeError: If GPU runs out of memory
        
    Examples:
        >>> model = MVDiff()
        >>> image = torch.randn(1, 3, 256, 256)
        >>> views = model.generate_views(image, num_views=8)
        >>> assert views.shape == (1, 8, 3, 256, 256)
    """
```

## Areas for Contribution

- [ ] Optimize memory usage in attention modules
- [ ] Add support for variable resolution inputs
- [ ] Implement additional 3D reconstruction methods
- [ ] Improve inference speed


### Feature Ideas
- [ ] Web API endpoint
- [ ] Mobile optimization
- [ ] Additional datasets support
- [ ] Multi-GPU training improvements

## Recognition

Contributors will be:
- Listed in [CONTRIBUTORS.md](CONTRIBUTORS.md)
- Mentioned in release notes
- Given credit in relevant documentation

## Development Metrics

We track:
- Code coverage (target: >80%)
- Documentation coverage (target: 100%)
- Performance benchmarks


## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make MVDiff better!