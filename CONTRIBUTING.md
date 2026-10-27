# Contributing Guide

Thank you for contributing to the MLOps Fraud Detection Platform! This guide will help you get started.

## 🚀 Development Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Docker & Docker Compose
- Git
- Pre-commit (installed via `make pre-commit`)

### Quick Start
```bash
# Fork and clone
git clone https://github.com/your-username/mlops-fraud-detection.git
cd mlops-fraud-detection

# Install dependencies
make install

# Install pre-commit hooks
make pre-commit

# Verify setup
make check
make test
```

## 📝 Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/) with commitlint validation.

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, etc.) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `build` | Build system changes |
| `ci` | CI/CD changes |
| `chore` | Maintenance tasks |
| `revert` | Revert commit |

### Examples
```bash
# Feature
git commit -m "feat(train): add XGBoost early stopping callback"

# Bug fix
git commit -m "fix(serve): handle missing features in prediction input"

# Documentation
git commit -m "docs: update architecture diagram with drift detection"

# Refactor
git commit -m "refactor(common): extract metrics calculation to utils"

# CI
git commit -m "ci: add retrain workflow with drift trigger"
```

### Scopes
- `train` - Training pipeline
- `serve` - FastAPI serving
- `monitor` - Drift detection
- `common` - Shared utilities
- `docker` - Dockerfiles & compose
- `ci` - GitHub Actions
- `docs` - Documentation
- `config` - Configuration files

## 🔄 Pull Request Process

### 1. Branch Naming
```
<type>/<short-description>
# Examples:
feat/add-lightgbm-support
fix/prediction-threshold-handling
docs/update-runbooks
```

### 2. PR Requirements
- [ ] All tests pass (`make test`)
- [ ] Code quality checks pass (`make check`)
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts with `main`
- [ ] Descriptive PR title (conventional commit format)
- [ ] PR description explains *what* and *why*
- [ ] Screenshots for UI changes (Grafana dashboards, etc.)

### 3. PR Template
```markdown
## Description
Brief description of changes

## Type
- [ ] Feature
- [ ] Bug Fix
- [ ] Documentation
- [ ] Refactor
- [ ] CI/CD
- [ ] Other

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing done (describe)

## Checklist
- [ ] `make check` passes
- [ ] `make test` passes
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
```

### 4. Review Process
1. Automated checks must pass (CI)
2. At least 1 approval required
3. No blocking comments unresolved
4. Squash and merge to `main`

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
└── contract/       # API contract tests
```

### Writing Tests
```python
# tests/unit/test_utils.py
import pytest
from src.common.utils import calculate_classification_metrics

def test_calculate_classification_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 1, 0, 0]
    y_proba = [0.1, 0.9, 0.8, 0.2, 0.4]
    
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    
    assert metrics["accuracy"] == 0.8
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 2/3
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
```

### Running Tests
```bash
# All tests
make test

# Specific test file
uv run pytest tests/unit/test_utils.py -v

# With coverage
uv run pytest --cov=src --cov-report=term-missing

# Watch mode (dev)
uv run pytest-watch
```

## 🎨 Code Style

### Formatting & Linting
```bash
# Format code
make format

# Check formatting
uv run black --check src tests

# Lint
make lint

# Type check
make typecheck

# All checks
make check
```

### Standards
- **Line length**: 100 chars (Black/Ruff)
- **Quotes**: Double quotes (Black)
- **Imports**: Sorted by Ruff (isort)
- **Type hints**: Required for public functions (MyPy)
- **Docstrings**: Google style for public APIs

### Ruff Rules
Key rules enforced:
- `E, W` - pycodestyle
- `F` - pyflakes
- `I` - isort
- `B` - flake8-bugbear
- `C4` - flake8-comprehensions
- `UP` - pyupgrade
- `ARG` - unused arguments
- `PTH` - pathlib usage
- `TID` - tidy imports

## 🐳 Docker Development

### Building Images
```bash
# Build all
make docker-build

# Build specific
docker build -f docker/Dockerfile.serve -t fraud-api .
docker build -f docker/Dockerfile.train -t fraud-train .
docker build -f docker/Dockerfile.monitor -t fraud-monitor .
```

### Running Services
```bash
# Full stack
make docker-up

# Specific services
docker compose -f docker/docker-compose.yml up -d minio mlflow api

# With logs
make docker-logs
```

## 📊 Adding New Metrics

1. **Update `params.yaml`** - Add to `deployment_gates`
2. **Update `train.py`** - Calculate in `evaluate_model()`
3. **Update `utils.py`** - Add to `calculate_classification_metrics()`
4. **Update MLflow logging** - Log in `log_to_mlflow()`
5. **Update gate check** - CI workflow reads `metrics/train.json`
6. **Add Grafana panel** - Dashboard JSON

## 🔧 Adding New Features

### New Model Type
1. Add to `params.yaml` under `training.model_type` options
2. Update `create_model()` in `train.py`
3. Add dependencies to `pyproject.toml`
4. Test with `make train`

### New API Endpoint
1. Add Pydantic model in `serve/main.py`
2. Add endpoint with proper validation
3. Add Prometheus metrics
4. Add contract test in `tests/contract/`
5. Update OpenAPI docs (auto-generated)

### New Drift Check
1. Add method to `DriftDetector` in `monitor/drift.py`
2. Add threshold to `params.yaml` under `monitoring`
3. Update `run_drift_checks()` to include new check
4. Add Grafana panel for visualization
5. Add alert rule in `prometheus/rules.yml`

## 📦 Dependency Management

### Adding Dependencies
```bash
# Production dependency
uv add package-name

# Dev dependency
uv add --dev package-name

# Optional dependency group
uv add --optional train package-name
```

### Lock File
- `uv.lock` is committed
- Run `uv lock` after changes
- CI uses `--frozen` flag

## 🚢 Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- `MAJOR` - Breaking changes
- `MINOR` - New features (backward compatible)
- `PATCH` - Bug fixes

### Release Steps
```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG.md
# 3. Commit
git commit -m "chore: release v1.2.0"

# 4. Tag
git tag v1.2.0
git push origin main --tags

# 5. GitHub Actions CD workflow triggers
# 6. Creates GitHub Release
# 7. Builds & pushes Docker images
# 8. Deploys to production (manual approval)
```

## 🐛 Issue Reporting

### Bug Report Template
```markdown
**Describe the bug**
Clear description

**To Reproduce**
Steps to reproduce

**Expected behavior**
What should happen

**Environment**
- OS:
- Python version:
- Docker version:
- Branch/commit:

**Logs/Screenshots**
If applicable
```

### Feature Request Template
```markdown
**Is your feature request related to a problem?**
Describe the problem

**Describe the solution you'd like**
Clear description

**Describe alternatives considered**
Other approaches

**Additional context**
Screenshots, mockups, etc.
```

## 📚 Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)

## 🤝 Code of Conduct

Be respectful, inclusive, and constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.