# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `signal-engine-core`
3. Description: "Shared ML/Governance/Validation engine for multi-asset signal generation platforms"
4. Visibility: **Private** (or Public if open-sourcing)
5. **DO NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## Step 2: Add Remote and Push

```bash
cd /c/Users/"North East Collision"/Desktop/signal-engine-core

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/signal-engine-core.git

# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/signal-engine-core.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 3: Configure Repository Settings

### Branch Protection (Recommended)
1. Go to Settings → Branches
2. Add rule for `main` branch:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging

### Secrets (For CI/CD)
If publishing to PyPI:
1. Settings → Secrets and variables → Actions
2. Add `PYPI_API_TOKEN` secret

### Topics (For Discoverability)
Add topics: `machine-learning`, `signal-processing`, `trading`, `python`, `abstraction`

## Step 4: Create GitHub Actions Workflow (Optional)

Create `.github/workflows/python-package.yml`:

```yaml
name: Python Package

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ -v --cov=signal_engine

    - name: Lint
      run: |
        flake8 signal_engine/
        black --check signal_engine/

  publish:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags')

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine

    - name: Build package
      run: python -m build

    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

## Step 5: Verify Setup

```bash
# Check remote
git remote -v

# Check GitHub connection
git ls-remote origin
```

## Ready!

Your signal-engine-core is now on GitHub and ready for:
- Module extraction from AlphaEngine and MoonWire
- CI/CD pipeline
- Package publishing
- Collaboration

## Next Steps

See `MIGRATION_GUIDE.md` for how to extract modules from the product repos.
