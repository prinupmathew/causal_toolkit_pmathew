# causal_toolkit_pmathew

[![Tests](https://github.com/prinupmathew/causal_toolkit_pmathew/workflows/Tests/badge.svg)](https://github.com/prinupmathew/causal_toolkit_pmathew/actions)
[![Publish to PyPI](https://github.com/prinupmathew/causal_toolkit_pmathew/actions/workflows/publish.yml/badge.svg)](https://github.com/prinupmathew/causal_toolkit_pmathew/actions/workflows/publish.yml)
[![PyPI version](https://badge.fury.io/py/causal-toolkit-pmathew.svg)](https://pypi.org/project/causal-toolkit-pmathew/)

`causal_toolkit_pmathew` packages the course implementations from Weeks 02 through 05 into an installable causal inference toolkit.

## Package contents

- `causal_toolkit_pmathew/rct.py`: randomized controlled trial estimators from Week 02
- `causal_toolkit_pmathew/propensity.py`: inverse propensity weighting and doubly robust estimators from Week 03
- `causal_toolkit_pmathew/meta_learners.py`: S-, T-, X-learner, and double machine learning CATE estimators from Weeks 04 and 05
- `tests/`: pytest coverage for the required modules

## Installation

```bash
# Install from PyPI
pip install causal_toolkit_pmathew

# Or install from source
git clone https://github.com/prinupmathew/causal_toolkit_pmathew.git
cd causal_toolkit_pmathew
uv pip install -e .
python -m uv pip install -e .
```

## Run tests

```bash
uv run pytest
python -m uv run pytest
```

## Run pytest module with coverage

```bash
python -m pytest tests/ -v --cov=causal_toolkit_pmathew --cov-report=term-missing
```

## Publish to PyPI

- Create accounts:
- Generate API tokens:
	- Go to Account Settings → API tokens → Add token
	- Scope: "Entire account"
	- Save token securely (starts with pypi-)
- Publish to PyPI:

```bash
python -m pip install --upgrade twine build

# Set pypi credential locally in terminal (cmd)
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=pypi-your-token-here

# Set pypi credential locally in terminal (bash)
export TWINE_USERNAME="__token__"
export TWINE_PASSWORD="pypi-your-token-here"

# Set pypi credential locally in terminal (powershell)
$env:TWINE_USERNAME="__token__"
$env:TWINE_PASSWORD="pypi-your-token-here"

# Verify tests pass
pytest tests/ -v

# Clean and rebuild
python -m build
python -m twine check dist/*
python -m twine upload dist/*

```

## Automated publishing (GitHub Release -> PyPI)

1. Add repository secret `PYPI_API_TOKEN` in GitHub Actions settings.
2. Create and publish a GitHub Release (for example tag `v0.1.1`).
3. GitHub Actions workflow `publish.yml` builds and uploads to PyPI automatically.

## Usage

```python
import pandas as pd

from causal_toolkit_pmathew import (
	calculate_ate_ci,
	doubly_robust,
	ipw,
	s_learner_discrete,
	t_learner_discrete,
)

trial_data = pd.DataFrame({
	"T": [1, 1, 0, 0],
	"Y": [11.2, 10.8, 8.9, 9.1],
})

ate, ci_lower, ci_upper = calculate_ate_ci(trial_data)

observational_data = pd.DataFrame({
	"x": [0.2, 0.7, -0.1, 1.3],
	"t": [1, 1, 0, 0],
	"y": [3.4, 2.8, 1.1, 1.6],
})

ipw_estimate = ipw(observational_data, "x", "t", "y")
dr_estimate = doubly_robust(observational_data, "x", "t", "y")

train = observational_data.rename(columns={"x": "x1"}).assign(x2=[1.0, 0.5, 0.1, -0.2])
test = train.copy()

s_cate = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")
t_cate = t_learner_discrete(train, test, ["x1", "x2"], "t", "y")
```

## API

- `calculate_ate_ci(data, alpha=0.05)`: estimate the average treatment effect and a two-sided confidence interval from an RCT dataset with `T` and `Y` columns
- `calculate_ate_pvalue(data)`: estimate the ATE, test statistic, and two-sided p-value for an RCT dataset with `T` and `Y` columns
- `ipw(df, ps_formula, T, Y)`: compute an inverse propensity weighted ATE
- `doubly_robust(df, formula, T, Y)`: compute a doubly robust ATE using propensity and outcome models
- `s_learner_discrete(train, test, X, T, y)`: estimate CATE with a single outcome model
- `t_learner_discrete(train, test, X, T, y)`: estimate CATE with separate treated and control models
- `x_learner_discrete(train, test, X, T, y)`: estimate CATE with the X-learner procedure
- `double_ml_cate(train, test, X, T, y)`: estimate CATE using residualized treatment and outcome models
