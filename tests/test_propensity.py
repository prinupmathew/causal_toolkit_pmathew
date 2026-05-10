from __future__ import annotations

import numpy as np
import pandas as pd

from causal_toolkit_pmathew.propensity import doubly_robust, ipw


def _make_positive_effect_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    x = np.random.normal(0, 1, n)
    prob_t = 1 / (1 + np.exp(-(0.5 * x)))
    t = np.random.binomial(1, prob_t, n)
    y = 2 * t + x + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"x": x, "t": t, "y": y})


def _make_categorical_data() -> pd.DataFrame:
    np.random.seed(101)
    n = 1000
    group = np.random.choice(["A", "B", "C"], n)
    group_effect = {"A": 0, "B": 1, "C": 2}
    x_numeric = np.array([group_effect[value] for value in group])

    prob_t = 1 / (1 + np.exp(-(0.5 * x_numeric)))
    t = np.random.binomial(1, prob_t, n)
    y = 2.0 * t + x_numeric + np.random.normal(0, 0.5, n)
    return pd.DataFrame({"group": group, "t": t, "y": y})


def test_ipw_and_dr_numeric_covariate() -> None:
    df = _make_positive_effect_data()

    ipw_estimate = ipw(df, "x", "t", "y")
    dr_estimate = doubly_robust(df, "x", "t", "y")

    assert abs(ipw_estimate - 2.0) <= 0.3
    assert abs(dr_estimate - 2.0) <= 0.2


def test_ipw_and_dr_categorical_covariate() -> None:
    df = _make_categorical_data()

    ipw_estimate = ipw(df, "C(group)", "t", "y")
    dr_estimate = doubly_robust(df, "C(group)", "t", "y")

    assert abs(ipw_estimate - 2.0) <= 0.3
    assert abs(dr_estimate - 2.0) <= 0.2
