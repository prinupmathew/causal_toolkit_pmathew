from __future__ import annotations

import numpy as np
import pandas as pd

from causal_toolkit_pmathew.rct import calculate_ate_ci, calculate_ate_pvalue


def _make_positive_effect_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "I": range(n),
        "T": np.random.binomial(1, 0.5, n),
    })
    df["Y"] = np.where(
        df["T"] == 1,
        np.random.normal(10, 2, n),
        np.random.normal(8, 2, n),
    )
    return df


def _make_no_effect_data() -> pd.DataFrame:
    np.random.seed(123)
    n = 500
    df = pd.DataFrame({
        "I": range(n),
        "T": np.random.binomial(1, 0.5, n),
    })
    df["Y"] = np.random.normal(5, 3, n)
    return df


def test_rct_estimators_detect_positive_effect() -> None:
    df = _make_positive_effect_data()

    ate, lower, upper = calculate_ate_ci(df)
    ate_p, _, p_value = calculate_ate_pvalue(df)

    assert abs(ate - ate_p) < 1e-8
    assert lower < ate < upper
    assert lower > 0 or upper < 0
    assert p_value < 0.05


def test_rct_estimators_detect_null_effect() -> None:
    df = _make_no_effect_data()

    ate, lower, upper = calculate_ate_ci(df)
    ate_p, _, p_value = calculate_ate_pvalue(df)

    assert abs(ate - ate_p) < 1e-8
    assert lower <= 0 <= upper
    assert p_value > 0.05
