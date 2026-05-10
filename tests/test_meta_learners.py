from __future__ import annotations

import numpy as np
import pandas as pd

from causal_toolkit_pmathew.meta_learners import (
    double_ml_cate,
    s_learner_discrete,
    t_learner_discrete,
    x_learner_discrete,
)


def simple_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    n = 1000

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    prob_t = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2)))
    t = np.random.binomial(1, prob_t, n)

    y = 2.0 * t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})
    return df.iloc[:800].copy(), df.iloc[800:].copy()


def heterogeneous_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(123)
    n = 1500

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    prob_t = 1 / (1 + np.exp(-(0.4 * x1)))
    t = np.random.binomial(1, prob_t, n)

    treatment_effect = 1.0 + 0.5 * x1
    y = treatment_effect * t + x1 + 0.3 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})
    return df.iloc[:1200].copy(), df.iloc[1200:].copy()


def continuous_treatment_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(789)
    n = 1000

    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)

    t = 10 + x1 + 2 * x2 + np.random.normal(0, 1, n)
    y = t + x1 + 0.5 * x2 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})
    return df.iloc[:800].copy(), df.iloc[800:].copy()


def _assert_valid_cate_frame(df: pd.DataFrame, expected_len: int) -> None:
    assert isinstance(df, pd.DataFrame)
    assert "cate" in df.columns
    assert len(df) == expected_len
    assert np.isfinite(df["cate"]).all()


def test_meta_learners_constant_effect() -> None:
    train, test = simple_data()

    s_result = s_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    t_result = t_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    x_result = x_learner_discrete(train, test, ["x1", "x2"], "t", "y")

    _assert_valid_cate_frame(s_result, len(test))
    _assert_valid_cate_frame(t_result, len(test))
    _assert_valid_cate_frame(x_result, len(test))

    assert abs(float(s_result["cate"].mean()) - 2.0) <= 0.6
    assert abs(float(t_result["cate"].mean()) - 2.0) <= 0.6
    assert abs(float(x_result["cate"].mean()) - 2.0) <= 0.6


def test_meta_learners_track_heterogeneous_effects() -> None:
    train, test = heterogeneous_data()
    true_cate = 1.0 + 0.5 * test["x1"].to_numpy()

    t_result = t_learner_discrete(train, test, ["x1", "x2"], "t", "y")
    x_result = x_learner_discrete(train, test, ["x1", "x2"], "t", "y")

    t_corr = float(np.corrcoef(t_result["cate"].to_numpy(), true_cate)[0, 1])
    x_corr = float(np.corrcoef(x_result["cate"].to_numpy(), true_cate)[0, 1])

    assert t_corr > 0.3
    assert x_corr > 0.3


def test_double_ml_handles_continuous_treatment() -> None:
    train, test = continuous_treatment_data()

    result = double_ml_cate(train, test, ["x1", "x2"], "t", "y")

    _assert_valid_cate_frame(result, len(test))
    assert abs(float(result["cate"].mean()) - 1.0) <= 0.4
