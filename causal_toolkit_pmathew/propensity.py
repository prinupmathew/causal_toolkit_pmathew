from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LinearRegression, LogisticRegression


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _design_matrix(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    return dmatrix(formula, df, return_type="dataframe")


def _fit_propensity_model(X: pd.DataFrame, treatment: np.ndarray) -> LogisticRegression:
    base_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=False,
    )
    if base_model.penalty is None:
        model = base_model
    else:
        model = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=1000,
            fit_intercept=False,
        )
    model.fit(X, treatment)
    return model


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    _require_columns(df, [T, Y])

    X = _design_matrix(df, ps_formula)
    treatment = df[T].to_numpy(dtype=float)
    outcome = df[Y].to_numpy(dtype=float)

    model = _fit_propensity_model(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

    weights = (treatment - propensity_scores) / (propensity_scores * (1 - propensity_scores))
    ate = float(np.mean(weights * outcome))
    return ate


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    _require_columns(df, [T, Y])

    X = _design_matrix(df, formula)
    treatment = df[T].to_numpy(dtype=float)
    outcome = df[Y].to_numpy(dtype=float)

    model = _fit_propensity_model(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

    mask_treated = treatment == 1
    mask_control = treatment == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    model_treated = LinearRegression(fit_intercept=False)
    model_control = LinearRegression(fit_intercept=False)
    model_treated.fit(X.loc[mask_treated], outcome[mask_treated])
    model_control.fit(X.loc[mask_control], outcome[mask_control])

    mu1 = model_treated.predict(X)
    mu0 = model_control.predict(X)

    treated_term = treatment * (outcome - mu1) / propensity_scores + mu1
    control_term = (1 - treatment) * (outcome - mu0) / (1 - propensity_scores) + mu0
    ate = float(np.mean(treated_term - control_term))
    return ate