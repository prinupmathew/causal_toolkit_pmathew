from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _as_list(values: Iterable[str]) -> List[str]:
    return list(values)


def _fit_propensity(train: pd.DataFrame, X: List[str], T: str) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    model.fit(train[X], train[T])
    return model


def s_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    features = _as_list(X) + [T]
    model = LGBMRegressor()
    model.fit(train[features], train[y])

    base = test[_as_list(X)].copy()
    test_treated = base.copy()
    test_treated[T] = 1
    test_control = base.copy()
    test_control[T] = 0

    mu1 = model.predict(test_treated[features])
    mu0 = model.predict(test_control[features])

    result = test.copy()
    result["cate"] = mu1 - mu0
    return result


def t_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    model_treated = LGBMRegressor()
    model_control = LGBMRegressor()
    model_treated.fit(train.loc[mask_treated, X], train.loc[mask_treated, y])
    model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

    mu1 = model_treated.predict(test[X])
    mu0 = model_control.predict(test[X])

    result = test.copy()
    result["cate"] = mu1 - mu0
    return result


def x_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    model_treated = LGBMRegressor()
    model_control = LGBMRegressor()
    model_treated.fit(train.loc[mask_treated, X], train.loc[mask_treated, y])
    model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

    mu1_on_control = model_treated.predict(train.loc[mask_control, X])
    mu0_on_treated = model_control.predict(train.loc[mask_treated, X])

    tau0 = mu1_on_control - train.loc[mask_control, y].to_numpy()
    tau1 = train.loc[mask_treated, y].to_numpy() - mu0_on_treated

    tau0_model = LGBMRegressor()
    tau1_model = LGBMRegressor()
    tau0_model.fit(train.loc[mask_control, X], tau0)
    tau1_model.fit(train.loc[mask_treated, X], tau1)

    propensity_model = _fit_propensity(train, X, T)
    e_test = propensity_model.predict_proba(test[X])[:, 1]

    tau0_hat = tau0_model.predict(test[X])
    tau1_hat = tau1_model.predict(test[X])

    result = test.copy()
    result["cate"] = e_test * tau0_hat + (1 - e_test) * tau1_hat
    return result


def double_ml_cate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    X_train = train[X]
    treatment_train = train[T].to_numpy()
    outcome_train = train[y].to_numpy()

    treatment_hat = np.zeros(len(train))
    outcome_hat = np.zeros(len(train))

    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    for train_idx, hold_idx in kf.split(X_train):
        model_treatment = LGBMRegressor()
        model_outcome = LGBMRegressor()
        X_train_fold = X_train.iloc[train_idx]
        X_holdout_fold = X_train.iloc[hold_idx]

        model_treatment.fit(X_train_fold, treatment_train[train_idx])
        model_outcome.fit(X_train_fold, outcome_train[train_idx])

        treatment_hat[hold_idx] = model_treatment.predict(X_holdout_fold)
        outcome_hat[hold_idx] = model_outcome.predict(X_holdout_fold)

    treatment_residual = treatment_train - treatment_hat
    outcome_residual = outcome_train - outcome_hat

    epsilon = 1e-6
    safe_treatment_residual = np.where(
        np.abs(treatment_residual) < epsilon,
        epsilon,
        treatment_residual,
    )
    pseudo_outcome = outcome_residual / safe_treatment_residual
    weights = treatment_residual**2

    tau_model = LGBMRegressor()
    tau_model.fit(X_train, pseudo_outcome, sample_weight=weights)

    result = test.copy()
    result["cate"] = tau_model.predict(test[X])
    return result