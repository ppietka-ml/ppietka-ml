"""
Sales demand forecasting pipeline for electronics distributor.
Combines LightGBM with time-series features for SKU-level predictions.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error


@dataclass
class ForecastResult:
    sku_id: str
    horizon_days: int
    predictions: list
    mape: float
    lower_bound: list
    upper_bound: list


def create_lag_features(df: pd.DataFrame, lags: list = [7, 14, 21, 28]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"sales_lag_{lag}"] = df["sales"].shift(lag)
    df["sales_rolling_7"] = df["sales"].rolling(7).mean()
    df["sales_rolling_28"] = df["sales"].rolling(28).mean()
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df.dropna()


class SalesForecastingPipeline:
    def __init__(self, horizon: int = 14):
        self.horizon = horizon
        self.model = None
        self.params = {
            "objective": "regression",
            "metric": "mape",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

    def fit(self, sales_series: pd.Series) -> dict:
        df = pd.DataFrame({"sales": sales_series})
        df = create_lag_features(df)
        feature_cols = [c for c in df.columns if c != "sales"]
        X, y = df[feature_cols], df["sales"]
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx], reference=dtrain)
            m = lgb.train(self.params, dtrain, 300, valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds = m.predict(X.iloc[val_idx])
            scores.append(mean_absolute_percentage_error(y.iloc[val_idx], preds))
        dtrain_full = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, dtrain_full, 300)
        return {"cv_mape": round(np.mean(scores), 4)}

    def predict(self, sku_id: str, recent_sales: pd.Series) -> ForecastResult:
        df = pd.DataFrame({"sales": recent_sales})
        df = create_lag_features(df)
        feature_cols = [c for c in df.columns if c != "sales"]
        preds = self.model.predict(df[feature_cols].tail(self.horizon))
        std_est = recent_sales.std() * 0.1
        return ForecastResult(
            sku_id=sku_id,
            horizon_days=self.horizon,
            predictions=preds.tolist(),
            mape=0.0,
            lower_bound=(preds - 1.96 * std_est).tolist(),
            upper_bound=(preds + 1.96 * std_est).tolist(),
        )
