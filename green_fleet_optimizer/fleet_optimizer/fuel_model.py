"""ML prediction of fuel consumption, sailing time, delay and on-time probability."""

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import FEATURES, feature_matrix

try:
    from xgboost import XGBRegressor
except ImportError:  # optional dependency
    XGBRegressor = None


def candidate_models(seed=42):
    models = {
        "Ridge (linear baseline)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1,
                                               random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                       max_depth=4, subsample=0.9, random_state=seed),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                         subsample=0.9, colsample_bytree=0.9, random_state=seed)
    # Fuel spans two orders of magnitude, so learn it in log space.
    return {name: TransformedTargetRegressor(m, func=np.log1p, inverse_func=np.expm1)
            for name, m in models.items()}


class FleetPredictor:
    """Trains the fuel + delay models and exposes batched voyage predictions."""

    def __init__(self, seed=42):
        self.seed = seed
        self.fuel_model = None
        self.fuel_model_name = None
        self.delay_model = None
        self.metrics = None
        self.delay_residuals = None
        self.test_frame = None

    def fit(self, history: pd.DataFrame):
        X = feature_matrix(history)
        y = history["fuel_t"].values
        X_tr, X_te, y_tr, y_te, h_tr, h_te = train_test_split(
            X, y, history, test_size=0.2, random_state=self.seed)

        rows, fitted = [], {}
        for name, model in candidate_models(self.seed).items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            fitted[name] = (model, pred)
            rows.append({"model": name, "R2": r2_score(y_te, pred),
                         "MAE_t": mean_absolute_error(y_te, pred),
                         "MAPE_%": 100 * mean_absolute_percentage_error(y_te, pred)})
        self.metrics = pd.DataFrame(rows).sort_values("MAPE_%").reset_index(drop=True)
        self.fuel_model_name = self.metrics.iloc[0]["model"]
        self.fuel_model, best_pred = fitted[self.fuel_model_name]
        self.test_frame = pd.DataFrame({"actual_fuel_t": y_te, "predicted_fuel_t": best_pred,
                                        "vessel_type": h_te["vessel_type"].values})

        self.delay_model = GradientBoostingRegressor(n_estimators=250, max_depth=3,
                                                     learning_rate=0.05, random_state=self.seed)
        self.delay_model.fit(X_tr, h_tr["delay_hr"])
        self.delay_residuals = np.sort(h_te["delay_hr"].values - self.delay_model.predict(X_te))
        self.delay_mae = mean_absolute_error(h_te["delay_hr"], self.delay_model.predict(X_te))
        return self

    def feature_importance(self) -> pd.Series:
        reg = self.fuel_model.regressor_
        est = reg[-1] if hasattr(reg, "steps") else reg
        if hasattr(est, "feature_importances_"):
            vals = est.feature_importances_
        else:
            vals = np.abs(est.coef_)
        return pd.Series(vals, index=FEATURES).sort_values(ascending=False)

    def predict(self, voyages: pd.DataFrame) -> pd.DataFrame:
        """voyages needs vessel spec columns + speed_kn, load_frac, distance_nm and weather.
        Returns fuel (t VLSFO-equivalent), sailing hours and expected delay."""
        X = feature_matrix(voyages)
        fuel = np.maximum(self.fuel_model.predict(X), 0.0)
        delay = np.maximum(self.delay_model.predict(X), 0.0)
        hours = X["sailing_hr_est"].values
        return pd.DataFrame({"fuel_t": fuel, "sailing_hr": hours, "delay_hr": delay},
                            index=voyages.index)

    def on_time_probability(self, slack_hr):
        """P(actual delay <= slack), using the empirical residual distribution of the delay model."""
        slack = np.atleast_1d(slack_hr)
        return np.searchsorted(self.delay_residuals, slack, side="right") / len(self.delay_residuals)
